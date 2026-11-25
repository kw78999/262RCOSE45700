"""Lower a split torch.compile graph to a cached Rust RunnableSequence.

The simplified graph keeps only routing nodes around compiled runnable segments.
"""

import dataclasses
import functools
import operator
from typing import Any, NamedTuple, Optional

import torch
from torch.utils import _pytree as pytree

from furiosa.torch import _utils
from furiosa.torch._tensor import _get_dram_shape
from furiosa.torch._C import ir
from furiosa.torch.profiler import instrument_with_name
from furiosa.torch._C import module as _C_module
from furiosa.torch._module import CompileModule
from furiosa.torch.export._meta import is_runnable_node
from furiosa.torch.export.passthrough import Passthrough


# Symbolic tensor handles lowered to Rust node ids.
@dataclasses.dataclass(frozen=True)
class _Arg:
    index: int


@dataclasses.dataclass(frozen=True)
class _Const:
    tensor: torch.Tensor = dataclasses.field(compare=False)
    key: int  # id(tensor), for hashing/dedup


@dataclasses.dataclass(frozen=True)
class _Output:
    step: int
    out: int


@dataclasses.dataclass(frozen=True, eq=False)
class _Passthrough:
    src: Any
    # Passthrough records are not value-comparable; use identity equality.
    pair: Passthrough


@dataclasses.dataclass(frozen=True)
class _Zero:
    meta: torch.Tensor = dataclasses.field(compare=False)
    key: int


@dataclasses.dataclass(frozen=True)
class _Slice:
    src: Any
    offset: int
    dram_shape: Any = dataclasses.field(compare=False)
    key: int = 0


def _const(tensor: torch.Tensor) -> _Const:
    return _Const(tensor=tensor, key=id(tensor))


@dataclasses.dataclass
class _SequencePlan:
    seq: Any
    # Pytree spec of the original graph output (tuple/list/dict/namedtuple/nested),
    # with the runnable tensors as leaves; used to rebuild the output structure.
    out_spec: pytree.TreeSpec
    # Drops DFG tensors that were inlined before planning.
    filter_inputs: Optional[Any] = None


def _example_tensor(node: torch.fx.Node) -> torch.Tensor:
    meta = node.meta
    if "val" in meta:
        return meta["val"]
    if "example_value" in meta:
        return meta["example_value"]
    raise KeyError(f"node {node} has no 'val'/'example_value' meta")


def _is_slice_node(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    target = node.target
    op = target.overloadpacket if isinstance(target, torch._ops.OpOverload) else target
    return op == torch.ops.aten.slice


def _handle_of(env: dict, node: Any) -> Any:
    if not isinstance(node, torch.fx.Node):
        raise NotImplementedError(f"non-Node graph value {node!r} is unsupported")
    handles = env[node]
    assert len(handles) == 1, f"{node} produces {len(handles)} values, expected 1"
    return handles[0]


class _StepSpec(NamedTuple):
    """A lowered runnable step, emitted by both step builders."""

    binary: torch.Tensor
    weight: Optional[torch.Tensor]
    stacks: Any  # Sequence[torch.Tensor] of reserved scratch tensors
    params: Any  # flattened runnable params (Sequence[int])
    runnable: Any  # ir.Runnable kept for profile spans, or None
    inputs: list  # flattened input handles


def _step_compile_module(
    cm: CompileModule,
    step: int,
    logical_inputs: list,
) -> tuple[_StepSpec, list]:
    inner = cm.runnable._inner

    # `flatten_inputs` may insert weights and drop zero-numel args; recover its
    # order by matching tagged examples by identity.
    weights = cm.module_weight.weights
    tagged_args = tuple(cm._segment_example_inputs)
    assert len(tagged_args) == len(logical_inputs)
    flat = cm.module_weight.flatten_inputs(tagged_args, {})
    id_to_handle: dict[int, Any] = {}
    for w in weights:
        id_to_handle[id(w)] = _const(w)
    for tagged, handle in zip(tagged_args, logical_inputs):
        id_to_handle[id(tagged)] = handle
    flat_input_handles = [id_to_handle[id(t)] for t in flat]

    num_outputs = len(inner.output_dram_shapes)
    raw = [_Output(step, j) for j in range(num_outputs)]

    reconstructed = cm.output_reconstructor.reconstruct_outputs(
        raw,
        lambda pair: _Passthrough(src=flat_input_handles[pair.input_idx], pair=pair),
        lambda meta: _Zero(meta=meta, key=id(meta)),
    )

    # Keep the ir.Runnable (``inner``) for profile spans; stacks reserved at load
    # time are passed explicitly, like binary/weight.
    spec = _StepSpec(
        binary=inner.binary,
        weight=inner.weight,
        stacks=inner.stacks,
        params=list(inner._flattened_params),
        runnable=inner,
        inputs=flat_input_handles,
    )
    return spec, reconstructed


def _flatten_submodule(
    sub: torch.fx.GraphModule,
    step: int,
    logical_inputs: list,
    args: tuple,
) -> tuple[Optional[_StepSpec], list]:
    """Lower a split submodule to a runnable spec and logical output handles.

    View-only submodules return ``spec=None``.
    """
    placeholders = sub.graph.find_nodes(op="placeholder")
    ph_index = {p: i for i, p in enumerate(placeholders)}
    op_nodes = [n for n in sub.graph.nodes if is_runnable_node(n)]
    assert len(op_nodes) <= 1, "a submodule holds at most one runnable op"
    op = op_nodes[0] if op_nodes else None

    num_outputs = 0
    if op is not None:
        inputs_list, binary_node, weight_node, stacks_list, params = op.args
        params = list(params)
        num_outputs = len(ir.Runnable._output_shapes_from_params(params))

    def node_handle(inner):
        if op is not None:
            if inner is op:
                return _Output(step, 0)
            if (
                inner.op == "call_function"
                and inner.target is operator.getitem
                and inner.args[0] is op
            ):
                return _Output(step, inner.args[1])
        if _is_slice_node(inner):
            return _slice_handle(inner, node_handle(inner.args[0]))
        if inner.op == "placeholder":
            return logical_inputs[ph_index[inner]]
        if inner.op == "get_attr":
            attr = functools.reduce(getattr, inner.target.split("."), sub)
            return _const(attr)
        raise NotImplementedError(f"unsupported submodule node {inner.op} {inner}")

    def resolve_const(handle) -> Optional[torch.Tensor]:
        if isinstance(handle, _Arg):
            return args[handle.index]
        if isinstance(handle, _Const):
            return handle.tensor
        raise NotImplementedError(
            "runnable binary/weight must come from a constant graph input"
        )

    spec = None
    flat_input_handles = None
    if op is not None:
        binary = resolve_const(node_handle(binary_node))
        weight = (
            None if weight_node is None else resolve_const(node_handle(weight_node))
        )
        stacks = [resolve_const(node_handle(n)) for n in stacks_list]
        runnable = getattr(binary, "_furiosa_runnable", None)
        flat_input_handles = [node_handle(n) for n in inputs_list]
        spec = _StepSpec(
            binary=binary,
            weight=weight,
            stacks=stacks,
            params=params,
            runnable=runnable,
            inputs=flat_input_handles,
        )

    returned = sub.graph.find_nodes(op="output")[0].args[0]
    if not isinstance(returned, (tuple, list)):
        returned = (returned,)
    logical_outputs = []
    for v in returned:
        # A multi-output runnable returned directly (no getitem) stands for its
        # whole output tuple; expand it so no trailing outputs are dropped.
        if op is not None and v is op and num_outputs != 1:
            logical_outputs.extend(_Output(step, j) for j in range(num_outputs))
        else:
            logical_outputs.append(node_handle(v))
    if op is not None:
        assert all(
            not isinstance(h, _Output) or h.out < num_outputs for h in logical_outputs
        )
    return spec, logical_outputs


def _slice_handle(node: torch.fx.Node, src_handle: Any) -> _Slice:
    """Build an unsharded strided-view handle for ``aten.slice.Tensor``.

    Rust composes the resulting byte offset and shape with later views.
    """
    sargs = node.args
    dim = int(sargs[1]) if len(sargs) > 1 else 0
    start = int(sargs[2]) if len(sargs) > 2 and sargs[2] is not None else 0
    src_meta = _example_tensor(sargs[0])
    out_meta = _example_tensor(node)
    if dim < 0:
        dim += src_meta.dim()
    if start < 0:
        start += src_meta.size(dim)
    # Match aten.slice clamping: start is pinned to [0, size] so out-of-range or
    # large-negative starts never produce a negative/oversized byte offset.
    start = min(max(start, 0), src_meta.size(dim))
    offset = start * src_meta.stride(dim) * out_meta.element_size()
    dram_shape = ir.FlatDramShape.new_as_broadcast_strided(
        out_meta.size(), out_meta.stride(), out_meta.dtype
    )
    return _Slice(src=src_handle, offset=offset, dram_shape=dram_shape, key=id(node))


def _passthrough_slice(pair: Passthrough) -> tuple[int, Any]:
    """View metadata for a passthrough output."""
    t = pair.target_tensor
    dram_shape = ir.FlatDramShape.new_as_broadcast_strided(
        t.size(), t.stride(), t.dtype
    )
    return pair.offset_src_to_dst, dram_shape


@instrument_with_name("torch_compile::build_sequence", propagate=False)
def build_sequence(split_gm: torch.fx.GraphModule, args: tuple) -> _SequencePlan:
    device = _utils.ensure_furiosa_device(
        [t for t in args if isinstance(t, torch.Tensor)]
    )

    # `env` maps FX values to symbolic handles while we collect runnable steps.
    env: dict[torch.fx.Node, list] = {}
    steps: list[_StepSpec] = []

    arg_counter = 0
    for node in split_gm.graph.nodes:
        if node.op == "placeholder":
            env[node] = [_Arg(arg_counter)]
            arg_counter += 1
        elif node.op == "get_attr":
            raise NotImplementedError("split graph unexpectedly has a get_attr node")
        elif node.op == "call_function" and node.target is operator.getitem:
            src, idx = node.args
            env[node] = [env[src][idx]]
        elif node.op == "call_module":
            # `split_module` moves slices into submodules.
            sub = getattr(split_gm, node.target)
            logical_inputs = [_handle_of(env, a) for a in node.args]
            if isinstance(sub, CompileModule):
                spec, logical_out = _step_compile_module(
                    sub, len(steps), logical_inputs
                )
                steps.append(spec)
            else:
                spec, logical_out = _flatten_submodule(
                    sub, len(steps), logical_inputs, args
                )
                if spec is not None:
                    steps.append(spec)
            env[node] = logical_out
        elif node.op == "output":
            final_values = node.args[0]
        else:
            raise NotImplementedError(f"unsupported top-level node {node.op} {node}")

    # Preserve the full output structure (tuple/list/dict/namedtuple/nested); FX
    # nodes are opaque to pytree, so they remain the flattened leaves.
    flat_values, out_spec = pytree.tree_flatten(final_values)
    final_handles = [_handle_of(env, v) for v in flat_values]

    # Flattened params per step; used to detect inter-chip sharding on a
    # sliced/passthrough view's source.
    step_output_shapes = [
        ir.Runnable._output_shapes_from_params(list(spec.params)) for spec in steps
    ]

    def _source_dram_shape(handle: Any):
        """Best-effort DRAM shape of the tensor a view ultimately reads from."""
        if isinstance(handle, _Output):
            shapes = step_output_shapes[handle.step]
            return shapes[handle.out] if 0 <= handle.out < len(shapes) else None
        if isinstance(handle, (_Slice, _Passthrough)):
            return _source_dram_shape(handle.src)
        if isinstance(handle, _Arg):
            return _get_dram_shape(args[handle.index])
        if isinstance(handle, _Const):
            return _get_dram_shape(handle.tensor)
        return None

    def _reject_sharded_view(src: Any) -> None:
        # Slice/passthrough lowering builds a broadcast (unsharded) DRAM shape,
        # which silently drops inter-chip axes. Reject rather than emit wrong
        # layout metadata; composing sharded views is not yet supported.
        base = _source_dram_shape(src)
        if base is not None and not base.is_broadcast():
            raise NotImplementedError(
                "sequence lowering cannot view/slice a tensor with inter-chip "
                f"sharding ({base}); this graph shape is not yet supported"
            )

    # Add source nodes before the nodes that use them. Rust keeps the node list.
    sequence_builder = _C_module._RunnableSequenceBuilder()
    placeholder_ids: dict[int, int] = {}
    const_ids: dict[int, int] = {}
    getitem_ids: dict[tuple[int, int], int] = {}
    runnable_ids: list[Optional[int]] = [None] * len(steps)

    def lower(handle: Any) -> int:
        if isinstance(handle, _Arg):
            if handle.index not in placeholder_ids:
                placeholder_ids[handle.index] = sequence_builder.placeholder(
                    handle.index
                )
            return placeholder_ids[handle.index]
        if isinstance(handle, _Const):
            if handle.key not in const_ids:
                const_ids[handle.key] = sequence_builder.const_tensor(handle.tensor)
            return const_ids[handle.key]
        if isinstance(handle, _Output):
            runnable_node_id = runnable_ids[handle.step]
            assert runnable_node_id is not None, (
                "runnable referenced before it was emitted"
            )
            key = (runnable_node_id, handle.out)
            if key not in getitem_ids:
                getitem_ids[key] = sequence_builder.getitem(
                    runnable_node_id, handle.out
                )
            return getitem_ids[key]
        if isinstance(handle, _Passthrough):
            _reject_sharded_view(handle.src)
            offset, dram_shape = _passthrough_slice(handle.pair)
            return sequence_builder.slice(lower(handle.src), offset, dram_shape)
        if isinstance(handle, _Slice):
            _reject_sharded_view(handle.src)
            return sequence_builder.slice(
                lower(handle.src), handle.offset, handle.dram_shape
            )
        if isinstance(handle, _Zero):
            zero = torch.empty_strided(
                handle.meta.shape,
                handle.meta.stride(),
                dtype=handle.meta.dtype,
                device=device,
            )
            return sequence_builder.const_tensor(zero)
        raise NotImplementedError(f"cannot lower handle {handle!r}")

    for step, spec in enumerate(steps):
        input_ids = [lower(h) for h in spec.inputs]
        runnable_ids[step] = sequence_builder.runnable(
            spec.binary,
            spec.weight,
            spec.stacks,
            spec.params,
            input_ids,
            spec.runnable,
        )

    sequence_builder.output([lower(h) for h in final_handles])

    return _SequencePlan(
        seq=sequence_builder.build(),
        out_spec=out_spec,
    )


@instrument_with_name("torch_compile::run_sequence", propagate=False)
def run_sequence(plan: _SequencePlan, args: tuple) -> Any:
    if plan.filter_inputs is not None:
        args = list(plan.filter_inputs(args))
    outputs = plan.seq.run(list(args))
    return pytree.tree_unflatten(outputs, plan.out_spec)
