import logging
import functools
from collections.abc import Callable, Mapping
from typing import Optional

import torch
from torch.utils import _pytree as pytree
from torch._subclasses.fake_tensor import FakeTensorMode

from furiosa.torch import error
from furiosa.torch import _utils
from furiosa.torch.export import PassContext, make_pass_context
from furiosa.torch.export._meta import is_runnable_node
from furiosa.torch.export.passes import (
    _inline_dfg_tensors_from_user_inputs,
    _split_by_runnable,
)
from furiosa.torch._module import CompileModule
from .runnable_sequence import build_sequence, run_sequence

logger = logging.getLogger("furiosa_torch")


class _Backend:
    def __init__(
        self,
        compiler_config=None,
        *,
        eager_fallback=False,
        decomp_overrides: Mapping[torch._ops.OpOverload, Callable] = {},
    ):
        self.compiler_config = compiler_config
        self.eager_fallback = eager_fallback
        self.decomp_overrides = dict(decomp_overrides)
        # Snapshot overrides so cached forwards use one stable decomposition table.
        self._pass_context = make_pass_context(self.decomp_overrides)

    @classmethod
    def with_config(
        cls,
        compiler_config=None,
        *,
        eager_fallback=False,
        decomp_overrides: Mapping[torch._ops.OpOverload, Callable] = {},
    ) -> "_Backend":
        """
        backend for proper compilation, it will compile the graph and execute it with the compiled Runnable

        Args:
            compiler_config: Compiler configuration
            eager_fallback: If True, fall back to eager execution when an op is
                unsupported on the device.
            decomp_overrides: Mapping of ``torch._ops.OpOverload`` to a Python
                decomposition callable. Same-key entries override the built-in
                ``STD_DECOMPOSITIONS`` (an override warning is logged). The
                mapping is folded into the backend's pass context at
                construction time, so later mutation of this attribute is not
                honored — build a new backend via ``with_config`` to switch
                rule sets.

        Returns:
            Callable: A function that takes inputs and returns outputs
        """
        return cls(
            compiler_config,
            eager_fallback=eager_fallback,
            decomp_overrides=decomp_overrides,
        )

    def __call__(
        self, gm: torch.fx.GraphModule, example_inputs: tuple[torch.Tensor, ...]
    ):
        """
        backend for proper compilation, it will compile the graph and execute it with the compiled Runnable

        Args:
            gm (torch.fx.GraphModule): GraphModule to be compiled

        Returns:
            Callable: A function that takes inputs and returns outputs
        """
        return _backend_inner(
            gm,
            self.compiler_config,
            self.eager_fallback,
            pass_context=self._pass_context,
        )

    def backend(
        self, gm: torch.fx.GraphModule, example_inputs: tuple[torch.Tensor, ...]
    ):
        """
        Alias of `__call__`.
        """
        return self(gm, example_inputs)

    def precompile(
        self, gm: torch.fx.GraphModule, example_inputs: tuple[torch.Tensor, ...]
    ):
        """
        backend for warm up compile cache
        """

        logger.debug(f"precompile gm {gm.graph}")

        seq_plan = None

        def forward(*args):
            nonlocal gm, seq_plan

            try:
                if seq_plan is None:
                    split_gm, filter_inputs = _build_segmented(
                        gm, args, self.compiler_config, self._pass_context
                    )
                    seq_plan = build_sequence(split_gm, filter_inputs(args))
                    seq_plan.filter_inputs = filter_inputs
            except Exception as e:
                logger.warning(f"failed to precompile target gm: {e}")
            return _run_fake(gm, *args)

        return forward


backend = _Backend()
backend_eager_fallback = _Backend(eager_fallback=True)


def _run_fake(gm: torch.fx.GraphModule, *args, **kwargs) -> tuple[torch.Tensor, ...]:
    fake_mode = FakeTensorMode()
    converter = fake_mode.fake_tensor_converter
    to_fake = functools.partial(converter.from_real_tensor, fake_mode)

    fake_args = pytree.tree_map(to_fake, args)
    fake_kwrags = pytree.tree_map(to_fake, kwargs)
    with fake_mode:
        fake_outputs = gm(*fake_args, **fake_kwrags)
    return tuple(
        torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device).zero_()
        for t in fake_outputs
    )


def _example_tensor(node: torch.fx.Node) -> torch.Tensor:
    meta = node.meta
    if "val" in meta:
        return meta["val"]
    if "example_value" in meta:
        return meta["example_value"]
    raise KeyError(f"node {node} has no 'val'/'example_value' meta")


def _build_segmented(
    gm: torch.fx.GraphModule,
    args: tuple[torch.Tensor, ...],
    compiler_config,
    pass_context: Optional[PassContext],
) -> tuple[torch.fx.GraphModule, Callable]:
    """Prepare the split graph and input filter for ``RunnableSequence``.

    DFG tensors are inlined as constants first, so the returned filter removes
    them from runtime inputs.
    """
    ep, filter_inputs = _inline_dfg_tensors_from_user_inputs(gm, args)
    inlined_gm = ep.module(check_guards=False)

    split_gm = _split_by_runnable(inlined_gm)
    logger.debug(f"building segmented gm, split {split_gm.graph}")

    segment_names = {
        name
        for name, sub in split_gm.named_children()
        if isinstance(sub, torch.fx.GraphModule)
        and not any(is_runnable_node(n) for n in sub.graph.nodes)
    }
    logger.debug(
        f"split into {len(list(split_gm.named_children()))} submodules, "
        f"{len(segment_names)} to compile"
    )

    for name in segment_names:
        submod = getattr(split_gm, name)
        seg_args = tuple(
            _example_tensor(n) for n in submod.graph.find_nodes(op="placeholder")
        )
        cm = CompileModule.from_module(
            submod,
            seg_args,
            compiler_config=compiler_config,
            pass_context=pass_context,
        )
        # Empty binaries have no device work to move.
        if cm.runnable._inner.binary.numel() > 0:
            device = _utils.ensure_furiosa_device(seg_args)
            cm.to(device)
        cm._segment_example_inputs = seg_args  # for `flatten_inputs` order recovery
        setattr(split_gm, name, cm)

    return split_gm, filter_inputs


def _backend_inner(
    gm: torch.fx.GraphModule,
    compiler_config=None,
    eager_fallback=False,
    *,
    pass_context: Optional[PassContext] = None,
):
    logger.debug(f"torch.compile {gm.graph}")
    seq_plan = None

    def forward(*args):
        nonlocal gm, compiler_config, seq_plan

        # Build the segmented RunnableSequence once; later calls only bind inputs.
        try:
            if seq_plan is None:
                split_gm, filter_inputs = _build_segmented(
                    gm, args, compiler_config, pass_context
                )
                seq_plan = build_sequence(split_gm, filter_inputs(args))
                seq_plan.filter_inputs = filter_inputs
            return run_sequence(seq_plan, args)
        except (error.UnsupportedOpError, NotImplementedError) as e:
            if not eager_fallback:
                # Implicit eager execution is disabled, re-raise the exception
                logger.error(
                    f"eager fallback is not allowed for the target gm", exc_info=e
                )
                raise e from None
            logger.debug(f"eager fallback for the target gm: {e}")
            return gm(*args)

    return forward
