import logging
from typing import Sequence, Optional, Callable, Any

import torch
from torch._guards import detect_fake_mode
from torch._subclasses.fake_tensor import fake_tensor_tls

from furiosa.torch import compiler
from furiosa.torch.export import ExportedProgramWeight, PassContext
from furiosa.torch.export.exported_program import (
    ExportedProgramOutputReconstructor,
    preprocess_exported_program,
)
from furiosa.torch.profiler import instrument_with_name
from furiosa.torch._tensor import _set_dram_shape
from furiosa.torch import _utils
from furiosa.torch._C import device, module, ir


logger = logging.getLogger("furiosa_torch")


@torch.compiler.assume_constant_result
def _deserialize_runnable_params(params: Sequence[int]) -> list[ir.FlatDramShape]:
    return ir.Runnable._output_shapes_from_params(params)


@torch.library.custom_op("furiosa::runnable", mutates_args={})
def runnable(
    inputs: Sequence[torch.Tensor],
    binaries: torch.Tensor,
    weight: Optional[torch.Tensor],
    stacks: Sequence[torch.Tensor],
    serialized_runnable_params: Sequence[int],
) -> list[torch.Tensor]:
    inputs = list(inputs)
    device = _utils.ensure_furiosa_device(inputs)

    tp = module.TpModule.from_runnable_params(
        device,
        inputs,
        binaries,
        weight,
        stacks,
        serialized_runnable_params,
    )
    return tp.run()


@torch.library.register_fake("furiosa::runnable")
def runnable_fake(
    inputs: Sequence[torch.Tensor],
    binaries: torch.Tensor,
    weight: Optional[torch.Tensor],
    stacks: Sequence[torch.Tensor],
    serialized_runnable_params: Sequence[int],
) -> list[torch.Tensor]:
    device = _utils.ensure_same_device(inputs)

    output_dram_shapes = _deserialize_runnable_params(serialized_runnable_params)
    outputs = []
    for dram_shape in output_dram_shapes:
        t = torch.empty_strided(
            size=dram_shape.host_sizes,
            stride=dram_shape.host_strides,
            dtype=dram_shape.dtype,
            device=device,
        )
        _set_dram_shape(t, dram_shape)
        outputs.append(t)
    return outputs


class TpModule(torch.nn.Module):
    def __init__(
        self,
        m: "Runnable",
        inputs: list[torch.Tensor],
    ):
        super().__init__()
        self._inner = module.TpModule.from_runnable(m._inner, inputs)

    @classmethod
    def _from_inner(cls, inner: module.TpModule) -> "TpModule":
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj._inner = inner
        return obj

    @property
    def inputs(self) -> list[torch.Tensor]:
        return self._inner.inputs

    @property
    def outputs(self) -> list[torch.Tensor]:
        return self._inner.outputs

    def forward(self) -> list[torch.Tensor]:
        # FIXME: Need to postprocess outputs like `CompileModule.forward`.
        return self._inner.run()


class Runnable(torch.nn.Module):
    def __init__(self, runnable: ir.Runnable):
        assert isinstance(runnable, ir.Runnable), (
            "runnable must be an instance of ir.Runnable"
        )
        super().__init__()
        self._inner = runnable
        self._tag_binary_with_runnable()

    def _tag_binary_with_runnable(self):
        # FX keeps the binary tensor but drops ir.Runnable; this tag restores profiling.
        self._inner.binary._furiosa_runnable = self._inner

    @instrument_with_name("Runnable::_apply", propagate=False)
    def _apply(self, fn, recurse=True):
        """
        Load the compiled Runnable module to the specified device.

        Args:
            device (torch.device): The target device to load the Runnable module onto.
        """
        super()._apply(fn, recurse)
        weight = self._inner.weight
        if weight is not None:
            self._inner.weight = fn(weight)
        binary = self._inner.binary
        self._inner.binary = fn(binary)
        self._tag_binary_with_runnable()
        # Reserve stack scratch on the new device so forward reuses it.
        self._inner.reserve_stacks()
        return self

    def build_tp_module(
        self,
        inputs: Sequence[torch.Tensor],
    ) -> TpModule:
        return TpModule._from_inner(self._inner.build_tp_module(inputs))

    def _forward_tracing(self, *args):
        # Allow real binary/weight/stack tensors.
        old_override = fake_tensor_tls.allow_non_fake_inputs_override
        fake_tensor_tls.allow_non_fake_inputs_override = True
        try:
            return torch.ops.furiosa.runnable(
                args,
                self._inner.binary,
                self._inner.weight,
                self._inner.stacks,
                self._inner._flattened_params,
            )
        finally:
            fake_tensor_tls.allow_non_fake_inputs_override = old_override

    @property
    def input_dram_shapes(self) -> list[ir.FlatDramShape]:
        return self._inner.input_dram_shapes

    @property
    def output_dram_shapes(self) -> list[ir.FlatDramShape]:
        return self._inner.output_dram_shapes

    @instrument_with_name("Runnable::forward", propagate=False)
    def forward(
        self,
        *args,
    ):
        if torch.compiler.is_compiling() or detect_fake_mode(args) is not None:
            # Runnable eager path cannot be used with fake tensors.
            return self._forward_tracing(*args)

        current_num_fusion = device.get_fusion().num_fusion
        if current_num_fusion != self._inner.expected_num_fusion:
            raise RuntimeError(
                "Expected device fusion of compiled Runnable doesn't match current device fusion "
                f"({self._inner.expected_num_fusion} != {current_num_fusion})"
            )

        outputs = self._inner.execute(args)

        return outputs


class CompileModule(torch.nn.Module):
    def __init__(
        self,
        runnable: Runnable,
        weight: ExportedProgramWeight,
        output_reconstructor: Optional[ExportedProgramOutputReconstructor] = None,
    ):
        super().__init__()
        assert isinstance(runnable, Runnable), (
            "runnable must be an instance of Runnable"
        )
        assert isinstance(weight, ExportedProgramWeight), (
            "weight must be an instance of ExportedProgramWeight"
        )
        if output_reconstructor is None:
            output_reconstructor = ExportedProgramOutputReconstructor()
        assert isinstance(output_reconstructor, ExportedProgramOutputReconstructor), (
            "output_reconstructor must be an instance of "
            "ExportedProgramOutputReconstructor"
        )
        self.runnable = runnable
        self.module_weight = weight
        self.output_reconstructor = output_reconstructor

    @classmethod
    def from_exported(
        cls,
        exported: torch.export.ExportedProgram,
        compiler_config=None,
        *,
        pass_context: Optional[PassContext] = None,
        task_name: Optional[str] = None,
    ):
        exported, output_reconstructor = preprocess_exported_program(
            exported, pass_context
        )
        runnable = Runnable(
            compiler.compile(exported, compiler_config, task_name=task_name)
        )
        weight = ExportedProgramWeight(exported)
        return cls(runnable, weight, output_reconstructor)

    @classmethod
    def from_module(
        cls,
        m: torch.nn.Module | torch.export.ExportedProgram | Callable,
        args: tuple[Any, ...],
        *,
        kwargs: Optional[dict[str, Any]] = None,
        compiler_config=None,
        pass_context: Optional[PassContext] = None,
        task_name: Optional[str] = None,
    ) -> "CompileModule":
        module = _utils.maybe_wrap_function(m)
        exported = torch.export.export(module, args, kwargs, strict=False)
        return cls.from_exported(
            exported,
            compiler_config=compiler_config,
            pass_context=pass_context,
            task_name=task_name,
        )

    def build_tp_module(
        self,
        *args,
        **kwargs,
    ) -> TpModule:
        inputs = self.module_weight.flatten_inputs(args, kwargs)
        return self.runnable.build_tp_module(inputs)

    @instrument_with_name("CompileModule::forward", propagate=False)
    def forward(self, *args, **kwargs):
        inputs = self.module_weight.flatten_inputs(args, kwargs)
        outputs = self.runnable(*inputs)
        assert isinstance(outputs, list)
        output_reconstructor = self.output_reconstructor
        device = None
        if output_reconstructor.zero_sized_tensor_info:
            device = self.runnable._inner.loaded_device
            assert device is not None
        reconstructed_outputs = output_reconstructor.reconstruct_outputs(
            outputs,
            lambda pair: pair.view_into_target(inputs[pair.input_idx]),
            lambda meta: output_reconstructor.zero_sized_tensor_info.make_output(
                meta, device
            ),
        )
        return self.module_weight.unflatten_outputs(reconstructed_outputs)
