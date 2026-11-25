import logging

import torch

from furiosa.torch import _utils, error, config
from furiosa.torch.profiler import RecordFunction

from ._exported_program import run_by_cpu
from .aten import AtenOp


logger = logging.getLogger("furiosa_torch")


@torch.compiler.disable
def run_aten_op(aten_op_name, op_callable, *args, **kwargs):
    qualified_aten_op_name = f"aten::{aten_op_name}"
    with RecordFunction(qualified_aten_op_name):
        return _run_aten_op(qualified_aten_op_name, op_callable, *args, **kwargs)


@torch.compiler.disable
def _run_aten_op(qualified_aten_op_name, op_callable, *args, **kwargs):
    with RecordFunction(f"run_aten_op::build_aten_op"):
        given_device = kwargs.pop("device", None)
        aten_op = AtenOp(qualified_aten_op_name, op_callable, args, kwargs)
        # Pass the flattened tensor-only leaves the program was traced from;
        # the original nested args would mismatch the exported input spec.
        traceable_args = aten_op.traceable_args
        traceable_kwargs = aten_op.traceable_kwargs

    try:
        with RecordFunction(f"run_aten_op::compile_and_load"):
            tensors = list(
                filter(lambda t: isinstance(t, torch.Tensor), traceable_args)
            )
            device = _utils.ensure_furiosa_device(tensors, given_device=given_device)
            m = aten_op.compile_and_load(device)
        outputs = m(*traceable_args, **traceable_kwargs)
    except error.UnsupportedOpError as e:
        if not config.get_cpu_fallback():
            raise e
        else:
            with RecordFunction(f"run_aten_op::build_exported_program"):
                exported = aten_op.build_exported_program()
                traceable_args = aten_op.traceable_args
                traceable_kwargs = aten_op.traceable_kwargs

            logger.debug(
                f"failed to compile & run {qualified_aten_op_name} on Furiosa, fallback to CPU",
                exc_info=e,
            )
            outputs = run_by_cpu(exported, traceable_args, traceable_kwargs)

    return outputs
