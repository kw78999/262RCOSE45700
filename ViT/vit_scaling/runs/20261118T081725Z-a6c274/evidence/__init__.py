__all__ = [
    # tensor
    "empty",
    "empty_sharded",
    "empty_strided_sharded",
    "copy_from_tensor",
    "resize",
    "with_memory_type",
    "MemoryType",
    # device API
    "device",
    "is_initialized",
    "get_amp_supported_dtype",
    "is_available",
    "is_register_hook",
    "manual_seed_all",
    "register_hook",
    "_generator_default_state_tensor",
    "_generator_state_tensor",
    "_is_in_bad_fork",
    # torch.compile backend
    "backend",
    "backend_eager_fallback",
    # modules
    "DfgModule",
    "LowerModule",
    "TclModule",
    "CompileModule",
    "Runnable",
    "TpModule",
    "compile_from_visa",
    "compile_from_visa_str",
    # furiosa custom device APIs
    "set_fusion",
    "get_fusion",
    "Fusion",
    "get_available_devices",
    "get_device_configuration",
    "MeshKind",
    "set_mesh_kind",
    "get_mesh_kind",
    "current_device",
    "exchange_device",
    # stream API
    "Stream",
    "Event",
    "current_stream",
    "default_stream",
    "set_stream",
    "stream",
    # compiler & configs
    "compiler",
    "config",
    # ir
    "ir",
    # extra
    "ext",
    "export",
    "dryrun",
    "error",
    "version",
]

from collections.abc import Sequence
from contextlib import contextmanager

import torch

from furiosa.torch._C import version
from furiosa.torch._C._ext import (
    get_amp_supported_dtype,
    is_available,
    is_register_hook,
    manual_seed_all,
    register_hook,
    _generator_default_state_tensor,
    _generator_state_tensor,
    _is_in_bad_fork,
)
from furiosa.torch._C.device import (
    set_fusion,
    get_fusion,
    Fusion,
    get_available_devices,
    get_device_configuration,
    MeshKind,
    set_mesh_kind,
    get_mesh_kind,
    exchange_device,
    current_device,
)
from furiosa.torch._C.stream import (
    Stream,
    Event,
)
from furiosa.torch.config._stream import current_stream, default_stream, set_stream
from furiosa.torch._C.tensor import empty, copy_from_tensor
from furiosa.torch import error, ir
from furiosa.torch._tensor import MemoryType, _set_dram_shape, with_memory_type
from furiosa.torch import export
from furiosa.torch import compiler
from furiosa.torch import config
from furiosa.torch._module import (
    CompileModule,
    DfgModule,
    LowerModule,
    TclModule,
    Runnable,
    TpModule,
    compile_from_visa,
    compile_from_visa_str,
)
from furiosa.torch._backend.torch_compile import backend, backend_eager_fallback
from furiosa.torch import ext
from furiosa.torch import dryrun
from furiosa.torch._custom_ops.empty import _resolve_device


@contextmanager
def stream(device=None):
    """
    A context manager that creates a new stream on the specified device and sets it as the current stream within the context.

    Args:
        device (torch.device, optional): The device on which to create the new stream. If None, the current device will be used.
    """
    device = device if device is not None else current_device()
    new_s = Stream(device)
    prev_s = set_stream(new_s)
    try:
        yield new_s
    finally:
        assert set_stream(prev_s).stream_id == new_s.stream_id


@contextmanager
def device(device):
    """
    A context manager that temporarily sets the current device to the specified device.

    Args:
        device (torch.device, optional): The device to set as the current device within the context.
    """
    this = device
    prev = exchange_device(device)
    try:
        yield device
    finally:
        assert exchange_device(prev) == this


def is_initialized() -> bool:
    """
    Check if the Furiosa NPU device has been initialized.

    Returns:
        bool: True if the device is initialized, False otherwise.
    """
    # Report backend hook registration state.
    return is_register_hook()


def empty_sharded(
    size: Sequence[int],
    *,
    shard_axes: Sequence[tuple[int, int]],
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Allocate from host/logical sizes with host-axis shard splits."""
    dtype = dtype if dtype is not None else torch.get_default_dtype()
    dram_shape = ir.FlatDramShape.from_host_sizes(list(size), shard_axes, dtype)
    return empty(dram_shape, _resolve_device(device))


def empty_strided_sharded(
    size: Sequence[int],
    stride: Sequence[int],
    *,
    shard_axes: Sequence[tuple[int, int]],
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Allocate from host/logical sizes and strides with host-axis shard splits."""
    dtype = dtype if dtype is not None else torch.get_default_dtype()
    dram_shape = ir.FlatDramShape.from_host_sizes_and_strides(
        list(size), list(stride), shard_axes, dtype
    )
    return empty(dram_shape, _resolve_device(device))


# FIXME, this is a temporary API for matching up dram shape, calling resize_ will break
# dram shape integrity, we need to find a better way to handle this.
def resize(t, size: list[int]):
    t.resize_(size)
    dram_shape = ir.FlatDramShape.new_as_broadcast(size, t.dtype)
    _set_dram_shape(t, dram_shape)
    return t
