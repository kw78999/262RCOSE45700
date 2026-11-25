import logging
from dataclasses import dataclass
from functools import lru_cache
from types import TracebackType
from typing import Optional

import torch
from torch.utils import _pytree as pytree

from furiosa.torch._module import CompileModule
from furiosa.torch import _utils
from furiosa.torch.export import PASSES
from furiosa.torch.profiler import instrument_with_name
from ._utils import cache_key_from_args


logger = logging.getLogger("furiosa_torch")


@dataclass
class AtenOpCompileResult:
    cached: torch.nn.Module | Exception
    lru_cache_hit: bool = False
    original_traceback: Optional[TracebackType] = None

    def out(self):
        if isinstance(self.cached, Exception):
            raise self.cached.with_traceback(self.original_traceback)
        return self.cached


@instrument_with_name("AtenOp::build_exported_program")
@torch.compiler.disable
def _build_exported_program(aten_op) -> torch.export.ExportedProgram:
    logger.info("Building exported program %s", aten_op)

    op_callable = aten_op.op_callable
    args = aten_op.traceable_args
    args_spec = aten_op.args_spec
    non_traceable_args = aten_op.non_traceable_args
    non_traceable_kwargs = aten_op.non_traceable_kwargs
    traceable_kwargs = aten_op.traceable_kwargs
    num_flattened_args = len(args) + len(non_traceable_args)

    class M(torch.nn.Module):
        def __init__(self, non_traceable_args, non_traceable_kwargs):
            super().__init__()
            # torch export can't trace argument-related types,
            # such as torch.device, torch.dtype, etc.
            self.non_traceable_args = non_traceable_args
            self.non_traceable_kwargs = non_traceable_kwargs

        def forward(self, *args, **kwargs):
            traceable_args = iter(args)
            flattened_args = [
                self.non_traceable_args[index]
                if index in self.non_traceable_args
                else next(traceable_args)
                for index in range(num_flattened_args)
            ]
            structured_args = pytree.tree_unflatten(flattened_args, args_spec)
            return op_callable(*structured_args, **self.non_traceable_kwargs, **kwargs)

    fake_args, fake_kwargs = _utils.to_fake_furiosa_tensors((args, traceable_kwargs))

    exported = torch.export.export(
        M(non_traceable_args, non_traceable_kwargs),
        fake_args,
        fake_kwargs,
        strict=False,
    )
    for p in PASSES:
        exported = p(exported)
    return exported


_build_exported_program_cached = lru_cache(maxsize=8192)(_build_exported_program)


def set_build_exported_program_cache(maxsize: int):
    global _build_exported_program_cached
    _build_exported_program_cached.cache_clear()
    _build_exported_program_cached = lru_cache(maxsize=maxsize)(_build_exported_program)


@instrument_with_name("AtenOp::compile")
@torch.compiler.disable
def _compile(aten_op) -> AtenOpCompileResult:
    location = _utils._find_user_call_site()
    location_suffix = f" at {location}" if location is not None else ""

    task_name = f"{aten_op.aten_op_name}{location_suffix}"

    try:
        exported = _build_exported_program_cached(aten_op)
        compiled = CompileModule.from_exported(exported, task_name=task_name)
        return AtenOpCompileResult(compiled)
    except Exception as e:
        import traceback

        original_tb = e.__traceback__
        # Clear locals from exception traceback to allow tensor deallocation
        traceback.clear_frames(original_tb)
        return AtenOpCompileResult(e, original_traceback=original_tb)


_compile_cached = lru_cache(maxsize=8192)(_compile)


def set_compile_cache(maxsize: int):
    global _compile_cached
    _compile_cached.cache_clear()
    _compile_cached = lru_cache(maxsize=maxsize)(_compile)


@instrument_with_name("AtenOp::compile_and_load")
@torch.compiler.disable
def _compile_and_load(aten_op, device: torch.device) -> AtenOpCompileResult:
    ret = _compile_cached(aten_op)
    if isinstance(ret.cached, torch.nn.Module):
        logger.info("Loading to %s for %s", device, aten_op)
        ret.cached.to(device)
    return ret


_compile_and_load_cached = lru_cache(maxsize=32)(_compile_and_load)


def set_load_cache(maxsize: int):
    global _compile_and_load_cached
    _compile_and_load_cached.cache_clear()
    _compile_and_load_cached = lru_cache(maxsize=maxsize)(_compile_and_load)


class AtenOp:
    def __init__(
        self,
        aten_op_name,
        op_callable,
        args,
        kwargs,
    ):
        self.aten_op_name = aten_op_name
        self.op_callable = op_callable

        flattened_args, self.args_spec = pytree.tree_flatten(args)

        self.non_traceable_args = {}
        traceable_args = []
        for index, arg in enumerate(flattened_args):
            if isinstance(arg, torch.Tensor):
                traceable_args.append(arg)
            else:
                self.non_traceable_args[index] = arg
        self.traceable_args = tuple(traceable_args)

        self.non_traceable_kwargs = {}
        self.traceable_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                self.traceable_kwargs[key] = value
            else:
                self.non_traceable_kwargs[key] = value

        self.args = args
        self.flatten_shape_params = cache_key_from_args(args, kwargs)
        self.aten_op_hash = hash((self.aten_op_name, self.flatten_shape_params))

    def build_exported_program(self) -> torch.export.ExportedProgram:
        return _build_exported_program_cached(self)

    def compile_and_load(self, device: torch.device) -> CompileModule:
        return _compile_and_load_cached(self, device).out()

    def compile(self) -> CompileModule:
        return _compile_cached(self).out()

    def compile_raw(self) -> AtenOpCompileResult:
        cache_info_before = _compile_cached.cache_info()
        compile_result = _compile_cached(self)
        cache_info_after = _compile_cached.cache_info()
        return AtenOpCompileResult(
            cached=compile_result.cached,
            lru_cache_hit=cache_info_after.hits > cache_info_before.hits,
        )

    def __hash__(self):
        return self.aten_op_hash

    def __eq__(self, other):
        if not isinstance(other, AtenOp):
            return False
        eq = self.aten_op_name == other.aten_op_name
        eq &= self.flatten_shape_params == other.flatten_shape_params
        return eq

    def __str__(self):
        # Exclude treespec when printing
        args = cache_key_from_args(self.args, {})[1:]
        traceable_kwargs = cache_key_from_args((), self.traceable_kwargs)[1:]
        non_traceable_kwargs = cache_key_from_args((), self.non_traceable_kwargs)[1:]

        return f"AtenOp(aten_op_name={self.aten_op_name}, args={args}, traceable_kwargs={traceable_kwargs}, non_traceable_kwargs={non_traceable_kwargs})"
