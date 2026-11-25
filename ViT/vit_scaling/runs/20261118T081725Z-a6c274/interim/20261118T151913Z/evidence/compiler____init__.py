__all__ = [
    "compile",
    "lower",
    "import_dfg",
    "compile_raw_dfg",
    "lower_raw_dfg",
    "version",
    "is_supported_aten",
    "is_importable",
]

import logging
from typing import Optional

import torch

from furiosa.torch import error, config, ir
from furiosa.torch._compile_probe import probe_native_compile
from furiosa.torch.profiler import instrument
from furiosa.torch._tensor import _ensure_dram_shape
from furiosa.torch._C import compiler, ir
from furiosa.torch._C.compiler import version, is_supported_aten, is_importable
from furiosa.torch.export import graph_source


logger = logging.getLogger("furiosa_torch")


def _extract_dram_shape(t):
    if isinstance(t, torch.Tensor):
        return _ensure_dram_shape(t)
    else:
        return None


def _extract_input_dram_shapes(
    ep: torch.export.ExportedProgram,
) -> list[ir.FlatDramShape]:
    from furiosa.torch.export import ExportedProgramWeight

    dram_shapes = []
    if ep.example_inputs is not None:
        args, kwargs = ep.example_inputs
        weights = ExportedProgramWeight(ep)
        inputs = weights.flatten_inputs(args, kwargs)
        for t in inputs:
            dram_shape = _extract_dram_shape(t)
            if dram_shape is not None:
                dram_shapes.append(dram_shape)
    else:
        for n in ep.graph.find_nodes(op="placeholder"):
            t = n.meta["val"]
            dram_shape = _extract_dram_shape(t)
            if dram_shape is not None:
                dram_shapes.append(dram_shape)

    return dram_shapes


@instrument
def compile(
    ep: torch.export.ExportedProgram,
    compiler_config: Optional[config.CompilerConfig] = None,
    *,
    task_name: Optional[str] = None,
) -> ir.Runnable:
    """
    Compile the given exported program to a Runnable.

    Args:
        ep (torch.export.ExportedProgram): The exported program to compile.
        compiler_config (Optional[config.CompilerConfig]): Optional compiler configuration to guide the compilation process.
        task_name: Cache-log label only; excluded from the cache key so naming
            a compile does not change cache reuse.

    Returns:
        The compiled Runnable
    """
    dram_shapes = _extract_input_dram_shapes(ep)
    try:
        # Records compile-probe metadata only when a probe context is active.
        with probe_native_compile(ep):
            runnable = compiler.compile(
                ep,
                compiler_config=compiler_config,
                dram_shapes=dram_shapes,
                task_name=task_name,
            )
        logger.debug(f"graph compiled successfully")
    except error.CompileError as e:
        node_names = e.gm or None
        # Print graph only if there is no GM provenance
        graph_str = f"{ep.graph}\n" if not node_names else ""
        raise error.UnsupportedOpError(
            f"failed to compile\n{graph_str}"
            f"source:\n{graph_source(ep.graph, indent=4, node_names=node_names)}\n"
            f"compile error:\n{e}"
        ) from e
    except BaseException as e:
        # catch and warp all exceptions to UnsupportedOpError to classify exceptions
        # Wrap PyO3 PanicException from Rust panic as UnsupportedOpError too.
        if isinstance(e, Exception) or type(e).__name__ == "PanicException":
            raise error.UnsupportedOpError(
                f"failed to compile\n{ep.graph}\n"
                f"source:\n{graph_source(ep.graph, indent=4)}\n"
                f"compile error:\n{e}"
            ) from e
        raise

    return runnable


@instrument
def lower(
    ep: torch.export.ExportedProgram,
    compiler_config: Optional[config.CompilerConfig] = None,
    *,
    task_name: Optional[str] = None,
) -> ir.Dfg:
    """
    Lower the given exported program to DFG, which is lowered shape has been decided but not yet scheduled.

    Args:
        ep (torch.export.ExportedProgram): The exported program to compile.
        compiler_config (Optional[config.CompilerConfig]): Optional compiler configuration to guide the compilation process.
        task_name: Cache-log label only; excluded from the cache key so naming
            a compile does not change cache reuse.

    Returns:
        The lowered DFG
    """
    dram_shapes = _extract_input_dram_shapes(ep)
    try:
        dfg = compiler.lower(
            ep,
            compiler_config=compiler_config,
            dram_shapes=dram_shapes,
            task_name=task_name,
        )
        logger.debug(f"graph lower successfully")
        return dfg
    except error.CompileError as e:
        node_names = e.gm or None
        # Print graph only if there is no GM provenance
        graph_str = f"{ep.graph}\n" if not node_names else ""
        raise error.UnsupportedOpError(
            f"failed to lower dfg\n{graph_str}"
            f"source:\n{graph_source(ep.graph, indent=4, node_names=node_names)}\n"
            f"compile error:\n{e}"
        ) from e
    except BaseException as e:
        # catch and warp all exceptions to UnsupportedOpError to classify exceptions
        # Wrap PyO3 PanicException from Rust panic as UnsupportedOpError too.
        if isinstance(e, Exception) or type(e).__name__ == "PanicException":
            raise error.UnsupportedOpError(
                f"failed to compile\n{ep.graph}\n"
                f"source:\n{graph_source(ep.graph, indent=4)}\n"
                f"compile error:\n{e}"
            ) from e
        raise


@instrument
def import_dfg(ep: torch.export.ExportedProgram) -> ir.Dfg:
    """
    Import the given exported program to DFG without any lowering or optimization, which is useful for debugging and testing.

    Args:
        ep (torch.export.ExportedProgram): The exported program to compile.

    Returns:
        The imported DFG
    """
    dram_shapes = _extract_input_dram_shapes(ep)
    try:
        dfg = compiler.import_dfg(ep, dram_shapes)
        logger.debug(f"graph imported successfully")
        return dfg
    except Exception as e:
        # catch and warp all exceptions to UnsupportedOpError to classify exceptions
        raise error.UnsupportedOpError(
            f"failed to import_dfg\n{ep.graph}\n"
            f"source:\n{graph_source(ep.graph, indent=4)}\n"
            f"compile error:\n{e}"
        ) from e


@instrument
def compile_raw_dfg(dfg: ir.Dfg, *, task_name: Optional[str] = None) -> ir.Runnable:
    """
    Compile the given raw DFG to a Runnable.

    Args:
        dfg (ir.Dfg): The raw DFG to compile.
        task_name: Cache-log label only; excluded from the cache key so naming
            a compile does not change cache reuse.

    Returns:
        The compiled Runnable
    """
    try:
        runnable = compiler.compile_raw_dfg(dfg, task_name=task_name)
        logger.debug(f"graph compiled successfully")
        return runnable
    except Exception as e:
        # catch and warp all exceptions to UnsupportedOpError to classify exceptions
        raise error.UnsupportedOpError(
            f"failed to compile the graph\ncompile error:\n{e}"
        ) from e


@instrument
def lower_raw_dfg(dfg: ir.Dfg, *, task_name: Optional[str] = None) -> ir.Dfg:
    """
    Lower the given raw DFG to DFG, which is lowered shape has been decided but not yet scheduled.

    Args:
        dfg (ir.Dfg): The raw DFG to lower.
        task_name: Cache-log label only; excluded from the cache key so naming
            a compile does not change cache reuse.

    Returns:
        The lowered DFG
    """
    try:
        dfg = compiler.lower_raw_dfg(dfg, task_name=task_name)
        logger.debug(f"graph lowered successfully")
        return dfg
    except Exception as e:
        # catch and warp all exceptions to UnsupportedOpError to classify exceptions
        raise error.UnsupportedOpError(
            f"failed to lower the graph\ncompile error:\n{e}"
        ) from e
