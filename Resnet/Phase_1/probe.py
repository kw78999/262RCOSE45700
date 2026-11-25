"""Minimum strict model support check; each mode runs in a separate process."""
import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

import torch
import furiosa.torch as ft

from Resnet.artifacts import write_json
from Resnet.models import load_model, make_input, weight_metadata
from Resnet.runtime import make_callable, measure_e2e, eager_cache_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("eager", "compile"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("furiosa_torch").setLevel(logging.DEBUG)
    result = {"mode": args.mode, "status": "running", "stage": "model_load"}
    code = 1
    try:
        torch.set_num_threads(1)
        ft.config.set_cpu_fallback(False)
        result["cpu_fallback"] = ft.config.get_cpu_fallback()
        model = load_model()
        result["weights"] = weight_metadata()
        result["stage"] = "model_to_device"
        device = torch.device("furiosa:0")
        model = model.to(device)
        function, counter = make_callable(model, args.mode)
        result["stage"] = "first_call"
        result["cache_before"] = eager_cache_stats()
        started = time.perf_counter()
        try:
            with ft.config.cache_context(str(args.output / "compiler-cache")), torch.inference_mode():
                elapsed, output = measure_e2e(function, make_input(1), device)
            result.update(status="ok", first_e2e_ms=elapsed, output_shape=list(output.shape), output_dtype=str(output.dtype))
            code = 0
        finally:
            result["attempt_elapsed_s"] = time.perf_counter() - started
            result["backend_calls"] = counter.calls
            result["cache_after"] = eager_cache_stats()
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        write_json(args.output / "result.json", result)
    return code


if __name__ == "__main__":
    sys.exit(main())
