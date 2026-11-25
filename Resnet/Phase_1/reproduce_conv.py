"""Isolate folded layer4.0.conv2 with its actual CPU-captured activation."""
import argparse
import sys
import time
import traceback
from pathlib import Path

import torch
import furiosa.torch as ft

from Resnet.analysis import compare_tensors
from Resnet.artifacts import write_json
from Resnet.models import load_model, make_input, weight_metadata
from Resnet.runtime import make_callable, measure_e2e, synchronize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("eager", "compile"), default="eager")
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    ft.config.set_cpu_fallback(False)
    result = {"status": "running", "mode": args.mode, "module": "layer4.0.conv2",
              "scope": "Isolated compiler reproducer, not full-model inference performance",
              "cpu_fallback": ft.config.get_cpu_fallback()}
    code = 1
    try:
        model = load_model(bn_folded=True)
        result.update(weights=weight_metadata(), model_transform=model._benchmark_transform)
        convolution = model.layer4[0].conv2
        activation = {}
        def capture(module, inputs):
            activation["input"] = inputs[0].detach().clone()
        handle = convolution.register_forward_pre_hook(capture)
        with torch.inference_mode():
            model(make_input(1))
        handle.remove()
        cpu_input = activation["input"]
        with torch.inference_mode():
            reference = convolution(cpu_input)
        result.update(input_shape=list(cpu_input.shape), input_stride=list(cpu_input.stride()),
                      weight_shape=list(convolution.weight.shape), bias_shape=list(convolution.bias.shape),
                      stride=list(convolution.stride), padding=list(convolution.padding),
                      reference_shape=list(reference.shape), dtype=str(cpu_input.dtype))
        torch.save(cpu_input, args.run_dir / "input.pt")
        torch.save(convolution.state_dict(), args.run_dir / "conv-state.pt")
        device = torch.device("furiosa:0")
        convolution = convolution.to(device)
        synchronize(device)
        function, counter = make_callable(convolution, args.mode)
        result["stage"] = "isolated_first_call"
        write_json(args.run_dir / "progress.json", result)
        started = time.perf_counter()
        try:
            with ft.config.cache_context(str((args.run_dir / "compiler-cache").resolve())), torch.inference_mode():
                elapsed, output = measure_e2e(function, cpu_input, device)
            result.update(status="ok", first_call_ms=elapsed,
                          cpu_reference_comparison=compare_tensors(reference, output, classifier=False))
            code = 0
        finally:
            result.update(attempt_elapsed_s=time.perf_counter() - started, backend_calls=counter.calls)
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    write_json(args.run_dir / "result.json", result)
    return code


if __name__ == "__main__":
    sys.exit(main())
