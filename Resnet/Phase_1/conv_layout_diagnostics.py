"""Small isolated compiler diagnostics; never included in Phase 1 benchmark tables."""
import argparse
import copy
import sys
import time
import traceback
from pathlib import Path

import torch
import furiosa.torch as ft

from Resnet.analysis import compare_tensors
from Resnet.artifacts import write_json
from Resnet.models import load_model, make_input
from Resnet.partitioning import OutputPartitionConv
from Resnet.runtime import make_callable, measure_e2e, synchronize


def prepare(case):
    source = Path(__file__).parent / "results/bn-folded-conv-repro-20261104"
    if case == "conv1":
        model = load_model(bn_folded=True)
        conv = model.layer4[0].conv1
        activation = {}
        def capture(module, inputs):
            activation["input"] = inputs[0].detach().clone()
        handle = conv.register_forward_pre_hook(capture)
        with torch.inference_mode():
            model(make_input(1))
        handle.remove()
        return conv, activation["input"], "layer4.0.conv1"
    state = torch.load(source / "conv-state.pt", weights_only=True, map_location="cpu")
    inputs = torch.load(source / "input.pt", weights_only=True, map_location="cpu")
    conv = torch.nn.Conv2d(512, 512, 3, stride=1 if case == "stride1" else 2,
                           padding=1, bias=True, dtype=torch.bfloat16).eval()
    conv.load_state_dict(state)
    return conv, inputs, "layer4.0.conv2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("conv1", "baseline", "stride1", "split_out", "dma_only", "vision_hint"), required=True)
    parser.add_argument("--mode", choices=("eager", "compile"), default="eager")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--chunk", type=int, default=256)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch._dynamo.config.suppress_errors = False
    ft.config.set_cpu_fallback(False)
    result = {"status": "running", "case": args.case, "mode": args.mode,
              "scope": "Isolated compiler diagnostic, not ResNet50 benchmark",
              "cpu_fallback": ft.config.get_cpu_fallback(), "sdk": str(ft.version())}
    exit_code = 1
    try:
        conv, inputs, module_path = prepare(args.case)
        result.update(module=module_path, input_shape=list(inputs.shape),
                      weight_shape=list(conv.weight.shape), stride=list(conv.stride),
                      logical_weight_bytes=conv.weight.numel() * conv.weight.element_size())
        with torch.inference_mode():
            reference = conv(inputs)
        module = conv
        if args.case == "split_out":
            if args.chunk < 1:
                raise ValueError("chunk must be positive")
            module = OutputPartitionConv(conv, args.chunk)
            with torch.inference_mode():
                torch.testing.assert_close(copy.deepcopy(module).float()(inputs.float()),
                                           copy.deepcopy(conv).float()(inputs.float()), rtol=1e-4, atol=1e-4)
                result["cpu_bf16_partition_vs_original"] = compare_tensors(reference, module(inputs), classifier=False)
            result.update(partition_axis="output_channels", chunk_size=args.chunk,
                          partition_weight_shapes=[list(part.weight.shape) for part in module.parts],
                          cpu_fp32_partition_equivalence_passed=True)
        config = ft.config.get_compiler_config()
        result["compiler_config_before"] = str(config)
        if args.case == "dma_only":
            config.use_dma_bridges_only = True
        if args.case == "vision_hint":
            config.tactic_hint = ft.config.TacticHintConfig.ForVisionModel
        ft.config.set_compiler_config(config)
        result["compiler_config_effective"] = str(ft.config.get_compiler_config())
        device = torch.device("furiosa:0")
        module = module.to(device)
        synchronize(device)
        function, counter = make_callable(module, args.mode)
        result["stage"] = "first_call"
        write_json(args.run_dir / "progress.json", result)
        started = time.perf_counter()
        try:
            with ft.config.cache_context(str((args.run_dir / "compiler-cache").resolve()), summary=args.summary), torch.inference_mode():
                elapsed, output = measure_e2e(function, inputs, device)
                warm_elapsed, warm_output = measure_e2e(function, inputs, device)
            if output.dtype != torch.bfloat16 or not torch.isfinite(output).all():
                raise RuntimeError("Expected finite BF16 output")
            result.update(status="ok", output_shape=list(output.shape), first_call_ms=elapsed,
                          warm_call_ms=warm_elapsed,
                          cpu_reference_comparison=compare_tensors(reference, output, classifier=False),
                          warm_output_comparison=compare_tensors(output, warm_output, classifier=False))
            torch.save(output, args.run_dir / "output.pt")
            exit_code = 0
        finally:
            result.update(attempt_elapsed_s=time.perf_counter() - started, backend_calls=counter.calls)
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    write_json(args.run_dir / "result.json", result)
    print(f"{args.case}/{args.mode}: {result['status']}", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
