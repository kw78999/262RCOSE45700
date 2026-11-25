"""One mode/target/workload in one fresh process; failures never become samples."""
import argparse
import logging
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import torch
import furiosa.torch as ft

from Resnet.artifacts import write_csv, write_json
from Resnet.models import load_model, make_input, weight_metadata
from Resnet.runtime import eager_cache_stats, make_callable, measure_e2e, measure_forward, synchronize

FIELDS = ("mode", "target", "model_variant", "cache_intent", "workload", "scope", "phase", "cycle", "step",
          "batch", "image_size", "elapsed_ms", "images_per_s", "backend_calls", "backend_calls_delta")


def inventory(root):
    return {str(p.relative_to(root)): {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in root.rglob("*") if p.is_file()}


def execute(args, result):
    import json
    config = json.loads(args.config.read_text())
    torch.set_num_threads(config["cpu_threads"])
    ft.config.set_cpu_fallback(False)
    torch._dynamo.config.suppress_errors = False
    result.update(config=config, cpu_fallback=ft.config.get_cpu_fallback(),
                  compiler_config=str(ft.config.get_compiler_config()),
                  compile_options="SDK defaults; counting wrapper delegates unchanged to ft.backend")
    device = torch.device(config["device"])
    stage = "model_load"
    result["stage"] = stage
    started = time.perf_counter_ns()
    model = load_model(config["seed"], bn_folded=config.get("bn_folded", False),
                       output_split_channels=config.get("output_split_channels"))
    result["model_load_ms"] = (time.perf_counter_ns() - started) / 1e6
    result["weights"] = weight_metadata()
    result["model_transform"] = model._benchmark_transform
    result["model_variant"] = model._benchmark_transform["name"]
    if config.get("model_variant", result["model_variant"]) != result["model_variant"]:
        raise ValueError("Configured model_variant does not match the loaded model")
    if args.target == "conv1":
        model = model.conv1
        result["interpretation"] = "Original ResNet50 conv1 only; diagnostic, NOT full-model performance"
    elif args.target == "bn1":
        model = model.bn1
        result["interpretation"] = "Isolated original bn1 support check; synthetic activation"
    else:
        result["interpretation"] = ("BN-folded ResNet50 with layer4.0/1/2.conv2 output-channel split128 in both modes"
                                    if config.get("output_split_channels") else
                                    "Pretrained ResNet50 with identical FP32 Conv/BN folding before BF16 in both modes"
                                    if config.get("bn_folded", False) else "Unmodified pretrained ResNet50")
    result["stage"] = "model_to_device"
    started = time.perf_counter_ns()
    model = model.to(device)
    synchronize(device)
    result["model_to_device_ms"] = (time.perf_counter_ns() - started) / 1e6
    started = time.perf_counter_ns()
    function, counter = make_callable(model, args.mode)
    result["mode_wrapper_ms"] = (time.perf_counter_ns() - started) / 1e6
    sequence = [args.batch] if args.workload == "fixed" else config["mixed_batches"]
    result["shape_call_frequencies_per_cycle"] = dict(Counter(sequence))
    inputs = {batch: make_input(batch, config["image_size"], config["seed"]) for batch in set(sequence)}
    if args.target == "bn1":
        generator = torch.Generator().manual_seed(config["seed"])
        inputs = {batch: torch.randn(batch, 64, 112, 112, generator=generator).to(torch.bfloat16)
                  for batch in set(sequence)}
    samples, observations, outputs = [], [], {}
    cache_before = inventory(args.cache)
    if args.cache_intent == "empty_project_cache" and cache_before:
        raise ValueError("Cold job requires a new empty project cache; refusing to delete existing files")
    if args.cache_intent == "disk_reuse_attempt" and not cache_before:
        raise ValueError("Reuse job requires nonempty compiler cache from a successful cold job")
    result["cache_files_before"] = cache_before
    result["stage"] = "first_call"

    def sample(batch, phase, cycle, step, scope="e2e", device_input=None):
        before = eager_cache_stats()
        calls_before = counter.calls
        result["stage"] = f"{phase}:{scope}:b{batch}:cycle{cycle}:step{step}"
        result["attempted_sample"] = {"batch": batch, "phase": phase, "scope": scope,
                                      "cycle": cycle, "step": step}
        write_json(args.output / "progress.json", result)
        if scope == "e2e":
            elapsed, cpu_output = measure_e2e(function, inputs[batch], device)
            if args.target == "resnet50" and list(cpu_output.shape) != [batch, 1000]:
                raise RuntimeError(f"Unexpected ResNet50 output shape: {cpu_output.shape}")
            if cpu_output.dtype != torch.bfloat16 or not torch.isfinite(cpu_output.float()).all():
                raise RuntimeError("Output must be finite BF16")
            outputs[batch] = cpu_output
        else:
            elapsed, _ = measure_forward(function, device_input, device)
        samples.append(dict(mode=args.mode, target=args.target, model_variant=result["model_variant"], cache_intent=args.cache_intent,
                            workload=args.workload, scope=scope, phase=phase, cycle=cycle, step=step,
                            batch=batch, image_size=config["image_size"], elapsed_ms=elapsed,
                            images_per_s=batch * 1000 / elapsed, backend_calls=counter.calls,
                            backend_calls_delta=counter.calls - calls_before))
        observations.append({"sample_index": len(samples) - 1, "eager_before": before,
                             "eager_after": eager_cache_stats(), "backend_calls": counter.calls})
        write_csv(args.output / "samples.csv", samples, FIELDS)
        write_json(args.output / "observations.json", observations)

    try:
        with ft.config.cache_context(str(args.cache)), torch.inference_mode():
            for step, batch in enumerate(sequence):
                sample(batch, "first" if args.workload == "fixed" else "first_cycle", -1, step)
            for cycle in range(config["warmup"]):
                for step, batch in enumerate(sequence):
                    sample(batch, "warmup", cycle, step)
            for cycle in range(config["iterations"]):
                for step, batch in enumerate(sequence):
                    sample(batch, "measure", cycle, step)
            # Separate scope after E2E: a forward-only host latency, NOT NPU kernel time.
            if args.workload == "fixed":
                device_input = inputs[args.batch].to(device)
                synchronize(device)
                for cycle in range(config["iterations"]):
                    sample(args.batch, "measure", cycle, 0, "forward", device_input)
            if args.profile:
                from furiosa.torch.profiler import Profiler, RecordFunction
                device_input = inputs[sequence[0]].to(device)
                synchronize(device)
                result["stage"] = "profile"
                with Profiler(str(args.output / "furiosa-trace.json")):
                    with RecordFunction(f"{args.target}/{args.mode}/forward"):
                        function(device_input)
                        synchronize(device)
                # Separate instrumented eager execution only: module names cannot be
                # mapped directly to fused Compile kernels or summed as E2E latency.
                if args.target == "resnet50" and args.mode == "eager":
                    from contextlib import ExitStack
                    spans, handles = {}, []
                    def pre(name):
                        def enter(module, inputs):
                            stack = ExitStack()
                            stack.enter_context(RecordFunction(name))
                            spans.setdefault(name, []).append(stack)
                        return enter
                    def post(name):
                        def leave(module, inputs, output):
                            spans[name].pop().close()
                        return leave
                    try:
                        for name in ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc"):
                            module = model.get_submodule(name)
                            handles.extend([module.register_forward_pre_hook(pre(name)),
                                            module.register_forward_hook(post(name), always_call=True)])
                        with Profiler(str(args.output / "eager-modules-trace.json")):
                            function(device_input)
                            synchronize(device)
                    finally:
                        for handle in handles:
                            handle.remove()
            for batch, output in outputs.items():
                torch.save(output, args.output / f"output-b{batch}.pt")
    finally:
        result["backend_calls"] = counter.calls
        result["cache_files_after"] = inventory(args.cache)
        result["cache_evidence_limit"] = "Same disk path and file changes are observations, not verified disk hits or compiler invocation counts"
        result["successful_sample_count"] = len(samples)
        write_csv(args.output / "samples.csv", samples, FIELDS)
        write_json(args.output / "observations.json", observations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("eager", "compile"), required=True)
    parser.add_argument("--target", choices=("resnet50", "conv1", "bn1"), default="resnet50")
    parser.add_argument("--workload", choices=("fixed", "mixed"), default="fixed")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--cache-intent", choices=("empty_project_cache", "disk_reuse_attempt"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    args.cache = args.cache.resolve()
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("furiosa_torch").setLevel(logging.WARNING)
    result = {"status": "running", "mode": args.mode, "target": args.target,
              "workload": args.workload, "batch": args.batch, "cache_intent": args.cache_intent,
              "cache_dir": str(args.cache), "profile_enabled": args.profile}
    started = time.perf_counter()
    exit_code = 1
    try:
        execute(args, result)
        result["status"] = "ok"
        exit_code = 0
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        result["process_elapsed_s"] = time.perf_counter() - started
        write_json(args.output / "result.json", result)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
