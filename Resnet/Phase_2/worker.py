"""One (mode, batch, trial) in one fresh process with its own compiler cache.

Schedule: first call (compile/load inclusive, recorded only) -> warmup -> repeats on the
primary scope, then a separate auxiliary loop for H2D / forward / D2H. A failure never
becomes a sample; the partial rows and the traceback are preserved.
"""
import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import torch
import furiosa.torch as ft

from Resnet.artifacts import write_csv, write_json
from Resnet.models import load_model, make_input, weight_metadata
from Resnet.Phase_2.measure import check_output, e2e_split, forward_to_cpu, idle_synchronize_cost
from Resnet.runtime import eager_cache_stats, make_callable, synchronize

FIELDS = ("job", "mode", "batch", "trial", "scope", "phase", "iteration", "elapsed_ms")
PRIMARY = "forward_to_cpu"


def execute(args, result, samples):
    config = json.loads(args.config.read_text())
    torch.set_num_threads(config["cpu_threads"])
    num_fusion = config.get("num_fusion", 1)
    if num_fusion != 1:
        ft.set_fusion(num_fusion)   # must precede every device lookup; once per process
    if ft.get_fusion().num_fusion != num_fusion:
        raise RuntimeError(f"PE fusion is {ft.get_fusion()}, expected num_fusion={num_fusion}")
    ft.config.set_cpu_fallback(False)
    torch._dynamo.config.suppress_errors = False
    device = torch.device(config["device"])
    if not ft.is_available() or config["device"] not in map(str, ft.get_available_devices()):
        raise RuntimeError("Configured Furiosa device is unavailable")
    result.update(config=config, num_fusion=num_fusion, fusion=str(ft.get_fusion()),
                  physical_pes=list(range(int(config["device"].split(":")[1]) * num_fusion,
                                          (int(config["device"].split(":")[1]) + 1) * num_fusion)), cpu_fallback=ft.config.get_cpu_fallback(),
                  device_configuration=str(ft.get_device_configuration()),
                  compiler_config=str(ft.config.get_compiler_config()),
                  compile_options="SDK defaults; torch.compile(model, backend=counter -> ft.backend)",
                  torch_threads=torch.get_num_threads(),
                  timing_scope={PRIMARY: "input pre-staged on NPU; forward until output.to('cpu') returned",
                                "aux": "separate loop: H2D | forward+synchronize | D2H, each closed by synchronize"})
    if result["cpu_fallback"]:
        raise RuntimeError("cpu_fallback must be False")

    def stage(name):
        result["stage"] = name
        write_json(args.output / "progress.json", result)

    cache_files = [p for p in args.cache.rglob("*") if p.is_file()] if args.cache.exists() else []
    if cache_files and not args.smoke_existing_cache:
        raise ValueError("A measured trial requires a new, empty compiler cache")
    result["cache_state"] = ("smoke_only_preexisting_cache_copy" if cache_files
                             else "new_empty_project_cache_directory")
    args.cache.mkdir(parents=True, exist_ok=True)

    stage("model_load")
    started = time.perf_counter_ns()
    model = load_model(config["seed"], bn_folded=config["bn_folded"],
                       output_split_channels=config["output_split_channels"])
    result["model_load_ms"] = (time.perf_counter_ns() - started) / 1e6
    result["weights"] = weight_metadata()
    result["model_variant"] = model._benchmark_transform["name"]
    result["state_sha256"] = model._benchmark_transform["state_sha256"]
    if result["model_variant"] != config["model_variant"]:
        raise ValueError("Configured model_variant does not match the loaded model")
    if result["state_sha256"] != config["expected_state_sha256"]:
        raise ValueError("Model state differs from the Phase 1 validated variant")
    plan = config.get("eager_batch_partition", {}).get(str(args.batch)) if args.mode == "eager" else None
    if plan:
        # Eager-only workaround for per-op compiler rejections; identical math and weights.
        from Resnet.folding import state_fingerprint
        from Resnet.partitioning import partition_batch
        model, partition = partition_batch(model, plan["modules"], plan["chunk"], inplace=True)
        if state_fingerprint(model) != config["expected_state_sha256"]:
            raise ValueError("Batch partitioning changed the model state")
        result["eager_batch_partition"] = partition
        result["model_variant"] = f"{result['model_variant']}+eager_batch_chunk{plan['chunk']}"

    stage("model_to_device")
    started = time.perf_counter_ns()
    model = model.to(device)
    synchronize(device)
    result["model_to_device_ms"] = (time.perf_counter_ns() - started) / 1e6
    function, counter = make_callable(model, args.mode)
    batch = args.batch
    cpu_input = make_input(batch, config["image_size"], config["seed"])
    stream = ft.current_stream(device)
    job = args.output.name

    def record(scope, phase, iteration, elapsed):
        samples.append(dict(job=job, mode=args.mode, batch=batch, trial=args.trial, scope=scope,
                            phase=phase, iteration=iteration, elapsed_ms=elapsed))

    def run_primary(phase, count, device_input):
        cpu_output, idle = None, True
        for iteration in range(1, count + 1):
            elapsed, _, cpu_output = forward_to_cpu(function, device_input, device)
            idle = idle and bool(stream.query())   # outside the timer
            record(PRIMARY, phase, iteration, elapsed)
        return cpu_output, idle

    with ft.config.cache_context(str(args.cache)), torch.inference_mode():
        stage("input_to_device")
        device_input = cpu_input.to(device)
        synchronize(device)
        result["input"] = {"shape": list(cpu_input.shape), "dtype": str(cpu_input.dtype),
                           "device": str(device_input.device), "kind": "synthetic normal, Phase 1 seed rule"}
        result["eager_cache_initial"] = eager_cache_stats()

        stage("first_call")
        first_wall = time.time()
        first_output, _ = run_primary("first", 1, device_input)
        result["first_call_wall"] = [first_wall, time.time()]   # for subtracting pauses by other jobs
        check_output(first_output, batch)
        result["first_call_ms"] = samples[-1]["elapsed_ms"]
        result["backend_calls_after_first"] = counter.calls
        result["eager_cache_after_first"] = eager_cache_stats()
        write_csv(args.output / "samples.csv", samples, FIELDS)

        exclusive = config.get("exclusive_measurement")
        if exclusive:
            # Parallel runners share the host: time only while every other benchmark process is stopped.
            from Resnet.Phase_2 import exclusive as gate
            stage("measure_wait")
            result["exclusive_measurement"] = gate.enter(exclusive, args.output, stage)
        else:
            stage("warmup")
        run_primary("warmup", config["warmup"], device_input)
        result["eager_cache_after_warmup"] = eager_cache_stats()

        stage("repeat")
        last_output, idle = run_primary("repeat", config["repeats"], device_input)
        check_output(last_output, batch)
        result["stream_idle_after_every_cpu_copy"] = idle
        result["eager_cache_after_repeat"] = eager_cache_stats()
        result["backend_calls_after_repeat"] = counter.calls
        result["first_vs_last_output_max_abs_diff"] = (
            first_output.float() - last_output.float()).abs().max().item()
        torch.save(last_output, args.output / "output.pt")
        write_csv(args.output / "samples.csv", samples, FIELDS)

        stage("aux_transfer")
        for phase, count in (("aux_warmup", config["aux_warmup"]), ("aux", config["aux_repeats"])):
            for iteration in range(1, count + 1):
                parts, aux_output = e2e_split(function, cpu_input, device)
                for scope, elapsed in parts.items():
                    record(scope, phase, iteration, elapsed)
        check_output(aux_output, batch)
        result["aux_vs_primary_output_max_abs_diff"] = (
            aux_output.float() - last_output.float()).abs().max().item()
        for iteration, elapsed in enumerate(idle_synchronize_cost(device), 1):
            record("idle_synchronize", "aux", iteration, elapsed)
        if exclusive:
            result["exclusive_measurement"]["timed_section_s"] = gate.leave()
        result["backend_calls"] = counter.calls
        result["eager_cache_final"] = eager_cache_stats()
    result["cache_file_count"] = sum(1 for p in args.cache.rglob("*") if p.is_file())
    result["cache_bytes"] = sum(p.stat().st_size for p in args.cache.rglob("*") if p.is_file())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("eager", "compile"), required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-existing-cache", action="store_true",
                        help="Code-path check on a copied cache; excluded from every result table")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    moved = json.loads(args.config.read_text()).get("moved_jobs", {}).get(args.output.name)
    if moved:
        # Rebalanced to another runner on free PEs mid-run; measured there, recorded here as a pointer.
        write_json(args.output / "result.json", {"status": "moved", "job": args.output.name,
                                                 "mode": args.mode, "batch": args.batch, "moved_to": moved})
        return 0
    args.cache = args.cache.resolve()
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("furiosa_torch").setLevel(logging.WARNING)
    result = {"status": "running", "job": args.output.name, "mode": args.mode, "batch": args.batch,
              "trial": args.trial, "cache_dir": str(args.cache), "smoke": args.smoke_existing_cache}
    samples = []
    started = time.perf_counter()
    exit_code = 1
    try:
        execute(args, result, samples)
        result["status"] = "ok"
        exit_code = 0
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        result["successful_sample_count"] = len(samples)
        result["process_elapsed_s"] = time.perf_counter() - started
        result["pgid"] = __import__("os").getpgid(0)
        write_csv(args.output / "samples.csv", samples, FIELDS)
        write_json(args.output / "result.json", result)
        from Resnet.Phase_2 import exclusive as gate
        gate.release()   # after result.json; no-op when the section was never entered
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
