"""One b1 mode/cache case; isolated, synchronized latency for every FX operation."""
import argparse
import gc
import json
import time
import traceback
from pathlib import Path

import torch
import furiosa.torch as ft

from Resnet.analysis import statistics_ms
from Resnet.artifacts import write_csv, write_json
from Resnet.models import load_model, make_input, weight_metadata
from Resnet.module_profiling import capture_operations, inventory, ReplayNode, measure_operation
from Resnet.runtime import make_callable, synchronize

SAMPLE_FIELDS = ("case", "mode", "cache_intent", "node", "module", "operator", "phase", "iteration", "elapsed_ms")
MODULE_FIELDS = ("case", "mode", "cache_intent", "node", "module", "operator", "kind", "status", "first_ms",
                 "n", "median_ms", "min_ms", "max_ms", "mean_ms", "stdev_ms", "cv",
                 "backend_calls", "cpu_max_abs_error", "cpu_rmse", "error_type", "error")


def execute(args, result):
    config = json.loads(args.config.read_text())
    torch.set_num_threads(config["cpu_threads"])
    ft.config.set_cpu_fallback(False)
    torch._dynamo.config.suppress_errors = False
    device = torch.device(config["device"])
    result.update(config=config, cpu_fallback=ft.config.get_cpu_fallback(),
                  compiler_config=str(ft.config.get_compiler_config()), stage="capture_cpu_activations")
    write_json(args.output / "progress.json", result)
    model = load_model(config["seed"], bn_folded=True, output_split_channels=128)
    result.update(model_transform=model._benchmark_transform, weights=weight_metadata())
    validation = json.loads((args.config.parent / config["validation_file"]).read_text())
    if validation["model_transform"] != model._benchmark_transform:
        raise ValueError("CPU validation and measured model must have identical transformed weights")
    operations, classifier_output = capture_operations(model, make_input(1, config["image_size"], config["seed"]))
    if list(classifier_output.shape) != [1, 1000] or not torch.isfinite(classifier_output.float()).all():
        raise ValueError("CPU capture must produce finite [1,1000] output")
    write_json(args.output / "module_inventory.json", inventory(operations))
    result.update(total_nodes=len(operations), measured_nodes=0, skipped_identity_nodes=0,
                  completed_nodes=0, failed_nodes=0, skipped_metadata_nodes=0)
    if args.cache_intent == "empty_project_cache" and any(args.cache.rglob("*")):
        raise ValueError("Cold case requires a new empty private compiler cache")
    cold_rows = {}
    if args.cold_result:
        import csv
        with args.cold_result.open() as stream:
            cold_rows = {row["node"]: row for row in csv.DictReader(stream)}
    samples, rows = [], []
    with ft.config.cache_context(str(args.cache)), torch.inference_mode():
        for index, item in enumerate(operations):
            base = {"case": args.output.name, "mode": args.mode, "cache_intent": args.cache_intent,
                    **{key: item[key] for key in ("node", "module", "operator", "kind")}}
            result.update(current_node=item["node"], current_module=item["module"], current_index=index + 1,
                          stage="module_prepare")
            write_json(args.output / "progress.json", result)
            row = {**base, "status": "running"}
            if item["identity"]:
                row.update(status="identity_no_kernel", n=0)
                result["skipped_identity_nodes"] += 1
            elif args.cold_result and cold_rows.get(item["node"], {}).get("status") == "metadata_only_no_binary":
                row.update(status="metadata_only_no_binary", n=0,
                           error="Cold case confirmed that standalone flatten produces no executable binary")
                result["skipped_metadata_nodes"] += 1
            elif args.cold_result and cold_rows.get(item["node"], {}).get("status") != "ok":
                row.update(status="skipped_cold_failure", error="No successful cold measurement for this node")
            else:
                node_samples = []
                try:
                    # Each node is intentionally compiled separately; avoid Dynamo's
                    # shared-frame recompile limit across unrelated diagnostic nodes.
                    if args.mode == "compile":
                        torch._dynamo.reset()
                    replay = ReplayNode(item["kind"], item["operation"]).eval().to(device)
                    synchronize(device)
                    function, counter = make_callable(replay, args.mode)
                    for phase, count in (("first", 1), ("warmup", config["warmup"]), ("measure", config["iterations"])):
                        result["stage"] = f"{item['node']}:{phase}"
                        write_json(args.output / "progress.json", result)
                        for iteration in range(count):
                            elapsed, output = measure_operation(function, item["args"], item["kwargs"], device)
                            node_samples.append({**base, "phase": phase, "iteration": iteration, "elapsed_ms": elapsed})
                    cpu_output = output.to("cpu")
                    synchronize(device)
                    if output.device.type != "furiosa" or cpu_output.dtype != torch.bfloat16:
                        raise ValueError("Expected NPU BF16 module output")
                    if cpu_output.shape != item["reference"].shape or not torch.isfinite(cpu_output.float()).all():
                        raise ValueError("Module output shape/finiteness validation failed")
                    delta = cpu_output.float() - item["reference"].float()
                    row.update(status="ok", first_ms=node_samples[0]["elapsed_ms"],
                               backend_calls=counter.calls, cpu_max_abs_error=delta.abs().max().item(),
                               cpu_rmse=delta.square().mean().sqrt().item(),
                               **statistics_ms([sample["elapsed_ms"] for sample in node_samples if sample["phase"] == "measure"]))
                    result["measured_nodes"] += 1
                    del output, function, replay
                    gc.collect()
                except Exception as error:
                    row.update(error_type=type(error).__name__, error=str(error))
                    if args.mode == "compile" and item["operator"] == "flatten" and isinstance(error, ValueError) and "RunnableSequence requires at least one runnable with a binary" in str(error):
                        row.update(status="metadata_only_no_binary", n=0)
                        result["skipped_metadata_nodes"] += 1
                    else:
                        row["status"] = "failed"
                        result["failed_nodes"] += 1
                        traceback.print_exc()
                samples.extend(node_samples)
            rows.append(row)
            result["completed_nodes"] = index + 1
            write_csv(args.output / "samples.csv", samples, SAMPLE_FIELDS)
            write_csv(args.output / "modules.csv", rows, MODULE_FIELDS)
            write_json(args.output / "progress.json", result)
            print(f"{index + 1}/{len(operations)} {item['node']} ({item['module']}): {row['status']}", flush=True)
    result.update(status="ok" if not result["failed_nodes"] and not any(r["status"] == "skipped_cold_failure" for r in rows) else "partial",
                  stage="finished")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("eager", "compile"), required=True)
    parser.add_argument("--cache-intent", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cold-result", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"status": "running", "mode": args.mode, "batch": 1, "cache_intent": args.cache_intent,
              "measurement_scope": "isolated_operation_forward_with_device_synchronization",
              "compile_scope": "per_node; cross-module fusion excluded; not full-model Compile latency"}
    started = time.monotonic()
    try:
        execute(args, result)
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
    finally:
        result["process_elapsed_s"] = time.monotonic() - started
        write_json(args.output / "result.json", result)
        write_json(args.output / "progress.json", result)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
