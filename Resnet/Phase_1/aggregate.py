"""Rebuild tidy tables from successful jobs, retaining explicit failure coverage."""
import argparse
import csv
import json
import hashlib
from collections import defaultdict
from pathlib import Path

from Resnet.analysis import break_even, compare_tensors, statistics_ms
from Resnet.artifacts import write_csv, write_json


def aggregate(root):
    root = Path(root)
    project = Path(__file__).resolve().parents[1]
    write_json(root / "analysis_source_hashes.json", {
        relative: hashlib.sha256((project / relative).read_bytes()).hexdigest()
        for relative in ("analysis.py", "artifacts.py", "Phase_1/aggregate.py")
    })
    jobs, samples, failures = [], [], []
    for path in sorted((root / "jobs").glob("*/result.json")):
        job = json.loads(path.read_text())
        job["directory"] = path.parent
        job["model_variant"] = job.get("model_variant", job.get("model_transform", {}).get("name", "identity"))
        jobs.append(job)
        if job["status"] != "ok":
            failures.append({"job": path.parent.name, "mode": job.get("mode"),
                             "model_variant": job["model_variant"],
                             "target": job.get("target"), "status": job["status"],
                             "stage": job.get("stage"), "error_type": job.get("error_type"),
                             "error": job.get("error", "").splitlines()[0]})
            continue
        if job.get("profile_enabled"):
            continue
        with (path.parent / "samples.csv").open() as stream:
            for sample in csv.DictReader(stream):
                sample["job"] = path.parent.name
                sample["model_variant"] = job["model_variant"]
                samples.append(sample)
    groups = defaultdict(list)
    group_fields = ("model_variant", "target", "mode", "workload", "cache_intent", "scope", "batch")
    for sample in samples:
        if sample["phase"] == "measure":
            key = tuple(sample[field] for field in group_fields)
            groups[key].append(float(sample["elapsed_ms"]))
    summaries = []
    for key, values in sorted(groups.items()):
        row = dict(zip(group_fields, key))
        row.update(statistics_ms(values))
        row["median_images_per_s"] = int(row["batch"]) * 1000 / row["median_ms"]
        summaries.append(row)
    write_csv(root / "summary.csv", summaries, group_fields + (
        "n", "median_ms", "min_ms", "max_ms", "mean_ms", "stdev_ms", "cv", "median_images_per_s"))
    write_csv(root / "failures.csv", failures,
              ("job", "model_variant", "mode", "target", "status", "stage", "error_type", "error"))

    cumulative, workload_totals = [], []
    for job in jobs:
        if job["status"] != "ok":
            continue
        rows = [sample for sample in samples if sample["job"] == job["directory"].name and sample["scope"] == "e2e"]
        total_ms = 0.0
        for call, sample in enumerate(rows, 1):
            total_ms += float(sample["elapsed_ms"])
            cumulative.append({**sample, "call": call, "cumulative_ms": total_ms})
        workload_totals.append({"job": job["directory"].name, "target": job["target"],
                                "model_variant": job["model_variant"],
                                "mode": job["mode"], "workload": job["workload"],
                                "cache_intent": job["cache_intent"], "calls": len(rows),
                                "total_actual_e2e_ms": total_ms,
                                "backend_calls": job.get("backend_calls")})
    write_csv(root / "cumulative.csv", cumulative,
              ("job", "model_variant", "target", "mode", "workload", "cache_intent", "phase", "cycle", "step", "batch", "call", "elapsed_ms", "cumulative_ms"))
    write_csv(root / "workload_totals.csv", workload_totals,
              ("job", "model_variant", "target", "mode", "workload", "cache_intent", "calls", "total_actual_e2e_ms", "backend_calls"))

    pairs = defaultdict(dict)
    for job in jobs:
        if job["status"] == "ok" and not job.get("profile_enabled"):
            key = (job["target"], job["workload"], job["cache_intent"], job["batch"], job["model_variant"])
            pairs[key][job["mode"]] = job
    correctness, amortization = [], []
    for key, modes in sorted(pairs.items()):
        if set(modes) != {"eager", "compile"}:
            continue
        eager, compiled = modes["eager"], modes["compile"]
        if eager["weights"]["sha256"] != compiled["weights"]["sha256"] or eager["config"] != compiled["config"]:
            raise ValueError("Refusing to compare runs with different weights/configurations")
        if eager.get("model_transform") != compiled.get("model_transform"):
            raise ValueError("Refusing to compare different model transformations or folded weights")
        base = dict(zip(("target", "workload", "cache_intent", "job_batch", "model_variant"), key))
        import torch
        for path in sorted(eager["directory"].glob("output-b*.pt")):
            other = compiled["directory"] / path.name
            if not other.exists():
                continue
            comparison = compare_tensors(torch.load(path, map_location="cpu", weights_only=True),
                                         torch.load(other, map_location="cpu", weights_only=True),
                                         classifier=key[0] == "resnet50")
            correctness.append({**base, "batch": int(path.stem.split("-b")[1]), **comparison})
        if key[1] != "fixed":
            continue
        timing = {}
        for mode, job in modes.items():
            rows = [sample for sample in samples if sample["job"] == job["directory"].name
                    and sample["scope"] == "e2e"]
            steady = statistics_ms([float(sample["elapsed_ms"]) for sample in rows if sample["phase"] == "measure"])["median_ms"]
            initial = [float(sample["elapsed_ms"]) for sample in rows if sample["phase"] in ("first", "warmup")]
            # A startup estimate, NOT an isolated measurement of compiler codegen.
            extra = max(0.0, sum(initial) - len(initial) * steady)
            setup = sum(job[field] for field in ("model_load_ms", "model_to_device_ms", "mode_wrapper_ms"))
            timing[mode] = {"steady": steady, "initial": setup + extra}
        amortization.append({**base, "eager_median_ms": timing["eager"]["steady"],
                             "compile_median_ms": timing["compile"]["steady"],
                             "eager_startup_estimate_ms": timing["eager"]["initial"],
                             "compile_startup_estimate_ms": timing["compile"]["initial"],
                             "speedup": timing["eager"]["steady"] / timing["compile"]["steady"],
                             **break_even(timing["eager"]["initial"], timing["compile"]["initial"],
                                          timing["eager"]["steady"], timing["compile"]["steady"])})
    write_json(root / "correctness.json", correctness)
    write_csv(root / "correctness.csv", correctness, ("model_variant", "target", "workload", "cache_intent", "batch",
              "status", "max_abs_error", "mean_abs_error", "rmse", "cosine_mean", "cosine_min",
              "cosine_undefined_samples", "prediction_agreement", "acceptance"))
    write_csv(root / "amortization_estimates.csv", amortization, (
        "model_variant", "target", "workload", "cache_intent", "job_batch", "eager_median_ms", "compile_median_ms",
        "eager_startup_estimate_ms", "compile_startup_estimate_ms", "speedup", "status",
        "boundary_calls", "boundary_ceil_calls", "first_strict_compile_win_calls"))
    write_json(root / "analysis_notes.json", {
        "successful_jobs": sum(job["status"] == "ok" for job in jobs), "failed_jobs": len(failures),
        "failed_job_samples_excluded": True,
        "startup_estimate": "model load + model transfer + mode wrapper + max(0, first/warmup E2E sum - call count * measured median)",
        "startup_estimate_is_not_pure_compile_time": True,
        "mixed_totals": "Actual measured E2E sums in recorded order; initial/fixed warmup/steady phases retained",
        "cache_hits": "Not established by timings, same path, or backend entry counts",
        "correctness": "Synthetic tensors, numerical agreement only; no dataset accuracy or acceptance threshold",
        "conv1": "Diagnostic submodule only; never interpret as ResNet50 inference performance",
    })
    return {"summary_rows": len(summaries), "failures": len(failures), "comparisons": len(correctness)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    print(aggregate(parser.parse_args().run_dir))
