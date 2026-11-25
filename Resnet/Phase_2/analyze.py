"""Rebuild every Phase 2 table from the raw per-job files (no NPU access).

Only jobs that finished with status ``ok`` on a new cache contribute to statistics.
First calls and warmups are reported separately and never enter steady-state numbers.
"""
import argparse
import csv
import json
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

from Resnet.artifacts import write_csv, write_json
from Resnet.Phase_2.stats import scaling_efficiency, speedup, summarize, throughput

PROJECT = Path(__file__).resolve().parents[1]
PRIMARY = "forward_to_cpu"
AUX_SCOPES = ("h2d", "forward_sync", "d2h", "e2e", "idle_synchronize")
STAT_FIELDS = ("n", "mean_ms", "stdev_ms", "median_ms", "p95_ms", "min_ms", "max_ms", "cv")
PHASE1_RUN = PROJECT / "Phase_1" / "results" / "bn-folded-split128-2h-20261104-162935"
PHASE1_CPU_REFERENCE = PROJECT / "Phase_1" / "evidence" / "full-model-split128-validation"


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def load_jobs(root):
    jobs = []
    for path in sorted((root / "jobs").glob("*/result.json")):
        if ".interrupted-" in path.parent.name:
            continue
        job = json.loads(path.read_text())
        job["directory"] = path.parent
        job.setdefault("job", path.parent.name)
        samples = path.parent / "samples.csv"
        job["samples"] = read_csv(samples) if samples.exists() else []
        for row in job["samples"]:
            row["elapsed_ms"] = float(row["elapsed_ms"])
        jobs.append(job)
    return jobs


def values(job, scope, phase):
    return [row["elapsed_ms"] for row in job["samples"] if row["scope"] == scope and row["phase"] == phase]


def lru_misses_per_forward(job):
    before, after = job.get("eager_cache_after_warmup"), job.get("eager_cache_after_repeat")
    if not before or not after:
        return None
    key = "_compile_and_load_cached"
    return (after[key]["misses"] - before[key]["misses"]) / job["config"]["repeats"]


def compare_outputs(left, right):
    import torch
    from Resnet.analysis import compare_tensors
    a = torch.load(left, weights_only=True)
    b = torch.load(right, weights_only=True)
    result = compare_tensors(a, b)
    if result["status"] == "measured":
        result["bitwise_equal"] = bool(torch.equal(a, b))
        result["top1_matches"] = int((a.float().argmax(1) == b.float().argmax(1)).sum())
        result["samples"] = a.shape[0]
    return result


def aggregate(root):
    root = Path(root)
    config = json.loads((root / "config.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text()) if (root / "manifest.json").exists() else {}
    jobs = load_jobs(root)
    ok = [job for job in jobs if job["status"] == "ok" and not job.get("smoke")]
    core_trials = config["core_trials"]

    raw = [{**row, "job_status": job["status"]} for job in jobs for row in job["samples"]]
    write_csv(root / "raw_latency.csv", raw,
              ("job", "job_status", "mode", "batch", "trial", "scope", "phase", "iteration", "elapsed_ms"))

    # Per trial (one fresh process).
    trial_rows = []
    for job in ok:
        row = {"job": job["job"], "mode": job["mode"], "batch": job["batch"], "trial": job["trial"],
               **summarize(values(job, PRIMARY, "repeat"))}
        row.update(throughput_samples_per_s=throughput(job["batch"], row["mean_ms"]),
                   first_call_s=job["first_call_ms"] / 1000,
                   warmup_sum_ms=sum(values(job, PRIMARY, "warmup")),
                   model_load_ms=job["model_load_ms"], model_to_device_ms=job["model_to_device_ms"],
                   process_elapsed_s=job["process_elapsed_s"], backend_calls=job["backend_calls"],
                   eager_load_lru_misses_per_forward=lru_misses_per_forward(job),
                   stream_idle_after_every_cpu_copy=job["stream_idle_after_every_cpu_copy"],
                   first_vs_last_output_max_abs_diff=job["first_vs_last_output_max_abs_diff"],
                   cache_bytes=job["cache_bytes"])
        for scope in AUX_SCOPES:
            aux = summarize(values(job, scope, "aux"))
            row[f"{scope}_mean_ms"], row[f"{scope}_median_ms"] = aux["mean_ms"], aux["median_ms"]
        trial_rows.append(row)
    trial_rows.sort(key=lambda r: (r["batch"], r["mode"], r["trial"]))
    trial_fields = ("job", "mode", "batch", "trial") + STAT_FIELDS + (
        "throughput_samples_per_s", "first_call_s", "warmup_sum_ms", "model_load_ms", "model_to_device_ms",
        "process_elapsed_s", "backend_calls", "eager_load_lru_misses_per_forward",
        "stream_idle_after_every_cpu_copy", "first_vs_last_output_max_abs_diff", "cache_bytes") + tuple(
        f"{scope}_{kind}_ms" for scope in AUX_SCOPES for kind in ("mean", "median"))
    write_csv(root / "trial_summary.csv", trial_rows, trial_fields)

    # Per condition, pooled over the scaling trials (trial <= core_trials for every batch).
    def pooled(mode, batch, scope, phase, trial_limit):
        selected = [job for job in ok if job["mode"] == mode and job["batch"] == batch
                    and job["trial"] <= trial_limit]
        merged = [x for job in selected for x in values(job, scope, phase)]
        if not merged:
            return None
        means = [statistics.fmean(values(job, scope, phase)) for job in selected]
        return {**summarize(merged), "trials": len(selected),
                "trial_means_mean_ms": statistics.fmean(means),
                "trial_means_stdev_ms": statistics.stdev(means) if len(means) > 1 else 0.0,
                "trial_means_min_ms": min(means), "trial_means_max_ms": max(means)}

    batches = sorted({job["batch"] for job in ok})
    condition_rows = []
    for batch in batches:
        for mode in ("eager", "compile"):
            for scope, phase in ((PRIMARY, "repeat"),) + tuple((scope, "aux") for scope in AUX_SCOPES):
                stats = pooled(mode, batch, scope, phase, core_trials)
                if stats:
                    condition_rows.append({"mode": mode, "batch": batch, "scope": scope, **stats})
    write_csv(root / "condition_summary.csv", condition_rows, ("mode", "batch", "scope", "trials") + STAT_FIELDS + (
        "trial_means_mean_ms", "trial_means_stdev_ms", "trial_means_min_ms", "trial_means_max_ms"))

    # Batch scaling: latency, throughput, speedup and efficiency against batch 1.
    primary = {(row["mode"], row["batch"]): row for row in condition_rows if row["scope"] == PRIMARY}
    trial_mean = {(row["mode"], row["batch"], row["trial"]): row["mean_ms"] for row in trial_rows}
    scaling_rows = []
    for batch in batches:
        row = {"batch": batch}
        for mode in ("eager", "compile"):
            stats = primary.get((mode, batch))
            if not stats:
                continue
            row.update({f"{mode}_{name}": stats[name] for name in ("trials", "n", "mean_ms", "stdev_ms", "median_ms", "p95_ms")})
            row[f"{mode}_throughput_samples_per_s"] = throughput(batch, stats["mean_ms"])
            row[f"{mode}_ms_per_sample"] = stats["mean_ms"] / batch
            base = primary.get((mode, 1))
            if base:
                row[f"{mode}_throughput_gain_vs_b1"] = (throughput(batch, stats["mean_ms"])
                                                        / throughput(1, base["mean_ms"]))
                row[f"{mode}_scaling_efficiency"] = scaling_efficiency(batch, stats["mean_ms"], 1, base["mean_ms"])
        if "eager_mean_ms" in row and "compile_mean_ms" in row:
            row["speedup_mean"] = speedup(row["eager_mean_ms"], row["compile_mean_ms"])
            row["speedup_median"] = speedup(row["eager_median_ms"], row["compile_median_ms"])
            paired = [speedup(trial_mean[("eager", batch, t)], trial_mean[("compile", batch, t)])
                      for t in range(1, core_trials + 1)
                      if ("eager", batch, t) in trial_mean and ("compile", batch, t) in trial_mean]
            if paired:
                row.update(speedup_trial_min=min(paired), speedup_trial_max=max(paired), speedup_trials=len(paired))
        scaling_rows.append(row)
    scaling_fields = ("batch",) + tuple(
        f"{mode}_{name}" for mode in ("eager", "compile") for name in (
            "trials", "n", "mean_ms", "stdev_ms", "median_ms", "p95_ms", "throughput_samples_per_s",
            "ms_per_sample", "throughput_gain_vs_b1", "scaling_efficiency")) + (
        "speedup_mean", "speedup_median", "speedup_trial_min", "speedup_trial_max", "speedup_trials")
    write_csv(root / "batch_scaling.csv", scaling_rows, scaling_fields)

    # Revalidation of the Phase 1 batch-1 result: every available independent set.
    revalidation = []
    batch = config["revalidation_batch"]
    for mode in ("eager", "compile"):
        sets = sorted((job for job in ok if job["mode"] == mode and job["batch"] == batch), key=lambda j: j["trial"])
        for job in sets:
            revalidation.append({"mode": mode, "batch": batch, "set": job["trial"],
                                 **summarize(values(job, PRIMARY, "repeat")),
                                 "forward_sync_median_ms": statistics.median(values(job, "forward_sync", "aux")),
                                 "e2e_median_ms": statistics.median(values(job, "e2e", "aux"))})
        stats = pooled(mode, batch, PRIMARY, "repeat", config["revalidation_trials"])
        if stats:
            revalidation.append({"mode": mode, "batch": batch, "set": f"pooled_{stats['trials']}_sets", **stats,
                                 "forward_sync_median_ms": pooled(mode, batch, "forward_sync", "aux", config["revalidation_trials"])["median_ms"],
                                 "e2e_median_ms": pooled(mode, batch, "e2e", "aux", config["revalidation_trials"])["median_ms"]})
    phase1 = PHASE1_RUN / "summary.csv"
    if phase1.exists():
        for row in read_csv(phase1):
            if row["cache_intent"] == "empty_project_cache" and row["batch"] == str(batch):
                revalidation.append({"mode": row["mode"], "batch": batch,
                                     "set": f"phase1_{row['scope']}_n{row['n']}",
                                     **{name: row[name] for name in ("n", "mean_ms", "stdev_ms", "median_ms", "min_ms", "max_ms", "cv")}})
    write_csv(root / "revalidation_b1.csv", revalidation, ("mode", "batch", "set") + STAT_FIELDS + (
        "forward_sync_median_ms", "e2e_median_ms"))

    # First call (compile/load inclusive): recorded, excluded from every comparison above.
    initial = []
    for batch in batches:
        for mode in ("eager", "compile"):
            first = [row["first_call_s"] for row in trial_rows if row["mode"] == mode and row["batch"] == batch]
            if first:
                initial.append({"mode": mode, "batch": batch, "trials": len(first),
                                "first_call_median_s": statistics.median(first),
                                "first_call_min_s": min(first), "first_call_max_s": max(first)})
    write_csv(root / "initial_costs.csv", initial,
              ("mode", "batch", "trials", "first_call_median_s", "first_call_min_s", "first_call_max_s"))

    # Output agreement. No dataset accuracy claim; numbers only.
    correctness = []
    fields = ("kind", "batch", "left", "right", "status", "samples", "top1_matches", "bitwise_equal",
              "max_abs_error", "mean_abs_error", "rmse", "cosine_min")
    by_key = {(job["mode"], job["batch"], job["trial"]): job["directory"] / "output.pt" for job in ok}
    for (mode, batch, trial), path in sorted(by_key.items(), key=lambda item: (item[0][1], item[0][2], item[0][0])):
        if mode == "eager" and ("compile", batch, trial) in by_key:
            correctness.append({"kind": "eager_vs_compile_same_trial", "batch": batch,
                                "left": f"eager-t{trial}", "right": f"compile-t{trial}",
                                **compare_outputs(path, by_key[("compile", batch, trial)])})
        if trial > 1 and (mode, batch, 1) in by_key:
            correctness.append({"kind": "same_mode_across_trials", "batch": batch,
                                "left": f"{mode}-t1", "right": f"{mode}-t{trial}",
                                **compare_outputs(by_key[(mode, batch, 1)], path)})
        reference = PHASE1_CPU_REFERENCE / f"cpu-output-b{batch}.pt"
        if trial == 1 and reference.exists():
            correctness.append({"kind": "phase1_cpu_bf16_reference_vs_npu", "batch": batch,
                                "left": "cpu-bf16", "right": f"{mode}-t1", **compare_outputs(reference, path)})
    write_csv(root / "correctness.csv", correctness, fields)

    # Failures with their logs, and coverage of the whole plan.
    errors = root / "errors"
    failures = []
    for job in jobs:
        if job["status"] == "ok":
            continue
        errors.mkdir(exist_ok=True)
        log = root / "jobs" / f"{job['job']}.log"
        if log.exists():
            shutil.copy2(log, errors / log.name)
        if job.get("traceback"):
            (errors / f"{job['job']}.traceback.txt").write_text(job["traceback"])
        failures.append({"job": job["job"], "mode": job.get("mode"), "batch": job.get("batch"),
                         "trial": job.get("trial"), "status": job["status"], "stage": job.get("stage"),
                         "error_type": job.get("error_type"),
                         "error": (job.get("error") or "").strip().splitlines()[0][:500] if job.get("error") else "",
                         "log": f"errors/{log.name}"})
    write_csv(root / "failures.csv", failures,
              ("job", "mode", "batch", "trial", "status", "stage", "error_type", "error", "log"))
    status = {job["job"]: job["status"] for job in jobs}
    skipped = {item["name"]: item["reason"] for item in manifest.get("skipped", [])}
    coverage = [{"job": item["name"], "phase": item["phase"], "mode": item["mode"], "batch": item["batch"],
                 "trial": item["trial"],
                 "state": status.get(item["name"], "skipped" if item["name"] in skipped else "not_run"),
                 "note": skipped.get(item["name"], "")} for item in manifest.get("plan", [])]
    write_csv(root / "coverage.csv", coverage, ("job", "phase", "mode", "batch", "trial", "state", "note"))
    tables = {"ok_jobs": len(ok), "failed_jobs": len(failures), "batches_measured": batches,
              "planned": len(coverage), "not_run": sum(row["state"] in ("not_run", "skipped") for row in coverage)}
    write_json(root / "analysis_notes.json", {
        **tables, "num_fusion": config.get("num_fusion", 1),
        "phase1_comparison": "Phase 1 used num_fusion=1" if config.get("num_fusion", 1) != 1 else "same fusion as Phase 1",
        "primary_scope": "input pre-staged on NPU; forward until output.to('cpu') returned",
        "scaling_trials": f"trial <= {core_trials} for every batch; pooled statistics use all repeats of those trials",
        "stdev": "sample standard deviation (n-1)", "p95": "linear interpolation",
        "speedup": "eager mean / compile mean", "throughput": "batch * 1000 / mean latency",
        "scaling_efficiency": "(throughput(B) / throughput(1)) / B",
        "numerical_acceptance": "not defined; metrics only, no ImageNet accuracy claim"})
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.run_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
