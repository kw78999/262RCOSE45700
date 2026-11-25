"""Four b1 cases for isolated module/operation latency diagnostics."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from Resnet.artifacts import write_csv, write_json
from Resnet.Phase_1.progress import write_progress
from Resnet.Phase_1.run import launch, validate_config

PROJECT = Path(__file__).resolve().parents[1]


def aggregate_modules(root):
    rows = []
    for path in sorted((root / "jobs").glob("*/modules.csv")):
        with path.open() as stream:
            rows.extend(csv.DictReader(stream))
    fields = list(rows[0]) if rows else ["case", "mode", "cache_intent", "node", "module", "status", "median_ms"]
    write_csv(root / "module_summary.csv", rows, fields)
    pairs = {}
    for row in rows:
        pairs.setdefault((row["cache_intent"], row["node"]), {})[row["mode"]] = row
    comparisons = []
    for (cache, node), modes in sorted(pairs.items()):
        if set(modes) != {"eager", "compile"}:
            continue
        eager, compiled = modes["eager"], modes["compile"]
        row = {"cache_intent": cache, "node": node, "module": eager["module"],
               "eager_status": eager["status"], "compile_status": compiled["status"]}
        if eager["status"] == compiled["status"] == "ok":
            row.update(eager_first_ms=eager["first_ms"], compile_first_ms=compiled["first_ms"],
                       eager_median_ms=eager["median_ms"], compile_median_ms=compiled["median_ms"],
                       isolated_speedup=float(eager["median_ms"]) / float(compiled["median_ms"]))
        comparisons.append(row)
    write_csv(root / "module_comparison.csv", comparisons,
              ("cache_intent", "node", "module", "eager_status", "compile_status", "eager_first_ms",
               "compile_first_ms", "eager_median_ms", "compile_median_ms", "isolated_speedup"))
    return {"module_rows": len(rows), "successful_node_measurements": sum(r["status"] == "ok" for r in rows),
            "failed_node_measurements": sum(r["status"] == "failed" for r in rows), "comparison_rows": len(comparisons)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    if config["batches"] != [1] or config["mixed_batches"] != [1]:
        parser.error("This diagnostic measures only batch 1")
    root = args.run_dir.resolve()
    with (PROJECT / ".rngd-benchmark.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root.mkdir(parents=True, exist_ok=False)
        (root / "jobs").mkdir()
        (root / "evidence").mkdir()
        validation = args.config.resolve().parent / config["validation_file"]
        shutil.copy2(validation, root / "evidence/transform-validation.json")
        config["validation_file"] = "evidence/transform-validation.json"
        write_json(root / "config.json", config)
        source = root / "source"
        shutil.copytree(PROJECT, source / "Resnet", ignore=shutil.ignore_patterns(
            "results", "evidence", "__pycache__", ".git", ".cache", ".tmp", ".*lock", "*.pyc"))
        write_json(root / "source_hashes.json", {str(p.relative_to(source / "Resnet")): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (source / "Resnet").rglob("*.py")})
        from Resnet.environment import collect
        collect(root / "evidence", include_installed_sources=False)
        manifest = {"started_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
                    "target": "resnet50", "experiment": "isolated_module_latency",
                    "model_variant": config["model_variant"], "status": "running", "phase": "module_measurement",
                    "runner_pid": os.getpid(), "jobs": [], "skipped": [], "planned_case_count": 4,
                    "support_gate": {"status": "per_node_validation"},
                    "deadline_kst": os.environ["RESNET_DEADLINE_KST"],
                    "max_runtime_s": int(os.environ["RESNET_MAX_RUNTIME_S"]),
                    "measurement_scope": "isolated node forward + NPU synchronization; H2D/D2H excluded",
                    "compile_scope": "per-node compilation; no cross-module fusion; not full-model latency",
                    "cache_scope": "empty at case start, shared within one mode's node sequence; reuse in a new process",
                    "worker_source_snapshot": str(source)}

        def save():
            write_json(root / "manifest.json", manifest)
            write_progress(root, manifest)

        save()
        for mode in ("eager", "compile"):
            cache = root / "caches" / mode
            cold = root / "jobs" / f"{mode}-b1-cold"
            for reuse in (False, True):
                name = f"{mode}-b1-{'reuse' if reuse else 'cold'}"
                intent = "disk_reuse_attempt" if reuse else "empty_project_cache"
                if reuse and not (cold / "modules.csv").exists():
                    manifest["skipped"].append({"name": name, "reason": "Cold case produced no module measurements"})
                    save()
                    continue
                output = root / "jobs" / name
                command = [sys.executable, "-u", "-m", "Resnet.Phase_1.module_worker", "--config", str(root / "config.json"),
                           "--mode", mode, "--cache-intent", intent, "--cache", str(cache), "--output", str(output)]
                if reuse:
                    command.extend(["--cold-result", str(cold / "modules.csv")])
                entry = {"name": name, "command": command, "status": "running", "profile": False}
                manifest["jobs"].append(entry)
                save()
                started = time.monotonic()

                def heartbeat(pid):
                    entry.update(worker_pid=pid, elapsed_s=time.monotonic() - started)
                    progress = output / "progress.json"
                    if progress.exists():
                        data = json.loads(progress.read_text())
                        entry.update({key: data.get(key) for key in
                                      ("current_node", "current_module", "completed_nodes", "total_nodes", "failed_nodes")})
                    manifest["tables"] = aggregate_modules(root)
                    save()

                environment = os.environ.copy()
                environment.update(FURIOSA_TORCH_LOG="error", FURIOSA_TORCH_PROFILE="error", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                environment.pop("TORCH_LOGS", None)
                print(f"{name}: start", flush=True)
                code, timed_out = launch(command, root / "jobs" / f"{name}.log", config["timeout_s"], environment, heartbeat, source)
                path = output / "result.json"
                if not path.exists():
                    result = json.loads((output / "progress.json").read_text()) if (output / "progress.json").exists() else {}
                    result.update(status="timeout" if timed_out else "process_failed", error=f"Worker exit {code}")
                    write_json(path, result)
                result = json.loads(path.read_text())
                entry.update(status=result["status"], exit_code=code, timed_out=timed_out, elapsed_s=time.monotonic() - started,
                             completed_nodes=result.get("completed_nodes"), total_nodes=result.get("total_nodes"),
                             failed_nodes=result.get("failed_nodes"))
                manifest["tables"] = aggregate_modules(root)
                save()
                print(f"{name}: {entry['status']} ({entry['elapsed_s']:.1f}s)", flush=True)
        manifest.update(status="complete" if len(manifest["jobs"]) == 4 and all(j["status"] == "ok" for j in manifest["jobs"]) else "partial",
                        phase="finished", finished_kst=datetime.now(ZoneInfo("Asia/Seoul")).isoformat())
        save()
        return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
