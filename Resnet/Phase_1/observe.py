"""Separate detailed Shape-switching logs; instrumented timings are not benchmarks."""
import argparse
import fcntl
import json
import os
import sys
from pathlib import Path

from Resnet.artifacts import write_json
from Resnet.Phase_1.run import PROJECT, launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Completed run with successful mixed jobs and existing caches")
    args = parser.parse_args()
    root = args.run_dir.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    config = json.loads((root / "config.json").read_text())
    if manifest["status"] != "complete":
        parser.error("Observation requires a successfully completed run")
    with (PROJECT / ".rngd-benchmark.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        destination = root / "observability"
        destination.mkdir(exist_ok=False)
        observations = []
        for mode in ("eager", "compile"):
            cache = root / "caches" / f"{mode}-mixed"
            output = destination / mode
            command = [sys.executable, "-u", "-m", "Resnet.Phase_1.worker", "--config", str(root / "config.json"),
                       "--mode", mode, "--target", manifest["target"], "--workload", "mixed",
                       "--cache-intent", "disk_reuse_attempt", "--cache", str(cache), "--output", str(output)]
            environment = os.environ.copy()
            environment.update(FURIOSA_TORCH_LOG="debug", FURIOSA_TORCH_PROFILE="error",
                               TORCH_LOGS="graph_breaks,recompiles",
                               OMP_NUM_THREADS=str(config["cpu_threads"]), MKL_NUM_THREADS=str(config["cpu_threads"]))
            # Worker Python SDK logs use WARNING; Rust SDK debug and Torch guard logs
            # are enabled only here. All these samples live outside jobs/.
            code, timed_out = launch(command, destination / f"{mode}.log", config["timeout_s"], environment)
            lines = (destination / f"{mode}.log").read_text().splitlines()
            excerpts = [line for line in lines if any(pattern in line for pattern in
                        ("Recompiling function", "triggered by the following guard", "size mismatch",
                         "Graph break", "fallback to CPU", "graph compiled successfully"))]
            observations.append({"mode": mode, "command": command, "exit_code": code, "timed_out": timed_out,
                                 "log_line_count": len(lines), "selected_log_lines": excerpts,
                                 "interpretation": "Log entries and backend counters observe recompilation paths, not verified disk hits or codegen counts"})
            write_json(destination / "manifest.json", {"instrumented_timings_excluded": True, "observations": observations})
            print(f"{mode}: exit={code}, timeout={timed_out}, log_lines={len(lines)}", flush=True)
        return 0 if all(item["exit_code"] == 0 for item in observations) else 1


if __name__ == "__main__":
    sys.exit(main())
