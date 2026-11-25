"""Serial, isolated experiment runner with an explicit model-support gate."""
import argparse
import fcntl
import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from Resnet.artifacts import write_json
from Resnet.Phase_1.progress import write_progress

PROJECT = Path(__file__).resolve().parents[1]


def validate_config(config):
    if type(config.get("bn_folded", False)) is not bool:
        raise ValueError("bn_folded must be boolean")
    split = config.get("output_split_channels")
    if split is not None and (type(split) is not int or split != 128 or not config.get("bn_folded")):
        raise ValueError("output_split_channels=128 requires bn_folded=true")
    if (config["model"], config["weights"], config["precision"]) != ("resnet50", "IMAGENET1K_V2", "bfloat16"):
        raise ValueError("This phase fixes ResNet50 IMAGENET1K_V2 BF16")
    if not config["device"].startswith("furiosa:"):
        raise ValueError("An explicit Furiosa device is required")
    for name in ("image_size", "iterations", "timeout_s", "cpu_threads"):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if "mixed_timeout_s" in config and (type(config["mixed_timeout_s"]) is not int or config["mixed_timeout_s"] < 1):
        raise ValueError("mixed_timeout_s must be a positive integer")
    if type(config["warmup"]) is not int or config["warmup"] < 0:
        raise ValueError("warmup must be a nonnegative integer")
    if not config["batches"] or any(type(x) is not int or x < 1 for x in config["batches"]):
        raise ValueError("batches must contain positive integers")
    if len(set(config["batches"])) != len(config["batches"]):
        raise ValueError("Duplicate batch conditions")
    if not config["mixed_batches"] or any(x not in config["batches"] for x in config["mixed_batches"]):
        raise ValueError("Mixed batches must come from the fixed batch grid")


def launch(command, log_path, timeout, environment, on_progress=None, working_directory=None):
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=working_directory or PROJECT.parent, stdout=log, stderr=subprocess.STDOUT,
                                   env=environment, start_new_session=True)
        deadline = time.monotonic() + timeout
        if on_progress:
            on_progress(process.pid)
        while True:
            try:
                return process.wait(timeout=max(0.01, min(30, deadline - time.monotonic()))), False
            except subprocess.TimeoutExpired:
                if time.monotonic() < deadline:
                    if on_progress:
                        on_progress(process.pid)
                    continue
                break
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        return process.returncode, True


def cold_job_succeeded(root, mode, condition):
    path = root / "jobs" / f"{mode}-{condition}-cold" / "result.json"
    if not path.exists():
        return False
    result = json.loads(path.read_text())
    return result.get("status") == "ok"


def verify_minimum_condition(root, batch, validation_path=None):
    import torch
    from Resnet.analysis import compare_tensors
    jobs = {mode: json.loads((root / "jobs" / f"{mode}-b{batch}-cold" / "result.json").read_text())
            for mode in ("eager", "compile")}
    if jobs["eager"]["model_transform"] != jobs["compile"]["model_transform"]:
        raise ValueError("Minimum condition has mismatched transformed weights")
    if jobs["eager"]["config"] != jobs["compile"]["config"]:
        raise ValueError("Minimum condition has mismatched configurations")
    outputs = {mode: torch.load(root / "jobs" / f"{mode}-b{batch}-cold" / f"output-b{batch}.pt", weights_only=True)
               for mode in ("eager", "compile")}
    comparison = compare_tensors(outputs["eager"], outputs["compile"])
    if comparison["status"] != "measured" or list(outputs["eager"].shape) != [batch, 1000]:
        raise ValueError("Minimum condition outputs must have matching finite classifier tensors")
    result = {"status": "passed", "basis": "Both full-model executions completed; same model state and configuration; finite expected output shape",
              "eager_vs_compile": comparison, "numerical_acceptance": "No dataset accuracy threshold defined"}
    if validation_path:
        validation = json.loads(validation_path.read_text())
        if validation["status"] != "ok" or validation["model_transform"] != jobs["eager"]["model_transform"]:
            raise ValueError("CPU transformation validation does not match the measured model")
        reference = validation_path.parent / f"cpu-output-b{batch}.pt"
        if reference.exists():
            cpu_output = torch.load(reference, weights_only=True)
            result["cpu_reference"] = {mode: compare_tensors(cpu_output, outputs[mode]) for mode in outputs}
    write_json(root / "minimum_condition.json", result)
    return result


def main():
    from Resnet.Phase_1.aggregate import aggregate
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--run-dir", type=Path, required=True, help="New directory; existing runs are never overwritten")
    parser.add_argument("--target", choices=("resnet50", "conv1", "bn1"), default="resnet50")
    parser.add_argument("--profiles", action="store_true", help="Separate instrumented processes; excluded from performance tables")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    if config.get("output_split_channels") and args.target != "resnet50":
        parser.error("The split128 configuration is a full ResNet50 experiment")
    validation_path = None
    if config.get("validation_file"):
        validation_path = (args.config.resolve().parent / config["validation_file"]).resolve()
        if not validation_path.exists() or json.loads(validation_path.read_text()).get("status") != "ok":
            parser.error("A successful CPU transformation validation is required")
    # The lock prevents two instances of this project from benchmarking together.
    with (PROJECT / ".rngd-benchmark.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another Resnet benchmark is already using the project lock")
        root = args.run_dir.resolve()
        root.mkdir(parents=True, exist_ok=False)
        (root / "jobs").mkdir()
        write_json(root / "config.json", config)
        source_root = root / "source"
        source_project = source_root / "Resnet"
        shutil.copytree(PROJECT, source_project,
                        ignore=shutil.ignore_patterns("results", "evidence", "__pycache__", ".git", ".cache", ".tmp", ".*lock", "*.pyc"))
        source_hashes = {str(path.relative_to(source_project)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in source_project.rglob("*.py")}
        write_json(root / "source_hashes.json", source_hashes)
        from Resnet.environment import collect, command_output
        collect(root / "evidence", include_installed_sources=not config.get("workspace_only", False))
        manifest = {"started_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
                    "target": args.target, "status": "running", "jobs": [], "skipped": [],
                    "model_variant": config.get("model_variant", "conv_bn_eval_folded" if config.get("bn_folded") else "identity"),
                    "runner_pid": os.getpid(), "phase": "minimum_condition", "support_gate": {"status": "pending"},
                    "worker_source_snapshot": str(source_root),
                    "new_process_per_job": True, "serial_device_execution": True,
                    "cache_reset_scope": "New project directory only; shared/system caches untouched"}
        if os.environ.get("RESNET_DEADLINE_KST"):
            manifest.update(deadline_kst=os.environ["RESNET_DEADLINE_KST"],
                            max_runtime_s=int(os.environ["RESNET_MAX_RUNTIME_S"]))

        def save():
            write_json(root / "manifest.json", manifest)
            write_progress(root, manifest)

        if validation_path:
            write_json(root / "evidence" / "transform-validation.json", json.loads(validation_path.read_text()))
            for reference in validation_path.parent.glob("cpu-output-b*.pt"):
                shutil.copy2(reference, root / "evidence" / reference.name)
            validation_path = root / "evidence" / "transform-validation.json"
        save()

        def job(mode, workload, batch, reuse=False, profile=False):
            condition = f"b{batch}" if workload == "fixed" else "mixed"
            cache = root / "caches" / f"{mode}-{condition}"
            intent = "disk_reuse_attempt" if reuse else "empty_project_cache"
            name = f"{mode}-{condition}-{'reuse' if reuse else 'cold'}"
            if profile:
                name += "-profile"
            parent = root / ("profiles" if profile else "jobs")
            parent.mkdir(exist_ok=True)
            output = parent / name
            command = [sys.executable, "-u", "-m", "Resnet.Phase_1.worker", "--config", str(root / "config.json"),
                       "--mode", mode, "--target", args.target, "--workload", workload, "--batch", str(batch),
                       "--cache-intent", intent, "--cache", str(cache), "--output", str(output)]
            if profile:
                command.append("--profile")
            environment = os.environ.copy()
            environment.update(FURIOSA_TORCH_LOG="error", FURIOSA_TORCH_PROFILE="trace" if profile else "error",
                               OMP_NUM_THREADS=str(config["cpu_threads"]), MKL_NUM_THREADS=str(config["cpu_threads"]))
            environment.pop("TORCH_LOGS", None)
            entry = {"name": name, "command": command, "status": "running",
                     "profile": profile,
                     "working_directory": str(source_root),
                     "environment_overrides": {key: environment[key] for key in
                        ("FURIOSA_TORCH_LOG", "FURIOSA_TORCH_PROFILE", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}
            manifest["jobs"].append(entry)
            save()
            print(f"{name}: start", flush=True)
            started = time.perf_counter()
            def heartbeat(pid):
                entry.update(worker_pid=pid, elapsed_s=time.perf_counter() - started)
                save()
            timeout = config.get("mixed_timeout_s", config["timeout_s"]) if workload == "mixed" else config["timeout_s"]
            entry["timeout_s"] = timeout
            exit_code, timed_out = launch(command, parent / f"{name}.log", timeout, environment, heartbeat, source_root)
            result_path = output / "result.json"
            if not result_path.exists():
                progress = output / "progress.json"
                result = json.loads(progress.read_text()) if progress.exists() else {}
                result.update(mode=mode, target=args.target, workload=workload, batch=batch,
                              model_variant=manifest["model_variant"],
                              cache_intent=intent, status="timeout" if timed_out else "process_failed",
                              error_type="TimeoutExpired" if timed_out else "ProcessExit",
                              error=f"Worker exit {exit_code}; see preserved log")
                write_json(result_path, result)
            result = json.loads(result_path.read_text())
            entry.update(status=result["status"], exit_code=exit_code, timed_out=timed_out,
                         elapsed_s=time.perf_counter() - started)
            from Resnet.Phase_1.aggregate import aggregate
            manifest["tables"] = aggregate(root)
            save()
            print(f"{name}: {entry['status']} ({entry['elapsed_s']:.1f}s)", flush=True)
            return exit_code == 0 and result["status"] == "ok"

        # Gate on the smallest fixed condition in both modes before the matrix.
        ordered_batches = sorted(config["batches"])
        first_batch = ordered_batches[0]
        gate = [job(mode, "fixed", first_batch) for mode in ("eager", "compile")]
        if all(gate) and args.target == "resnet50":
            manifest["support_gate"] = verify_minimum_condition(root, first_batch, validation_path)
        else:
            manifest["support_gate"] = {"status": "passed" if all(gate) else "failed"}
        save()
        plan = []
        for index, batch in enumerate(ordered_batches):
            order = ("eager", "compile") if index % 2 == 0 else ("compile", "eager")
            if index > 0:
                plan.extend((mode, "fixed", batch, False) for mode in order)
            plan.extend((mode, "fixed", batch, True) for mode in reversed(order))
        plan.extend((mode, "mixed", first_batch, reuse) for reuse in (False, True) for mode in ("eager", "compile"))
        if all(gate):
            manifest["phase"] = "measurement_matrix"
            save()
            for mode, workload, batch, reuse in plan:
                cache = root / "caches" / f"{mode}-{'mixed' if workload == 'mixed' else f'b{batch}'}"
                condition = "mixed" if workload == "mixed" else f"b{batch}"
                if reuse and (not cold_job_succeeded(root, mode, condition) or not any(cache.rglob("output.bin"))):
                    manifest["skipped"].append({"mode": mode, "workload": workload, "batch": batch,
                                                "reuse": reuse, "reason": "A successful cold job and a nonempty compiler cache are required; output.bin alone can contain a cached error"})
                    save()
                    continue
                job(mode, workload, batch, reuse)
            if args.profiles:
                manifest["phase"] = "profiling"
                save()
                for mode in ("eager", "compile"):
                    job(mode, "fixed", first_batch, reuse=True, profile=True)
            manifest["status"] = "complete" if all(entry["status"] == "ok" for entry in manifest["jobs"]) and not manifest["skipped"] else "partial"
        else:
            manifest["status"] = "blocked_model_support"
            manifest["skipped"] = [{"mode": mode, "workload": workload, "batch": batch, "reuse": reuse,
                                     "reason": "Smallest original-model condition failed; no mode comparison available"}
                                    for mode, workload, batch, reuse in plan]
            if args.profiles:
                manifest["skipped"].extend({"mode": mode, "workload": "fixed", "batch": first_batch,
                                            "profile": True, "reason": "Model support gate failed"}
                                           for mode in ("eager", "compile"))
        from Resnet.Phase_1.aggregate import aggregate
        manifest["tables"] = aggregate(root)
        manifest["finished_kst"] = datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
        manifest["phase"] = "finished"
        write_json(root / "evidence" / "device-after.json", {
            "status": command_output(["furiosa-smi", "status"]),
            "processes": command_output(["furiosa-smi", "ps"])})
        save()
        print(json.dumps({"run_dir": str(root), "status": manifest["status"], "tables": manifest["tables"]}, ensure_ascii=False), flush=True)
        return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
