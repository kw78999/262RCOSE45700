"""Serial Phase 2 runner: one fresh process and one new compiler cache per trial.

Order is trial-major so that every finished trial is a complete batch sweep.
A (mode, batch) that failed or timed out once is not repeated in later trials;
its log and traceback are kept and the sweep continues with the next condition.
"""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from Resnet.artifacts import write_json
from Resnet.Phase_1.run import launch   # process-group timeout handling, reused unchanged

PROJECT = Path(__file__).resolve().parents[1]
KST = ZoneInfo("Asia/Seoul")
BAD = ("failed", "timeout", "process_failed")


def job_name(mode, batch, trial):
    return f"{mode}-b{batch}-t{trial}"


def build_plan(config):
    """Every planned (phase, mode, batch, trial), in execution order."""
    plan = []

    def add(phase, batch, trial, position):
        order = ("eager", "compile") if (position + trial) % 2 else ("compile", "eager")
        plan.extend({"phase": phase, "mode": mode, "batch": batch, "trial": trial,
                     "name": job_name(mode, batch, trial)} for mode in order
                    if mode in config.get("modes", ("eager", "compile")))

    for trial in range(1, config["core_trials"] + 1):
        for position, batch in enumerate(config["core_batches"]):
            add("core", batch, trial, position)
    for trial in range(config["core_trials"] + 1, config["revalidation_trials"] + 1):
        add("revalidation_extra", config["revalidation_batch"], trial, 0)
    for position, batch in enumerate(config["extension_batches"]):
        for trial in range(1, config["extension_trials"] + 1):
            add("extension", batch, trial, position)
    return plan


def skip_reason(job, history, config):
    """history maps (mode, batch) -> list of finished statuses."""
    mode, batch = job["mode"], job["batch"]
    if any(status in BAD for status in history.get((mode, batch), [])):
        return "same mode and batch already failed or timed out in an earlier trial"
    if job["phase"] == "extension":
        smaller = [b for b in config["core_batches"][-1:] + config["extension_batches"] if b < batch]
        for other in smaller:
            statuses = history.get((mode, other), [])
            if not statuses or any(status in BAD for status in statuses):
                return f"extension stops for this mode: batch {other} did not succeed"
    return None


def validate_config(config):
    for name in ("warmup", "repeats", "aux_warmup", "aux_repeats", "core_trials",
                 "revalidation_trials", "extension_trials", "timeout_s", "max_total_s", "cpu_threads"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    batches = config["core_batches"] + config["extension_batches"]
    if any(type(b) is not int or b < 1 for b in batches) or batches != sorted(set(batches)):
        raise ValueError("Batches must be unique positive integers in increasing order")
    if config["revalidation_batch"] not in config["core_batches"]:
        raise ValueError("The revalidation batch must be part of the core sweep")
    if config["revalidation_trials"] < config["core_trials"]:
        raise ValueError("Revalidation extends the core trials")
    modes = config.get("modes", ["eager", "compile"])
    if not modes or not set(modes) <= {"eager", "compile"} or len(set(modes)) != len(modes):
        raise ValueError("modes must be a non-empty subset of eager / compile")
    partition = config.get("eager_batch_partition", {})
    if any(type(e.get("chunk")) is not int or e["chunk"] < 1 or not e.get("modules") for e in partition.values()):
        raise ValueError("eager_batch_partition entries need a positive chunk and modules")
    if config.get("num_fusion", 1) not in (1, 2, 4):
        raise ValueError("num_fusion must be 1, 2 or 4 (single RNGD chip, hardware fusion)")
    if not config_pes(config) <= set(range(8)):
        raise ValueError("Configured device and fusion exceed the 8 PEs of one RNGD chip")
    if (config["model_variant"], config["precision"]) != ("bn_folded_output_split128", "bfloat16"):
        raise ValueError("Phase 2 measures the Phase 1 validated BF16 variant only")


def device_snapshot():
    from Resnet.environment import command_output
    return {name: command_output(["furiosa-smi", name]).get("stdout") for name in ("info", "ps")}


def toolchain_preflight():
    """The SDK cross-compiles generated C for the NPU; without libc headers every new compile fails."""
    import subprocess
    try:
        check = subprocess.run(["aarch64-linux-gnu-gcc", "-E", "-", "-o", os.devnull], input="#include <assert.h>\n",
                               capture_output=True, text=True, timeout=30)
        return {"ok": check.returncode == 0, "command": "aarch64-linux-gnu-gcc -E (assert.h)",
                "stderr": check.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"ok": False, "command": "aarch64-linux-gnu-gcc -E (assert.h)", "stderr": str(error)}


def config_pes(config):
    """Physical PEs of the configured logical device: furiosa:k with fusion n -> PEs k*n .. k*n+n-1."""
    n = config.get("num_fusion", 1)
    k = int(config["device"].split(":")[1])
    return set(range(k * n, k * n + n))


def foreign_npu_processes(text, pes=None):
    """PIDs listed by ``furiosa-smi ps``; with ``pes``, only those holding one of these PEs.

    A row whose core list cannot be parsed counts as overlapping (never measure blind)."""
    import re
    pids = []
    for line in (text or "").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not (cells and cells[0].isdigit()):
            continue
        if pes is not None and len(cells) > 1:
            match = re.search(r"\[([\d,\s]*)\]", cells[1])
            single = re.search(r":\s*(\d+)\s*$", cells[1])   # one PE prints as "npu0:2", no brackets
            numbers = ([int(x) for x in match.group(1).split(",") if x.strip()] if match
                       else [int(single.group(1))] if single else [])
            # furiosa-smi prints an inclusive range: "[0, 3]" = PEs 0..3, "[0, 1]" = PEs 0..1
            held = (set(range(numbers[0], numbers[1] + 1)) if len(numbers) == 2
                    else set(numbers) if numbers else None)
            if held is not None and not held & pes:
                continue
        pids.append(int(cells[0]))
    return pids


def write_progress(root, manifest):
    done = [j for j in manifest["jobs"] if j["status"] != "running"]
    lines = ["# Phase 2 진행 상태", "", f"갱신: {datetime.now(KST).isoformat()}", "",
             f"- 상태: `{manifest['status']}`", f"- 시작: {manifest['started_kst']}",
             f"- runner PID: {manifest.get('runner_pid')}",
             f"- PE fusion: {manifest.get('num_fusion', 1)}, 장치 {manifest.get('device', 'furiosa:0')}"
             f" (물리 PE {manifest.get('physical_pes', [0])})"
             + (f" / 재개 지점 `{manifest['continues']['start_from']}`" if manifest.get("continues") else ""),
             f"- 계획 {len(manifest['plan'])}개 / 종료 {len(done)}개 "
             f"(성공 {sum(j['status'] == 'ok' for j in done)}, 실패·timeout {sum(j['status'] in BAD for j in done)}) "
             f"/ 건너뜀 {len(manifest['skipped'])}개",
             f"- worker 누적 시간: {manifest['worker_elapsed_s']:.0f} s / 예산 {manifest['max_total_s']} s", "",
             "| 실행 | 단계 | 상태 | 경과 s | 첫 호출 s |", "| --- | --- | --- | ---: | ---: |"]
    for job in manifest["jobs"]:
        first = job.get("first_call_ms")
        lines.append(f"| {job['name']} | {job['phase']} | {job['status']} | {job.get('elapsed_s', 0):.1f} | "
                     f"{'' if first is None else format(first / 1000, '.1f')} |")
    for item in manifest["skipped"]:
        lines.append(f"| {item['name']} | {item['phase']} | skipped: {item['reason']} | | |")
    (root / "PROGRESS.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Continue an existing run directory")
    parser.add_argument("--max-total-s", type=int, help="Override the worker-time budget on resume")
    parser.add_argument("--start-from", help="New run only: begin at this planned job; earlier jobs are skipped")
    parser.add_argument("--previous-run", type=Path, help="Run whose earlier jobs --start-from continues")
    args = parser.parse_args()
    root = args.run_dir.resolve()
    # One runner per PE set: parallel runners on disjoint PEs are allowed when their timed stages
    # are serialized by the worker (config "exclusive_measurement").
    lock_config = json.loads(((root / "config.json") if args.resume else args.config).read_text())
    lock_name = ".rngd-benchmark.lock"
    if lock_config.get("exclusive_measurement"):
        pes = sorted(config_pes(lock_config))
        lock_name = f".rngd-benchmark-pe{pes[0]}-{pes[-1]}.lock"
    with (PROJECT / lock_name).open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another Resnet benchmark is already using the project lock")
        from Resnet.environment import collect
        from Resnet.Phase_2.analyze import aggregate
        toolchain = toolchain_preflight()
        if not toolchain["ok"]:
            parser.error("Cross-compiler preflight failed (aarch64 libc headers, package libc6-dev-arm64-cross): "
                         + toolchain["stderr"])
        if args.resume:
            config = json.loads((root / "config.json").read_text())
            manifest = json.loads((root / "manifest.json").read_text())
            manifest["resumes"] = manifest.get("resumes", []) + [datetime.now(KST).isoformat()]
        else:
            config = json.loads(args.config.read_text())
            validate_config(config)
            root.mkdir(parents=True, exist_ok=False)
            (root / "jobs").mkdir()
            write_json(root / "config.json", config)
            source = root / "source" / "Resnet"
            shutil.copytree(PROJECT, source, ignore=shutil.ignore_patterns(
                "results", "evidence", "__pycache__", ".git", ".cache", ".tmp", ".*lock", "*.pyc"))
            write_json(root / "source_hashes.json", {
                str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(source.rglob("*.py"))})
            collect(root / "evidence", include_installed_sources=False)
            manifest = {"experiment": config["experiment"], "started_kst": datetime.now(KST).isoformat(),
                        "status": "running", "plan": build_plan(config), "jobs": [], "skipped": [],
                        "worker_elapsed_s": 0.0, "max_total_s": config["max_total_s"],
                        "num_fusion": config.get("num_fusion", 1), "device": config["device"],
                        "physical_pes": sorted(config_pes(config)),
                        "new_process_per_job": True, "new_cache_directory_per_job": True,
                        "serial_device_execution": True,
                        "cache_reset_scope": "new project directory per job; system and device state untouched"}
        if args.start_from and not args.resume:
            names = [job["name"] for job in manifest["plan"]]
            if args.start_from not in names:
                parser.error(f"--start-from {args.start_from} is not a planned job")
            previous = str(args.previous_run.resolve()) if args.previous_run else None
            manifest["continues"] = {"previous_run": previous, "start_from": args.start_from,
                                     "previous_num_fusion": 1 if previous else None}
            reason = (f"before start point {args.start_from}; ran in {previous} with a different setup"
                      if previous else f"before start point {args.start_from}")
            manifest["skipped"] += [{**job, "reason": reason}
                                    for job in manifest["plan"][:names.index(args.start_from)]]
        if args.max_total_s:
            manifest["max_total_s"] = args.max_total_s
        manifest.update(status="running", runner_pid=os.getpid(), toolchain_preflight=toolchain)
        source_root = root / "source"

        def save():
            write_json(root / "manifest.json", manifest)
            write_progress(root, manifest)

        # An interrupted job leaves a directory without result.json; never reuse it or its cache.
        for entry in list(manifest["jobs"]):
            if entry["status"] == "running":
                stamp = datetime.now(KST).strftime("%H%M%S")
                for path in (root / "jobs" / entry["name"], root / "caches" / entry["name"],
                             root / "jobs" / f"{entry['name']}.log"):
                    if path.exists():
                        path.rename(path.with_name(f"{path.name}.interrupted-{stamp}"))
                manifest["jobs"].remove(entry)
                manifest.setdefault("interrupted", []).append({**entry, "status": "interrupted"})
        history = {}
        for entry in manifest["jobs"]:
            history.setdefault((entry["mode"], entry["batch"]), []).append(entry["status"])
        finished = {entry["name"] for entry in manifest["jobs"]}
        skipped = {item["name"] for item in manifest["skipped"]}
        save()

        for job in manifest["plan"]:
            name = job["name"]
            if name in finished or name in skipped:
                continue
            remaining = manifest["max_total_s"] - manifest["worker_elapsed_s"]
            reason = skip_reason(job, history, config)
            if reason is None and remaining < 60:
                reason = "worker-time budget exhausted before this job started"
            if reason:
                manifest["skipped"].append({**job, "reason": reason})
                save()
                continue
            waited = 0
            pes = config_pes(config)
            while foreign_npu_processes(device_snapshot()["ps"], pes) and waited < 1800:
                time.sleep(30)   # never measure next to another NPU user; never stop their process
                waited += 30
            before = device_snapshot()
            if foreign_npu_processes(before["ps"], pes):
                manifest["skipped"].append({**job, "reason": f"PEs {sorted(pes)} held by another process for 30 minutes"})
                save()
                continue
            output, cache, scratch = root / "jobs" / name, root / "caches" / name, root / "tmp" / name
            scratch.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-u", "-m", "Resnet.Phase_2.worker", "--config", str(root / "config.json"),
                       "--mode", job["mode"], "--batch", str(job["batch"]), "--trial", str(job["trial"]),
                       "--cache", str(cache), "--output", str(output)]
            environment = os.environ.copy()
            overrides = dict(FURIOSA_TORCH_LOG="error", FURIOSA_TORCH_PROFILE="error",
                             OMP_NUM_THREADS=str(config["cpu_threads"]), MKL_NUM_THREADS=str(config["cpu_threads"]),
                             XDG_CACHE_HOME=str(PROJECT / ".cache"), TMPDIR=str(scratch))
            environment.update(overrides)
            environment.pop("TORCH_LOGS", None)
            timeout = int(min(config["timeout_s"], remaining))
            entry = {**job, "status": "running", "command": command, "environment_overrides": overrides,
                     "timeout_s": timeout, "waited_for_device_s": waited, "device_before": before,
                     "started_kst": datetime.now(KST).isoformat()}
            manifest["jobs"].append(entry)
            save()
            print(f"{name}: start", flush=True)
            started = time.perf_counter()

            def heartbeat(pid):
                entry.update(worker_pid=pid, elapsed_s=time.perf_counter() - started)
                save()

            exit_code, timed_out = launch(command, root / "jobs" / f"{name}.log", timeout,
                                          environment, heartbeat, source_root)
            result_path = output / "result.json"
            if not result_path.exists():
                progress = output / "progress.json"
                result = json.loads(progress.read_text()) if progress.exists() else {
                    "job": name, "mode": job["mode"], "batch": job["batch"], "trial": job["trial"]}
                result.update(status="timeout" if timed_out else "process_failed",
                              error_type="TimeoutExpired" if timed_out else "ProcessExit",
                              error=f"Worker exit {exit_code}"
                                    + (f" after the {timeout} s limit" if timed_out else "")
                                    + "; see the preserved log")
                output.mkdir(parents=True, exist_ok=True)
                write_json(result_path, result)
            result = json.loads(result_path.read_text())
            elapsed = time.perf_counter() - started
            manifest["worker_elapsed_s"] += elapsed
            entry.update(status=result["status"], exit_code=exit_code, timed_out=timed_out, elapsed_s=elapsed,
                         first_call_ms=result.get("first_call_ms"), stage=result.get("stage"),
                         device_after=device_snapshot())
            history.setdefault((job["mode"], job["batch"]), []).append(result["status"])
            shutil.rmtree(scratch, ignore_errors=True)
            manifest["tables"] = aggregate(root)
            save()
            print(f"{name}: {entry['status']} ({elapsed:.1f}s)", flush=True)

        manifest["status"] = "finished"
        manifest["finished_kst"] = datetime.now(KST).isoformat()
        manifest["tables"] = aggregate(root)
        save()
        print(json.dumps({"run_dir": str(root), "status": manifest["status"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
