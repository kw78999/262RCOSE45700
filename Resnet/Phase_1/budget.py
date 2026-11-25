"""Detached wall-clock supervisor; workers use independent process groups."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parents[1]
# Load only this project package when invoked as a script from inside Resnet.
if "Resnet" not in sys.modules:
    spec = importlib.util.spec_from_file_location("Resnet", PROJECT / "__init__.py",
                                                submodule_search_locations=[str(PROJECT)])
    package = importlib.util.module_from_spec(spec)
    sys.modules["Resnet"] = package
    spec.loader.exec_module(package)
from Resnet.artifacts import write_json
from Resnet.Phase_1.progress import write_progress


def runner_command(config, root, profiles, module="Resnet.Phase_1.run"):
    if module not in ("Resnet.Phase_1.run", "Resnet.Phase_1.module_run"):
        raise ValueError("Unsupported experiment runner")
    bootstrap = ("import importlib.util,sys,runpy;from pathlib import Path;"
                 "p=Path.cwd();s=importlib.util.spec_from_file_location('Resnet',p/'__init__.py',"
                 "submodule_search_locations=[str(p)]);m=importlib.util.module_from_spec(s);"
                 "sys.modules['Resnet']=m;s.loader.exec_module(m);"
                 f"runpy.run_module({module!r},run_name='__main__')")
    return [sys.executable, "-u", "-c", bootstrap, "--config", str(config),
            "--run-dir", str(root)] + (["--profiles"] if profiles else [])


def descendants(pid):
    rows = subprocess.check_output(["ps", "-eo", "pid=,ppid=,pgid="], text=True, timeout=2)
    entries = [tuple(map(int, line.split())) for line in rows.splitlines()]
    found = {pid}
    while True:
        children = {child for child, parent, _ in entries if parent in found}
        expanded = found | children
        if expanded == found:
            break
        found = expanded
    return {group for child, _, group in entries if child in found}


def send_group(group, sig):
    try:
        os.killpg(group, sig)
    except ProcessLookupError:
        pass


def stop_tree(process, groups):
    # Freeze the runner before the final tree scan, preventing new jobs.
    send_group(process.pid, signal.SIGSTOP)
    groups.update(descendants(process.pid))
    groups.discard(os.getpgrp())
    for group in groups - {process.pid}:
        send_group(group, signal.SIGTERM)
    send_group(process.pid, signal.SIGKILL)
    process.wait(timeout=2)
    time.sleep(2)
    for group in groups - {process.pid}:
        send_group(group, signal.SIGKILL)


def mark_limit(root, reason):
    path = root / "manifest.json"
    if not path.exists():
        return
    manifest = json.loads(path.read_text())
    now = datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
    for job in manifest["jobs"]:
        if job["status"] != "running":
            continue
        job.update(status="time_budget_exceeded", stopped_kst=now, stop_reason=reason)
        directory = root / ("profiles" if job.get("profile") else "jobs") / job["name"]
        progress = directory / "progress.json"
        result_path = directory / "result.json"
        # Preserve a job that finished just as the watchdog stopped the runner.
        if result_path.exists():
            job["status"] = json.loads(result_path.read_text())["status"]
        else:
            result = json.loads(progress.read_text()) if progress.exists() else {}
            result.update(status="time_budget_exceeded", error_type="WallClockBudget",
                          error=reason, stopped_kst=now)
            write_json(result_path, result)
            write_json(progress, result)
    manifest.update(status="time_budget_exceeded", phase="finished", finished_kst=now,
                    stop_reason=reason, remaining_plan_not_executed=True)
    if manifest.get("experiment") == "isolated_module_latency":
        from Resnet.Phase_1.module_run import aggregate_modules
        manifest["tables"] = aggregate_modules(root)
    write_json(path, manifest)
    write_progress(root, manifest)
    # Successful jobs have already been aggregated after each completion.


def supervise(command, root, budget_s, environment):
    if budget_s < 15 or budget_s > 7200:
        raise ValueError("Wall-clock budget must be between 15 and 7200 seconds")
    started = time.monotonic()
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    deadline = now + timedelta(seconds=budget_s)
    environment = {**environment, "RESNET_DEADLINE_KST": deadline.isoformat(),
                   "RESNET_MAX_RUNTIME_S": str(budget_s)}
    record = {"status": "running", "started_kst": now.isoformat(),
              "deadline_kst": deadline.isoformat(), "max_runtime_s": budget_s,
              "supervisor_pid": os.getpid(), "command": command,
              "cleanup_reserve_s": 10, "run_dir": str(root)}
    evidence = root.parent / (root.name + "-budget.json")
    with evidence.with_suffix(".console.log").open("w") as log:
        process = subprocess.Popen(command, cwd=PROJECT, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record["runner_pid"] = process.pid
        write_json(evidence, record)
        groups = {process.pid}
        try:
            while process.poll() is None:
                if time.monotonic() - started >= budget_s - 10:
                    stop_tree(process, groups)
                    mark_limit(root, "전체 실험시간 제한; 2시간 이내 종료를 위해 정리 시간 10초 확보")
                    record["status"] = "time_budget_exceeded"
                    break
                groups = descendants(process.pid) | {process.pid}
                time.sleep(0.5)
            else:
                record["status"] = "finished"
        finally:
            if process.poll() is None:
                stop_tree(process, groups)
            else:
                # A runner can fail before reaping a worker in another session.
                for group in groups - {process.pid, os.getpgrp()}:
                    send_group(group, signal.SIGTERM)
                time.sleep(0.2)
                for group in groups - {process.pid, os.getpgrp()}:
                    send_group(group, signal.SIGKILL)
            record.update(exit_code=process.returncode, elapsed_s=time.monotonic() - started,
                          finished_kst=datetime.now(ZoneInfo("Asia/Seoul")).isoformat())
            write_json(evidence, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-runtime-s", type=int, default=7200)
    parser.add_argument("--profiles", action="store_true")
    parser.add_argument("--runner", choices=("Resnet.Phase_1.run", "Resnet.Phase_1.module_run"), default="Resnet.Phase_1.run")
    args = parser.parse_args()
    config, root = args.config.resolve(), args.run_dir.resolve()
    if not config.is_relative_to(PROJECT) or not root.is_relative_to(PROJECT):
        parser.error("Config and results must be inside Resnet")
    if root.exists():
        parser.error("A new run directory is required")
    root.parent.mkdir(parents=True, exist_ok=True)
    with (PROJECT / ".budgeted-run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = supervise(runner_command(config, root, args.profiles, args.runner), root,
                           args.max_runtime_s, os.environ.copy())
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
