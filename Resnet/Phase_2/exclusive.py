"""Exclusive timed stages for benchmark jobs that share one host.

Several runners may run on disjoint PEs at once, but the timed loops of a job (warmup, repeat, aux)
must see the same quiet host as a serial run. `enter()` therefore
  1. takes a host-wide file lock (one timed section at a time; waiting is never timed),
  2. waits until no other job, including one that does not use this lock, is in a timed stage,
  3. SIGSTOPs every other Resnet benchmark process group (runners, workers, compile probes),
and `leave()` sends SIGCONT. Each pause interval is appended to a JSONL log, so first-call
(compile) times of paused jobs can be corrected afterwards. The lock is released by `release()`
after the job wrote result.json, so a probe gate never sees two jobs in a timed stage.
"""
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path

MEASURED = {"warmup", "repeat", "aux_transfer"}
MARKERS = ("-m Resnet.Phase_2.run", "-m Resnet.Phase_2.worker", "-m Resnet.Phase_2.eager_op_probe")
_state = {}


def benchmark_groups():
    """{pgid: command} of every Resnet runner / worker / probe except this process group."""
    own = os.getpgid(0)
    out = subprocess.run(["ps", "-eo", "pgid=,args="], capture_output=True, text=True).stdout
    groups = {}
    for line in out.splitlines():
        pgid, _, args = line.strip().partition(" ")
        if "python" in args and any(m in args for m in MARKERS) and int(pgid) != own:
            groups.setdefault(int(pgid), args.strip()[:200])
    return groups


def timed_jobs(results, prefix="matrix-", exclude=None):
    """Job dirs whose worker is in a timed stage (running, no result.json yet)."""
    found = []
    for progress in Path(results).glob(f"{prefix}*/jobs/*/progress.json"):
        job = progress.parent
        if job == exclude or (job / "result.json").exists():
            continue
        try:
            if json.loads(progress.read_text()).get("stage") in MEASURED:
                found.append(job)
        except (OSError, ValueError):
            pass
    return found


def _signal(groups, sig):
    for pgid in groups:
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass


def enter(settings, job_dir, stage):
    """Block until this job may time; returns a dict of what was waited for and paused."""
    started = time.time()
    lock = open(settings["lock"], "a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    locked = time.time()
    while timed_jobs(settings["results"], exclude=Path(job_dir)) and time.time() - locked < 3600:
        time.sleep(0.2)
    quiet = time.time()
    stage("warmup")
    groups = benchmark_groups()
    _signal(groups, signal.SIGSTOP)
    _state.update(lock=lock, settings=settings, job=str(job_dir), groups=groups, t0=time.time())
    return {"lock_wait_s": locked - started, "other_timed_job_wait_s": quiet - locked,
            "paused_groups": {str(k): v for k, v in groups.items()}}


def leave():
    if "t0" not in _state:
        return None
    groups = {**_state["groups"], **benchmark_groups()}   # also any group that appeared meanwhile
    t1 = time.time()
    _signal(groups, signal.SIGCONT)
    entry = {"t0": _state.pop("t0"), "t1": t1, "by": _state["job"],
             "pgids": sorted(_state["groups"]), "commands": {str(k): v for k, v in _state["groups"].items()}}
    with open(_state["settings"]["pause_log"], "a") as log:
        log.write(json.dumps(entry) + "\n")
    return entry["t1"] - entry["t0"]


def release():
    """After result.json: continue everything (safety) and let the next job time."""
    leave()
    _signal(benchmark_groups(), signal.SIGCONT)
    lock = _state.pop("lock", None)
    if lock is not None:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
