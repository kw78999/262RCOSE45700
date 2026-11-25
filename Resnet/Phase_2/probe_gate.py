"""Fallback guard for shared-host benchmarking (the worker's exclusive section is the primary one).

Every `--interval` s:
  * exactly one job in a timed stage (warmup, repeat, aux_transfer): SIGSTOP every other Resnet
    runner / worker / probe process group; this also covers a worker without the exclusive lock;
  * no job in a timed stage: SIGCONT all of them, so a crashed worker never leaves others stopped;
  * two or more jobs timing at once (should not happen): touch nothing, log it.
Transitions are logged with timestamps.
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Resnet.Phase_2.exclusive import benchmark_groups, timed_jobs   # noqa: E402


def worker_pgid(job_dir):
    out = subprocess.run(["ps", "-eo", "pgid=,args="], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "-m Resnet.Phase_2.worker" in line and f"--output {job_dir}" in line:
            return int(line.split()[0])
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.2)
    args = parser.parse_args()
    state = None
    with args.log.open("a") as log:
        while True:
            jobs = timed_jobs(args.results.resolve())
            groups = benchmark_groups()
            if len(jobs) == 1:
                keep = worker_pgid(jobs[0])
                for pgid in groups:
                    if pgid != keep:
                        try:
                            os.killpg(pgid, signal.SIGSTOP)
                        except ProcessLookupError:
                            pass
                now = f"STOP all but {jobs[0].parent.parent.name}/{jobs[0].name} (pgid {keep})"
            elif not jobs:
                for pgid in groups:
                    try:
                        os.killpg(pgid, signal.SIGCONT)
                    except ProcessLookupError:
                        pass
                now = "CONT"
            else:
                now = "CONFLICT " + " ".join(j.name for j in jobs)
            if now != state:
                log.write(f"{datetime.now().isoformat(timespec='seconds')} {now} groups={sorted(groups)}\n")
                log.flush()
            state = now
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
