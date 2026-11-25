"""Container memory log + guard for the serial finish of the fusion matrix (2026-11-25).

Every 15 s: append container memory (cgroup memory.current) and the largest Resnet benchmark
process group (summed RSS) to mem.csv, and keep each group's peak in mem_peaks.json.
If memory.current exceeds --kill-gib, SIGKILL the largest benchmark group and log it, so that one
runaway compile cannot take the whole container down (the suspected cause of two restarts).
"""
import argparse
import csv
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))
MARKERS = ("-m Resnet.Phase_2.worker", "-m Resnet.Phase_2.eager_op_probe", "-m Resnet.Phase_2.run")


def groups():
    out = subprocess.run(["ps", "-eo", "pgid=,rss=,args="], capture_output=True, text=True).stdout
    found = {}
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pgid, rss, args = int(parts[0]), int(parts[1]), parts[2]
        entry = found.setdefault(pgid, {"rss_kib": 0, "cmd": ""})
        entry["rss_kib"] += rss
        if any(m in args for m in MARKERS) and not entry["cmd"]:
            entry["cmd"] = args[:160]
    return {k: v for k, v in found.items() if v["cmd"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--kill-gib", type=float, default=150.0)
    args = parser.parse_args()
    log, peaks_path = args.dir / "mem.csv", args.dir / "mem_peaks.json"
    peaks = json.loads(peaks_path.read_text()) if peaks_path.exists() else {}
    new = not log.exists()
    with log.open("a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["time_kst", "container_gib", "top_group_gib", "top_cmd", "action"])
        while True:
            current = int(Path("/sys/fs/cgroup/memory.current").read_text()) / 2**30
            found = groups()
            top = max(found.items(), key=lambda kv: kv[1]["rss_kib"], default=None)
            for pgid, entry in found.items():
                key = f"{pgid} {entry['cmd']}"
                peaks[key] = max(peaks.get(key, 0), entry["rss_kib"] / 2**20)
            action = ""
            if current > args.kill_gib and top:
                os.killpg(top[0], signal.SIGKILL)
                action = f"SIGKILL pgid {top[0]} at {current:.1f} GiB"
            writer.writerow([datetime.now(KST).isoformat(timespec="seconds"), f"{current:.2f}",
                             f"{top[1]['rss_kib'] / 2**20:.2f}" if top else "", top[1]["cmd"] if top else "", action])
            f.flush()
            peaks_path.write_text(json.dumps(peaks, indent=1, ensure_ascii=False) + "\n")
            time.sleep(15)


if __name__ == "__main__":
    main()
