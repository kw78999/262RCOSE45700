"""One-line progress / ETA for the 18-job PE-fusion matrix (fusion 1, 2, 4 x eager, compile x b1, 16, 64).

Remaining time = sum of per-job estimates for jobs not finished. Estimates start from earlier runs
(fusion 4 measured; fusion 1/2 guessed) and are rescaled by the ratio actual / estimate observed so
far in the same mode, so they tighten as jobs finish.
"""
import json
import sys
import time
from datetime import datetime, timedelta, timezone
KST = timezone(timedelta(hours=9))
from pathlib import Path

RESULTS = Path(__file__).with_name("results")
START = datetime(2026, 11, 25, 1, 53, tzinfo=timezone.utc)   # first probe / sweep launch
ORDER = (4, 1, 2)
BATCHES = (1, 16, 64)
# job wall seconds: fusion 4 from 2026-11-19/20 runs; fusion 1 b1 and b2/b4 compile from the 1 PE run
EST = {
    (4, "eager"): {1: 470, 16: 860, 64: 1290}, (4, "compile"): {1: 505, 16: 1040, 64: 1790},
    (1, "eager"): {1: 470, 16: 900, 64: 1400}, (1, "compile"): {1: 435, 16: 2000, 64: 3500},
    (2, "eager"): {1: 470, 16: 880, 64: 1350}, (2, "compile"): {1: 470, 16: 1400, 64: 2500},
}


MOVED = {(1, "compile", 64), (2, "compile", 64), (2, "eager", 1)}   # f1x on PE 3, f2x on PE 6-7


def RERUN_DONE():
    job = RESULTS / "matrix-f4-20261125" / "jobs" / "eager-b1-t1"
    return (job.parent / "eager-b1-t1.contended").exists() and (job / "result.json").exists()


def fmt(seconds):
    seconds = int(max(0, seconds))
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def main():
    done, running, ratio = {}, None, {"eager": [], "compile": []}
    running_all = []
    for fusion in ORDER:
      for run in sorted(RESULTS.glob(f"matrix-f{fusion}-*")) + sorted(RESULTS.glob(f"matrix-f{fusion}x-*")):
        if not (run / "manifest.json").exists():
            continue
        manifest = json.loads((run / "manifest.json").read_text())
        for job in manifest.get("jobs", []):
            if job.get("status") == "moved":
                continue
            key = (fusion, job["mode"], job["batch"])
            if job.get("status") in (None, "running"):
                stage = "?"
                progress = run / "jobs" / job["name"] / "progress.json"
                if progress.exists():
                    stage = json.loads(progress.read_text()).get("stage")
                started = datetime.fromisoformat(job["started_kst"]).timestamp()
                running = (key, stage, time.time() - started)
                running_all.append(running)
                continue
            elapsed = job.get("elapsed_s") or job.get("worker_elapsed_s") or 0
            done[key] = (job.get("status"), elapsed)
            if job.get("status") == "ok" and elapsed and job.get("started_kst", "") >= "2026-11-25T12:10":
                # observed slowdown of jobs that compiled next to other lanes
                ratio[job["mode"]].append(elapsed / EST[(fusion, job["mode"])][job["batch"]])
    scale = {m: (sum(r) / len(r) if r else 1.0) for m, r in ratio.items()}
    # Parallel lanes since 2026-11-25 12:10 KST: A = PE 4-7 (fusion 4, then the f4 eager-b1 re-run),
    # B = PE 2 (fusion 1), C = PE 0-1 (fusion 2). Parallel compiles run ~1.3x slower than alone.
    lanes = {"A": 0.0, "B": 0.0, "C": 0.0}   # C = fusion 2 on PE 0-1 (from ~12:30 KST)
    if not list(RESULTS.glob("matrix-f2-*/manifest.json")):
        lanes["C"] += 1200   # waits for the fusion-2 small-batch probe on PE 0-1
    running_keys = {r[0]: r[2] for r in running_all}
    for fusion in ORDER:
        for mode in ("eager", "compile"):
            for batch in BATCHES:
                key = (fusion, mode, batch)
                if key in done:
                    continue
                estimate = EST[(fusion, mode)][batch] * max(scale[mode], 1.3)
                if key in running_keys:
                    estimate = max(estimate - running_keys[key], 60)
                lane = {4: "A", 1: "B", 2: "C"}[fusion]
                if (fusion, mode, batch) in MOVED:
                    lane += "2"
                lanes[lane] = lanes.get(lane, 0.0) + estimate + 15
    if not RERUN_DONE():   # after the f2x run, on PE 4-7
        lanes["C2"] = lanes.get("C2", 0.0) + 470 * 1.3
    remaining = max(lanes.values())
    now = datetime.now(timezone.utc)
    ok = sum(1 for s, _ in done.values() if s == "ok")
    bad = len(done) - ok
    current = ", ".join(f"f{r[0][0]} {r[0][1]}-b{r[0][2]} [{r[1]}, {fmt(r[2])}]" for r in running_all) or "-"
    print(f"{now.astimezone(KST).strftime('%H:%M')} 경과 {fmt((now - START).total_seconds())} | "
          f"완료 {len(done)}/18 (성공 {ok}, 실패 {bad}) | 현재 {current} | 남은 예상 {fmt(remaining)} "
          f"(종료 예상 {datetime.fromtimestamp(now.timestamp() + remaining, KST).strftime('%m-%d %H:%M')})",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
