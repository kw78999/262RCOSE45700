"""Tables for the PE-fusion matrix (fusion 1, 2, 4 x eager, compile x batch 1, 16, 64; one trial each).

Reads results/matrix-f{1,2,4}-*/jobs/*/{result.json,samples.csv} and writes, next to them in
results/fusion-matrix-<date>/:
  matrix_summary.csv  one row per job, every number below
  MATRIX_REPORT.md    forward-only total time and throughput, per-stage latency breakdown,
                      Eager / Compile ratio and PE-fusion scaling

Scopes (all from the worker, one fresh process + fresh compiler cache per job):
  forward      'forward_sync' of the aux loop: forward on a pre-staged NPU input + stream
               synchronize, no transfer (the "forward only" time), 50 repeats
  h2d / d2h    aux loop, input copy to NPU / output copy to CPU, each closed by synchronize
  e2e          aux loop, h2d + forward + d2h of the same iteration
  primary      'forward_to_cpu' repeat loop: forward until output.to("cpu") returned, 50 repeats
  first call   first forward of the process, compile + load inclusive
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

FUSIONS, MODES, BATCHES = (1, 2, 4), ("eager", "compile"), (1, 16, 64)
# Known measurement caveats, marked † in the tables (see fusion-matrix-20261125/NOTES.md).
CAVEATS = {
    (4, "eager", 64): "측정 구간 보호가 0.2 s 폴링 gate 방식(이후 job은 worker가 직접 잠금). 구간 분리 루프 forward "
                      "50회는 120.5–121.5 ms로 깨끗하고, 반복 루프에 254 ms 이상치 1회",
}


def p95(values):
    values = sorted(values)
    k = (len(values) - 1) * 0.95
    lo = int(k)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (k - lo)


def pause_intervals(matrix_dir):
    """[(t0, t1, pgids)] when other benchmark processes were SIGSTOPped (worker log + gate log)."""
    import datetime as dt
    out = []
    log = matrix_dir / "pauses.jsonl"
    if log.exists():
        for line in log.read_text().splitlines():
            e = json.loads(line)
            out.append((e["t0"], e["t1"], set(e["pgids"])))
    gate = matrix_dir / "gate.log"
    if gate.exists():
        lines = gate.read_text().splitlines()
        for i, line in enumerate(lines):
            if " STOP all but " in line and i + 1 < len(lines):
                t0 = dt.datetime.fromisoformat(line.split()[0]).replace(tzinfo=dt.timezone.utc).timestamp()
                t1 = dt.datetime.fromisoformat(lines[i + 1].split()[0]).replace(tzinfo=dt.timezone.utc).timestamp() + 1
                keep = int(line.split("(pgid ")[1].split(")")[0]) if "(pgid " in line and "None" not in line else None
                groups = set(json.loads(line.split("groups=")[1]))
                out.append((t0, t1, groups - {keep}))
    return out


PAUSES = []


def job_row(fusion, mode, batch, runs):
    """The job's result from any run of this fusion; a `moved` pointer is followed to its target run."""
    row = dict(fusion=fusion, mode=mode, batch=batch, status="not_run")
    found = [run / "jobs" / f"{mode}-b{batch}-t1" for run in runs]
    found = [j for j in found if (j / "result.json").exists()
             and json.loads((j / "result.json").read_text()).get("status") != "moved"]
    if not found:
        return row
    found.sort(key=lambda j: json.loads((j / "result.json").read_text())["status"] != "ok")
    job = found[0]
    row["run"] = job.parent.parent.name
    result = json.loads((job / "result.json").read_text())
    row.update(status=result["status"], device=result.get("config", {}).get("device"),
               physical_pes=" ".join(map(str, result.get("physical_pes", []))),
               variant=result.get("model_variant"), process_s=result.get("process_elapsed_s"))
    if result["status"] != "ok":
        row["error"] = (result.get("error_type", "") + ": " + result.get("error", ""))[:160]
        return row
    samples = list(csv.DictReader((job / "samples.csv").open()))

    def values(scope, phase):
        return [float(s["elapsed_ms"]) for s in samples if s["scope"] == scope and s["phase"] == phase]

    forward = values("forward_sync", "aux")
    primary = values("forward_to_cpu", "repeat")
    h2d, d2h, e2e = values("h2d", "aux"), values("d2h", "aux"), values("e2e", "aux")
    paused = 0.0
    if result.get("first_call_wall") and result.get("pgid"):
        a, b = result["first_call_wall"]
        spans = sorted((max(a, t0), min(b, t1)) for t0, t1, groups in PAUSES
                       if result["pgid"] in groups and min(b, t1) > max(a, t0))
        end = a
        for s, e in spans:   # union of overlapping intervals
            if e > end:
                paused += e - max(s, end)
                end = e
    row.update(
        first_call_s=result["first_call_ms"] / 1e3,
        first_call_paused_s=paused if result.get("first_call_wall") else None,
        first_call_active_s=result["first_call_ms"] / 1e3 - paused,
        forward_n=len(forward), forward_mean_ms=statistics.fmean(forward),
        forward_median_ms=statistics.median(forward), forward_sd_ms=statistics.stdev(forward),
        forward_p95_ms=p95(forward), forward_total_s=sum(forward) / 1e3,
        forward_throughput=batch * 1000 / statistics.fmean(forward),
        h2d_mean_ms=statistics.fmean(h2d), d2h_mean_ms=statistics.fmean(d2h),
        e2e_mean_ms=statistics.fmean(e2e), e2e_throughput=batch * 1000 / statistics.fmean(e2e),
        primary_mean_ms=statistics.fmean(primary), primary_median_ms=statistics.median(primary),
        primary_p95_ms=p95(primary), primary_total_s=sum(primary) / 1e3,
        load_cache_misses_per_forward=(result["eager_cache_after_repeat"]["_compile_and_load_cached"]["misses"]
                                       - result["eager_cache_after_warmup"]["_compile_and_load_cached"]["misses"])
        / len(primary) if mode == "eager" else None,
        op_calls_per_forward=sum(result["eager_cache_after_repeat"]["_compile_and_load_cached"][k]
                                 - result["eager_cache_after_warmup"]["_compile_and_load_cached"][k]
                                 for k in ("hits", "misses")) / len(primary) if mode == "eager" else None)
    return row


def cell(row, key, fmt="{:.2f}"):
    if row["status"] != "ok":
        return {"not_run": "미실행", "failed": "실패", "timeout": "timeout"}.get(row["status"], row["status"])
    value = row.get(key)
    return "–" if value is None else fmt.format(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = {f: sorted(args.results.glob(f"matrix-f{f}-*")) + sorted(args.results.glob(f"matrix-f{f}x-*")) for f in FUSIONS}
    PAUSES.extend(pause_intervals(args.output))
    rows = [job_row(f, m, b, runs[f]) for f in FUSIONS for m in MODES for b in BATCHES]
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (args.output / "matrix_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    get = {(r["fusion"], r["mode"], r["batch"]): r for r in rows}
    md = ["# ResNet50 PE fusion × 모드 × 배치 실험 (18 조건)", "",
          "조건마다 새 프로세스 + 새 compiler 캐시, 1회차. 첫 호출 1회 → 워밍업 10 → 반복 50, 이어서 구간 분리 루프"
          " (워밍업 5 → 50회). 빈 PE를 쓰려고 job마다 물리 PE가 다르다(표 1의 물리 PE 열); 측정 구간은 한 번에"
          " 한 job만, 나머지 프로세스는 모두 일시정지한 상태에서 잰다. BF16, `bn_folded_output_split128`, CPU fallback 끔.", "",
          f"**진행: 18개 중 {sum(r['status'] == 'ok' for r in rows)}개 측정 완료.** 실행 경과(병렬 실행, 컨테이너 재시작 2회,"
          " 순차 마무리)는 `NOTES.md`. † = 측정 조건 주의(4절), ‡ = Eager 배치 분할 변형 포함"
          " (분할 크기가 fusion마다 달라 같은 모델 구성의 비교가 아님).", "",
          "## 1. Forward 실행 시간과 처리량", "",
          "Forward = 입력이 NPU에 있는 상태에서 forward + stream 동기화 (전송 제외). 합계는 50회 합.", "",
          "| PE fusion | 모드 | 배치 | 물리 PE | Forward 평균 (ms) | 중앙값 / P95 (ms) | Forward 합계 50회 (s) | 처리량 (img/s) | E2E 처리량 (img/s) |",
          "| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for f in FUSIONS:
        for m in MODES:
            for b in BATCHES:
                r = get[(f, m, b)]
                mark = " †" if (f, m, b) in CAVEATS and r["status"] == "ok" else ""
                md.append(f"| {f} | {m} | {b} | {r.get('physical_pes', '–').replace(' ', ',')} | {cell(r, 'forward_mean_ms')}{mark} | "
                          f"{cell(r, 'forward_median_ms')} / {cell(r, 'forward_p95_ms')} | "
                          f"{cell(r, 'forward_total_s', '{:.3f}')} | {cell(r, 'forward_throughput', '{:,.0f}')} | "
                          f"{cell(r, 'e2e_throughput', '{:,.0f}')} |")
    md += ["", "## 2. 구간별 지연 breakdown (평균 ms, 비중은 E2E 대비)", "",
           "| PE fusion | 모드 | 배치 | H2D | Forward | D2H | E2E | H2D / Forward / D2H 비중 | 첫 호출 (s)* |",
           "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |"]
    for f in FUSIONS:
        for m in MODES:
            for b in BATCHES:
                r = get[(f, m, b)]
                share = "–"
                if r["status"] == "ok":
                    e = r["e2e_mean_ms"]
                    share = " / ".join(f"{100 * r[k] / e:.0f}%" for k in ("h2d_mean_ms", "forward_mean_ms", "d2h_mean_ms"))
                md.append(f"| {f} | {m} | {b} | {cell(r, 'h2d_mean_ms', '{:.3f}')} | {cell(r, 'forward_mean_ms')} | "
                          f"{cell(r, 'd2h_mean_ms', '{:.3f}')} | {cell(r, 'e2e_mean_ms')} | {share} | "
                          f"{cell(r, 'first_call_active_s', '{:,.0f}')} |")
    md += ["", "\\* 첫 호출 = 컴파일·로드 포함 첫 forward. 다른 job의 측정 구간 동안 일시정지된 시간은 뺐다. "
           "여러 job이 병렬로 컴파일했으므로(최대 4개 job + 컴파일 전용 probe) 단독 실행 때보다 길 수 있어 "
           "조건 간 절대 비교용이 아니다.", "", "## 3. Eager / Compile 배속과 PE fusion 효과 (Forward 평균 기준)", "",
           "| 배치 | E/C fusion 1 | E/C fusion 2 | E/C fusion 4 | Compile f1→f4 | Eager f1→f4 |",
           "| ---: | ---: | ---: | ---: | ---: | ---: |"]

    def ratio(a, b):
        if not a["status"] == b["status"] == "ok":
            return "–"
        split = {r.get("variant", "") for r in (a, b) if "eager_batch_chunk" in r.get("variant", "")}
        mark = " ‡" if split else ""
        mark += " †" if any((r["fusion"], r["mode"], r["batch"]) in CAVEATS for r in (a, b)) else ""
        return f"{a['forward_mean_ms'] / b['forward_mean_ms']:.2f}×{mark}"
    for b in BATCHES:
        md.append(f"| {b} | " + " | ".join(ratio(get[(f, 'eager', b)], get[(f, 'compile', b)]) for f in FUSIONS)
                  + f" | {ratio(get[(1, 'compile', b)], get[(4, 'compile', b)])} | "
                  f"{ratio(get[(1, 'eager', b)], get[(4, 'eager', b)])} |")
    notes = [f"- † fusion {f} {m} b{b}: {text}" for (f, m, b), text in CAVEATS.items()
             if get[(f, m, b)]["status"] == "ok"]
    for r in rows:
        if r["status"] == "ok" and r.get("variant", "").endswith(tuple(f"chunk{c}" for c in (1, 2, 4, 8))):
            notes.append(f"- fusion {r['fusion']} eager b{r['batch']}: `{r['variant']}` (컴파일 실패 op만 배치 분할), "
                         f"op 호출 {r['op_calls_per_forward']:.0f}회/forward")
        elif r["status"] not in ("ok", "not_run"):
            notes.append(f"- fusion {r['fusion']} {r['mode']} b{r['batch']}: {r['status']} — {r.get('error', '')}")
    notes.append("- 첫 호출 단독 실행 값(2026-11-25 17:35 이후 순차 실행): fusion 4 eager b1, fusion 2 eager b16·b64, "
                 "fusion 1 eager b64, fusion 1·2 compile b64. 나머지는 병렬 컴파일 중 측정")
    notes.append("- 메모리 최대 사용량(단독 실행, `mem_peaks.json`): compile b64 fusion 2 91.9 GiB, fusion 1 71.3 GiB; "
                 "eager job 최대 11.0 GiB. 컨테이너 한도 160 GiB")
    if notes:
        md += ["", "## 4. 비고", ""] + notes
    (args.output / "MATRIX_REPORT.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
