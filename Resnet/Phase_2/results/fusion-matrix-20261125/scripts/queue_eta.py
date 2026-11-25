"""남은 시간 한 줄 표시: 순차 큐(serial_queue.sh)의 queue.jsonl과 단계별 추정치로 계산.

추정치는 단독 실행 기준 가정값이다(이 조합들은 단독으로 잰 적이 없음). 끝난 단계의 실제/추정 비율의
평균으로 남은 단계를 보정한다.
"""
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))
DIR = Path(__file__).resolve().parents[1]
STEPS = [("f4-eager-b1-rerun", "f4 eager b1 재측정", 600), ("f2-eager-b16", "f2 eager b16", 1500),
         ("f1-eager-b64", "f1 eager b64", 2700), ("f1-compile-b64", "f1 compile b64", 3600),
         ("f2-b64-probe", "f2 b64 op 진단", 5400), ("f2-compile-eager-b64", "f2 compile b64 + eager b64", 5700)]


def fmt(s):
    s = int(max(0, s))
    return f"{s // 3600}시간 {s % 3600 // 60:02d}분"


def main():
    events = [json.loads(l) for l in (DIR / "queue.jsonl").read_text().splitlines()] if (DIR / "queue.jsonl").exists() else []
    start, end, skipped = {}, {}, set()
    for e in events:
        if e["event"] == "start":
            start[e["step"]] = e["t"]
        elif e["event"] == "end":
            end[e["step"]] = e["t"]
        elif e["event"] == "skip":
            skipped.add(e["step"])
    ratios = [(end[s] - start[s]) / est for s, _, est in STEPS if s in end and s in start]
    scale = sum(ratios) / len(ratios) if ratios else 1.0
    now, remaining, current, done = time.time(), 0.0, "대기", 0
    for step, label, est in STEPS:
        if step in end or step in skipped:
            done += 1
            continue
        if step in start:
            elapsed = now - start[step]
            remaining += max(est * scale - elapsed, 300)
            current = f"{label} ({fmt(elapsed)} 경과, 추정 {fmt(est * scale)})"
        else:
            remaining += est * scale
    finish = datetime.fromtimestamp(now + remaining, KST).strftime("%H:%M")
    all_done = any(e["step"] == "all" for e in events)
    print(f"{datetime.now(KST):%H:%M} | 단계 {done}/{len(STEPS)} 완료 | 현재: {'모두 완료' if all_done else current} | "
          f"남은 예상 {fmt(0 if all_done else remaining)} (종료 예상 {finish}, 보정계수 {scale:.2f})")


if __name__ == "__main__":
    main()
