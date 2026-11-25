"""File-based run progress for long NPU compilation jobs."""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def write_progress(root, manifest):
    root = Path(root)
    lines = ["# 실험 진행 상태", "",
             f"갱신: {datetime.now(ZoneInfo('Asia/Seoul')).isoformat()}", "",
             f"- 변형: `{manifest.get('model_variant', manifest['target'])}`",
             f"- 상태: `{manifest['status']}` / 단계: `{manifest.get('phase', 'preparing')}`",
             f"- 시작: {manifest['started_kst']}",
             f"- runner PID: {manifest.get('runner_pid')}",
             f"- 최소 조건: `{manifest.get('support_gate', {}).get('status', 'pending')}`",
             f"- 성공: {sum(j['status'] == 'ok' for j in manifest['jobs'])} / 실패: {sum(j['status'] in ('failed', 'timeout', 'process_failed') for j in manifest['jobs'])} / 미실행: {len(manifest['skipped'])}", "",
             "| 실행 | 상태 | 경과 s | worker PID | 상세 |",
             "| --- | --- | ---: | ---: | --- |"]
    if manifest.get("deadline_kst"):
        lines[5:5] = [f"- 종료 제한: {manifest['deadline_kst']} (전체 {manifest['max_runtime_s']}초 이내)"]
    if manifest.get("experiment") == "isolated_module_latency":
        lines[5:5] = ["- 계측: 배치 1, Eager/Compile × cold/reuse 4개 케이스; 모듈/연산 독립 실행 + NPU 동기화",
                      "- Compile은 모듈별 독립 컴파일이며 전체 모델 fusion 성능과 구분한다."]
    for job in manifest["jobs"]:
        parent = "profiles" if job.get("profile") else "jobs"
        name = job["name"]
        seconds = job.get("elapsed_s")
        elapsed = f"{seconds:.1f}" if seconds is not None else ""
        lines.append(f"| {name} | {job['status']} | {elapsed} | {job.get('worker_pid', '')} | [로그]({parent}/{name}.log) / [상태]({parent}/{name}/progress.json) |")
    for job in manifest["jobs"]:
        if job.get("current_node") and job["status"] == "running":
            lines += ["", f"현재 연산: `{job['current_node']}` / 모듈: `{job.get('current_module')}`",
                      f"처리 완료: {job.get('completed_nodes')} / {job.get('total_nodes')}; 실패: {job.get('failed_nodes')}"]
    lines += ["", "[전체 manifest](manifest.json)"]
    for name in ("minimum_condition.json", "summary.csv", "correctness.csv", "failures.csv", "amortization_estimates.csv", "module_summary.csv", "module_comparison.csv"):
        if (root / name).exists():
            lines.append(f" / [{name}]({name})")
    lines += ["", "통계는 성공한 본 측정 값만 포함한다. 첫 호출·워밍업·profile은 별도로 기록한다.",
              "상세 진행은 worker 상태 파일과 로그를 확인한다. 수치 오차의 데이터셋 정확도 합격 기준은 별도다.", ""]
    temporary = root / "PROGRESS.md.tmp"
    temporary.write_text("\n".join(lines))
    temporary.replace(root / "PROGRESS.md")
