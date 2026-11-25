# 실험 진행 상태

갱신: 2026-11-04T17:17:31.612696+09:00

- 변형: `bn_folded_output_split128`
- 종료 제한: 2026-11-04T18:29:35.123505+09:00 (전체 7200초 이내)
- 상태: `stopped_by_user` / 단계: `finished`
- 시작: 2026-11-04T16:29:37.976317+09:00
- runner PID: 59767
- 최소 조건: `passed`
- 성공: 5 / 실패: 0 / 미실행: 0

| 실행 | 상태 | 경과 s | worker PID | 상세 |
| --- | --- | ---: | ---: | --- |
| eager-b1-cold | ok | 461.3 | 59780 | [로그](jobs/eager-b1-cold.log) / [상태](jobs/eager-b1-cold/progress.json) |
| compile-b1-cold | ok | 424.2 | 62670 | [로그](jobs/compile-b1-cold.log) / [상태](jobs/compile-b1-cold/progress.json) |
| compile-b1-reuse | ok | 7.7 | 65931 | [로그](jobs/compile-b1-reuse.log) / [상태](jobs/compile-b1-reuse/progress.json) |
| eager-b1-reuse | ok | 18.5 | 65958 | [로그](jobs/eager-b1-reuse.log) / [상태](jobs/eager-b1-reuse/progress.json) |
| compile-b4-cold | ok | 1179.4 | 66003 | [로그](jobs/compile-b4-cold.log) / [상태](jobs/compile-b4-cold/progress.json) |
| eager-b4-cold | interrupted | 450.0 | 71006 | [로그](jobs/eager-b4-cold.log) / [상태](jobs/eager-b4-cold/progress.json) |

[전체 manifest](manifest.json)
 / [minimum_condition.json](minimum_condition.json)
 / [summary.csv](summary.csv)
 / [correctness.csv](correctness.csv)
 / [failures.csv](failures.csv)
 / [amortization_estimates.csv](amortization_estimates.csv)

통계는 성공한 본 측정 값만 포함한다. 첫 호출·워밍업·profile은 별도로 기록한다.
상세 진행은 worker 상태 파일과 로그를 확인한다. 수치 오차의 데이터셋 정확도 합격 기준은 별도다.

현재 모듈별 세부 측정: [새 PROGRESS.md](../b1-module-latency-20261104-173245/PROGRESS.md).
