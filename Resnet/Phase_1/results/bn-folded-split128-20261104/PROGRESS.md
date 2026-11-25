# 실험 진행 상태

갱신: 2026-11-04T13:19:21.707618+09:00

- 변형: `bn_folded_output_split128`
- 상태: `stopped_by_user` / 단계: `stopped`
- 시작: 2026-11-04T13:05:42.894428+09:00
- runner PID: 32722
- 최소 조건: `pending`
- 성공: 1 / 실패: 0 / 미실행: 0

| 실행 | 상태 | 경과 s | worker PID | 상세 |
| --- | --- | ---: | ---: | --- |
| eager-b1-cold | ok | 460.7 | 32749 | [로그](jobs/eager-b1-cold.log) / [상태](jobs/eager-b1-cold/progress.json) |
| compile-b1-cold | interrupted | 300.0 | 34754 | [로그](jobs/compile-b1-cold.log) / [상태](jobs/compile-b1-cold/progress.json) |

[전체 manifest](manifest.json)
 / [summary.csv](summary.csv)
 / [correctness.csv](correctness.csv)
 / [failures.csv](failures.csv)
 / [amortization_estimates.csv](amortization_estimates.csv)

통계는 성공한 본 측정 값만 포함한다. 첫 호출·워밍업·profile은 별도로 기록한다.
상세 진행은 worker 상태 파일과 로그를 확인한다. 수치 오차의 데이터셋 정확도 합격 기준은 별도다.
