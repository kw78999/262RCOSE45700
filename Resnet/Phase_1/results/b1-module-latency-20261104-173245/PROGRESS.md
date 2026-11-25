# 실험 진행 상태

갱신: 2026-11-04T17:50:17.428932+09:00

- 변형: `bn_folded_output_split128`
- 계측: 배치 1, Eager/Compile × cold/reuse 4개 케이스; 모듈/연산 독립 실행 + NPU 동기화
- Compile은 모듈별 독립 컴파일이며 전체 모델 fusion 성능과 구분한다.
- 종료 제한: 2026-11-04T18:29:32.778651+09:00 (전체 3407초 이내)
- 상태: `complete` / 단계: `finished`
- 시작: 2026-11-04T17:32:48.631144+09:00
- runner PID: 74996
- 최소 조건: `per_node_validation`
- 성공: 4 / 실패: 0 / 미실행: 0

| 실행 | 상태 | 경과 s | worker PID | 상세 |
| --- | --- | ---: | ---: | --- |
| eager-b1-cold | ok | 475.4 | 75008 | [로그](jobs/eager-b1-cold.log) / [상태](jobs/eager-b1-cold/progress.json) |
| eager-b1-reuse | ok | 33.9 | 77970 | [로그](jobs/eager-b1-reuse.log) / [상태](jobs/eager-b1-reuse/progress.json) |
| compile-b1-cold | ok | 491.0 | 78046 | [로그](jobs/compile-b1-cold.log) / [상태](jobs/compile-b1-cold/progress.json) |
| compile-b1-reuse | ok | 48.5 | 80975 | [로그](jobs/compile-b1-reuse.log) / [상태](jobs/compile-b1-reuse/progress.json) |

[전체 manifest](manifest.json)
 / [module_summary.csv](module_summary.csv)
 / [module_comparison.csv](module_comparison.csv)

통계는 성공한 본 측정 값만 포함한다. 첫 호출·워밍업·profile은 별도로 기록한다.
상세 진행은 worker 상태 파일과 로그를 확인한다. 수치 오차의 데이터셋 정확도 합격 기준은 별도다.
