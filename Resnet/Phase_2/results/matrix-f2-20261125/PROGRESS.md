# Phase 2 진행 상태

갱신: 2026-11-25T18:01:50.217648+09:00

- 상태: `finished`
- 시작: 2026-11-25T12:09:22.209189+09:00
- runner PID: 11918
- PE fusion: 2, 장치 furiosa:0 (물리 PE [0, 1])
- 계획 6개 / 종료 5개 (성공 3, 실패·timeout 0) / 건너뜀 1개
- worker 누적 시간: 6514 s / 예산 36000 s

| 실행 | 단계 | 상태 | 경과 s | 첫 호출 s |
| --- | --- | --- | ---: | ---: |
| compile-b1-t1 | core | ok | 1339.5 | 1329.8 |
| compile-b16-t1 | core | ok | 4046.4 | 4029.9 |
| eager-b16-t1 | core | ok | 1121.6 | 1104.7 |
| eager-b64-t1 | core | moved | 3.3 |  |
| compile-b64-t1 | core | moved | 3.3 |  |
| eager-b1-t1 | core | skipped: PEs [0, 1] held by another process for 30 minutes | | |
