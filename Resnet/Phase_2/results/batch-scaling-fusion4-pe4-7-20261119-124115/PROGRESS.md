# Phase 2 진행 상태

갱신: 2026-11-20T00:09:42.501452+09:00

- 상태: `finished`
- 시작: 2026-11-19T12:41:18.679384+09:00
- runner PID: 186536
- PE fusion: 4, 장치 furiosa:1 (물리 PE [4, 5, 6, 7]) / 재개 지점 `eager-b2-t1`
- 계획 58개 / 종료 39개 (성공 35, 실패·timeout 4) / 건너뜀 19개
- worker 누적 시간: 41290 s / 예산 43200 s

| 실행 | 단계 | 상태 | 경과 s | 첫 호출 s |
| --- | --- | --- | ---: | ---: |
| eager-b2-t1 | core | failed | 7.1 |  |
| eager-b4-t1 | core | ok | 494.0 | 488.1 |
| compile-b4-t1 | core | ok | 889.2 | 882.6 |
| compile-b8-t1 | core | ok | 954.6 | 947.7 |
| eager-b8-t1 | core | ok | 576.0 | 570.0 |
| eager-b16-t1 | core | failed | 601.4 |  |
| compile-b16-t1 | core | ok | 1011.0 | 1003.3 |
| compile-b32-t1 | core | ok | 1421.7 | 1411.9 |
| eager-b32-t1 | core | failed | 1152.2 |  |
| compile-b1-t2 | core | ok | 501.5 | 495.6 |
| eager-b1-t2 | core | ok | 465.8 | 459.9 |
| compile-b2-t2 | core | failed | 30.5 |  |
| compile-b4-t2 | core | ok | 897.2 | 890.7 |
| eager-b4-t2 | core | ok | 496.5 | 490.6 |
| eager-b8-t2 | core | ok | 578.7 | 572.7 |
| compile-b8-t2 | core | ok | 942.5 | 935.5 |
| compile-b16-t2 | core | ok | 1041.7 | 1034.1 |
| compile-b32-t2 | core | ok | 1432.4 | 1422.5 |
| eager-b1-t3 | core | ok | 465.8 | 460.1 |
| compile-b1-t3 | core | ok | 505.0 | 499.1 |
| eager-b4-t3 | core | ok | 496.5 | 490.6 |
| compile-b4-t3 | core | ok | 911.4 | 904.9 |
| compile-b8-t3 | core | ok | 945.5 | 938.5 |
| eager-b8-t3 | core | ok | 585.2 | 579.2 |
| compile-b16-t3 | core | ok | 1038.7 | 1031.1 |
| compile-b32-t3 | core | ok | 1442.4 | 1432.6 |
| compile-b1-t4 | revalidation_extra | ok | 504.6 | 498.7 |
| eager-b1-t4 | revalidation_extra | ok | 466.4 | 460.7 |
| eager-b1-t5 | revalidation_extra | ok | 466.1 | 460.3 |
| compile-b1-t5 | revalidation_extra | ok | 507.6 | 501.7 |
| compile-b64-t1 | extension | ok | 1789.0 | 1776.6 |
| compile-b64-t2 | extension | ok | 1815.4 | 1803.2 |
| compile-b64-t3 | extension | ok | 1774.8 | 1762.7 |
| compile-b128-t1 | extension | ok | 2184.7 | 2170.4 |
| compile-b128-t2 | extension | ok | 2205.1 | 2190.0 |
| compile-b128-t3 | extension | ok | 2267.3 | 2251.9 |
| compile-b256-t1 | extension | ok | 2474.4 | 2456.5 |
| compile-b256-t2 | extension | ok | 2491.7 | 2468.5 |
| compile-b256-t3 | extension | ok | 2459.1 | 2439.7 |
| eager-b1-t1 | core | skipped: before start point eager-b2-t1; ran in /root/Resnet/Phase_2/results/batch-scaling-1h-20261119-103155 with a different setup | | |
| compile-b1-t1 | core | skipped: before start point eager-b2-t1; ran in /root/Resnet/Phase_2/results/batch-scaling-1h-20261119-103155 with a different setup | | |
| compile-b2-t1 | core | skipped: before start point eager-b2-t1; ran in /root/Resnet/Phase_2/results/batch-scaling-1h-20261119-103155 with a different setup | | |
| eager-b2-t2 | core | skipped: same mode and batch already failed or timed out in an earlier trial | | |
| eager-b16-t2 | core | skipped: same mode and batch already failed or timed out in an earlier trial | | |
| eager-b32-t2 | core | skipped: same mode and batch already failed or timed out in an earlier trial | | |
| compile-b2-t3 | core | skipped: same mode and batch already failed or timed out in an earlier trial | | |
| eager-b2-t3 | core | skipped: same mode and batch already failed or timed out in an earlier trial | | |
| eager-b16-t3 | core | skipped: same mode and batch already failed or timed out in an earlier trial | | |
| eager-b32-t3 | core | skipped: same mode and batch already failed or timed out in an earlier trial | | |
| eager-b64-t1 | extension | skipped: extension stops for this mode: batch 32 did not succeed | | |
| eager-b64-t2 | extension | skipped: extension stops for this mode: batch 32 did not succeed | | |
| eager-b64-t3 | extension | skipped: extension stops for this mode: batch 32 did not succeed | | |
| eager-b128-t1 | extension | skipped: extension stops for this mode: batch 32 did not succeed | | |
| eager-b128-t2 | extension | skipped: extension stops for this mode: batch 32 did not succeed | | |
| eager-b128-t3 | extension | skipped: extension stops for this mode: batch 32 did not succeed | | |
| eager-b256-t1 | extension | skipped: extension stops for this mode: batch 32 did not succeed | | |
| eager-b256-t2 | extension | skipped: extension stops for this mode: batch 32 did not succeed | | |
| eager-b256-t3 | extension | skipped: extension stops for this mode: batch 32 did not succeed | | |
