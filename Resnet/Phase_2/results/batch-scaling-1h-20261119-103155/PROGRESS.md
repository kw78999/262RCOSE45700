# Phase 2 진행 상태

갱신: 2026-11-19T11:32:01.527400+09:00

- 상태: `finished`
- 시작: 2026-11-19T10:31:58.809618+09:00
- runner PID: 60816
- 계획 58개 / 종료 6개 (성공 3, 실패·timeout 3) / 건너뜀 52개
- worker 누적 시간: 3602 s / 예산 3600 s

| 실행 | 단계 | 상태 | 경과 s | 첫 호출 s |
| --- | --- | --- | ---: | ---: |
| eager-b1-t1 | core | ok | 466.6 | 460.4 |
| compile-b1-t1 | core | ok | 432.9 | 426.6 |
| compile-b2-t1 | core | ok | 665.8 | 659.0 |
| eager-b2-t1 | core | failed | 380.9 |  |
| eager-b4-t1 | core | failed | 568.1 |  |
| compile-b4-t1 | core | timeout | 1087.7 |  |
| compile-b8-t1 | core | skipped: worker-time budget exhausted before this job started | | |
| eager-b8-t1 | core | skipped: worker-time budget exhausted before this job started | | |
| eager-b16-t1 | core | skipped: worker-time budget exhausted before this job started | | |
| compile-b16-t1 | core | skipped: worker-time budget exhausted before this job started | | |
| compile-b32-t1 | core | skipped: worker-time budget exhausted before this job started | | |
| eager-b32-t1 | core | skipped: worker-time budget exhausted before this job started | | |
| compile-b1-t2 | core | skipped: worker-time budget exhausted before this job started | | |
| eager-b1-t2 | core | skipped: worker-time budget exhausted before this job started | | |
| eager-b2-t2 | core | skipped: same mode and batch already failed or timed out in an earlier trial | | |
| compile-b2-t2 | core | skipped: worker-time budget exhausted before this job started | | |
| compile-b4-t2 | core | skipped: same mode and batch already failed or timed out in an earlier trial | | |
| eager-b4-t2 | core | skipped: same mode and batch already failed or timed out in an earlier trial | | |
| eager-b8-t2 | core | skipped: worker-time budget exhausted before this job started | | |
| compile-b8-t2 | core | skipped: worker-time budget exhausted before this job started | | |
| compile-b16-t2 | core | skipped: worker-time budget exhausted before this job started | | |
| eager-b16-t2 | core | skipped: worker-time budget exhausted before this job started | | |
| eager-b32-t2 | core | skipped: worker-time budget exhausted before this job started | | |
| compile-b32-t2 | core | skipped: worker-time budget exhausted before this job started | | |
| eager-b1-t3 | core | skipped: worker-time budget exhausted before this job started | | |
| compile-b1-t3 | core | skipped: worker-time budget exhausted before this job started | | |
| compile-b2-t3 | core | skipped: worker-time budget exhausted before this job started | | |
| eager-b2-t3 | core | skipped: same mode and batch already failed or timed out in an earlier trial | | |
| eager-b4-t3 | core | skipped: same mode and batch already failed or timed out in an earlier trial | | |
| compile-b4-t3 | core | skipped: same mode and batch already failed or timed out in an earlier trial | | |
| compile-b8-t3 | core | skipped: worker-time budget exhausted before this job started | | |
| eager-b8-t3 | core | skipped: worker-time budget exhausted before this job started | | |
| eager-b16-t3 | core | skipped: worker-time budget exhausted before this job started | | |
| compile-b16-t3 | core | skipped: worker-time budget exhausted before this job started | | |
| compile-b32-t3 | core | skipped: worker-time budget exhausted before this job started | | |
| eager-b32-t3 | core | skipped: worker-time budget exhausted before this job started | | |
| compile-b1-t4 | revalidation_extra | skipped: worker-time budget exhausted before this job started | | |
| eager-b1-t4 | revalidation_extra | skipped: worker-time budget exhausted before this job started | | |
| eager-b1-t5 | revalidation_extra | skipped: worker-time budget exhausted before this job started | | |
| compile-b1-t5 | revalidation_extra | skipped: worker-time budget exhausted before this job started | | |
| eager-b64-t1 | extension | skipped: extension stops for this mode: batch 32 did not succeed | | |
| compile-b64-t1 | extension | skipped: extension stops for this mode: batch 32 did not succeed | | |
| compile-b64-t2 | extension | skipped: extension stops for this mode: batch 32 did not succeed | | |
| eager-b64-t2 | extension | skipped: extension stops for this mode: batch 32 did not succeed | | |
| eager-b64-t3 | extension | skipped: extension stops for this mode: batch 32 did not succeed | | |
| compile-b64-t3 | extension | skipped: extension stops for this mode: batch 32 did not succeed | | |
| compile-b128-t1 | extension | skipped: extension stops for this mode: batch 32 did not succeed | | |
| eager-b128-t1 | extension | skipped: extension stops for this mode: batch 32 did not succeed | | |
| eager-b128-t2 | extension | skipped: extension stops for this mode: batch 32 did not succeed | | |
| compile-b128-t2 | extension | skipped: extension stops for this mode: batch 32 did not succeed | | |
| compile-b128-t3 | extension | skipped: extension stops for this mode: batch 32 did not succeed | | |
| eager-b128-t3 | extension | skipped: extension stops for this mode: batch 32 did not succeed | | |
| eager-b256-t1 | extension | skipped: extension stops for this mode: batch 32 did not succeed | | |
| compile-b256-t1 | extension | skipped: extension stops for this mode: batch 32 did not succeed | | |
| compile-b256-t2 | extension | skipped: extension stops for this mode: batch 32 did not succeed | | |
| eager-b256-t2 | extension | skipped: extension stops for this mode: batch 32 did not succeed | | |
| eager-b256-t3 | extension | skipped: extension stops for this mode: batch 32 did not succeed | | |
| compile-b256-t3 | extension | skipped: extension stops for this mode: batch 32 did not succeed | | |
