# Attention CPU fallback: 원인·해결 실험

**[실행 확인] native MHA fallback을 재현했고, fastpath 해제와 사용자 코드 경계의 Attention-only Compile 모두 B1/B4 strict 진단·유한 출력·Top-1 일치성을 확보했다.** 부분 Compile은 실행 가능하지만, 이번 전체 ViT 반복 호출은 전체 Compile이 더 짧았다. 단일 trial의 예비 결과다.

[소스/실행 확인] Eager의 native MHA fastpath에는 Furiosa 전용 kernel이 없어 CPU fallback이 발생했다. fastpath 해제 또는 Compile의 functional MHA/SDPA 경로에서는 strict B1/B4가 성공했다. 기본 MHA에 직접 Module.compile을 적용하면 Dynamo top-frame skip으로 backend 진입이 0회였다. 기존 MHA를 그대로 호출하는 사용자 코드 경계를 Compile하자 실제 backend 진입과 정상 출력을 확인했다.

| 조건 | B | first ms | repeat 평균 ± SD ms | images/s |
| --- | --- | --- | --- | --- |
| 기본 Eager (CPU fallback 포함) | 1 | 61894.5772 | 1046.970 ± 33.268 | 0.9551 |
| Attention 경계만 Compile | 1 | 113301.6091 | 40.886 ± 0.172 | 24.4584 |
| Eager, MHA fastpath 해제 | 1 | 97833.8038 | 44.149 ± 0.096 | 22.6505 |
| 전체 Compile | 1 | 204166.3484 | 3.817 ± 0.771 | 261.9563 |
| 기본 Eager (CPU fallback 포함) | 4 | 90765.5893 | 1095.124 ± 37.339 | 3.6526 |
| Attention 경계만 Compile | 4 | 208352.8310 | 118.172 ± 0.072 | 33.8491 |
| Eager, MHA fastpath 해제 | 4 | 166638.7932 | 122.648 ± 0.148 | 32.6137 |
| 전체 Compile | 4 | 437272.8579 | 10.454 ± 0.007 | 382.6378 |

독립 trial 1회, warmup 5회 뒤 repeat 20회. SD는 ddof=1이며 CI가 아니다. 시간은 전체 ViT 배치 호출부터 current-stream 완료까지로 내부 dispatch/전송을 포함한다. 기본 Eager는 CPU/NPU 혼합 기준선이다. first-call은 순수 compile 시간이 아니다. 전용 SDK cache는 처음 없던 경로이며 시스템 cold·실제 hit/miss는 미확인이다.

**변동 주의:** 전체 Compile B1은 repeat 17의 7.094ms를 포함해 `not_stable`로 분류했다. 나머지 호출은 약 3.63–3.66ms였고 느린 정상 호출을 삭제하지 않았다. 원인은 미확인이다. 다른 조건은 휴리스틱상 preliminary_stable이며 steady state 증명은 아니다.

| strict/경로 진단 | B | native MHA host 횟수 | NPU task | backend 진입 |
| --- | --- | --- | --- | --- |
| eager_default | 1 | 12 | 125 | 0 |
| eager_fastpath_off | 1 | 0 | 185 | 0 |
| eager_default | 4 | 12 | 125 | 0 |
| eager_fastpath_off | 4 | 0 | 184 | 0 |
| Attention 경계 | 1 | 0 | 137 | 1 |
| Attention 경계 | 4 | 0 | 137 | 1 |

대표 호출의 NPU task는 codegen 횟수가 아니다. 실제 codegen/cache 계수는 unknown이다. strict 성공은 SDK 제어 계약의 범위이며 전체 연산 배치 증명은 아니다.

| 조건 | B | 최대/평균 절대오차 | 최소 cosine | Top-1 일치 |
| --- | --- | --- | --- | --- |
| Attention 경계만 Compile | 1 | 0.03125 / 0.00574 | 0.99991040 | 1/1 |
| Eager, MHA fastpath 해제 | 1 | 0.04199 / 0.00619 | 0.99989062 | 1/1 |
| 전체 Compile | 1 | 0.02148 / 0.00421 | 0.99995221 | 1/1 |
| Attention 경계만 Compile | 4 | 0.04688 / 0.00508 | 0.99993405 | 4/4 |
| Eager, MHA fastpath 해제 | 4 | 0.03125 / 0.00534 | 0.99991918 | 4/4 |
| 전체 Compile | 4 | 0.06250 / 0.00475 | 0.99994270 | 4/4 |

실제 사진 고유 4장, 참조는 같은 입력의 이번 기본 Eager logits다. 모든 출력은 [B,1000] BF16 유한값이다. B1은 B4의 첫 사진이므로 고유 표본 5장이 아니다. 정답 라벨이 없어 accuracy/loss는 계산하지 않았다. 허용 오차 판정은 pending이다.

**조건부 진행:** Attention만 Compile은 가능하며, 이번 전체 호출 시간은 전체 Compile이 가장 짧았다. fallback 제거 뒤에도 Eager dispatch/프로그램 호출 비용이 남을 수 있으나 비용 비중은 미확인이다. BF16 출력 허용 기준 합의와 선택 경로의 독립 trial 확대가 다음 최소 조치다. 피드백 질문은 SDK native MHA 지원/dispatch 개선 가능 여부와 BF16 atol/rtol 기준이다.

원본/상세: [study_report.md](study_report.md), [검증](study_verification.json), [그림](figures/01_attention_paths.png). 누적 worker 2320.4초/7200초. 기존 결과·환경은 보존했다.
