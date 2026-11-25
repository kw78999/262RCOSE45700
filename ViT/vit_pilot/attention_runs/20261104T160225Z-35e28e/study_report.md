# Attention CPU fallback: 원인·해결 실험

**[실행 확인] native MHA fallback을 재현했고, fastpath 해제와 사용자 코드 경계의 Attention-only Compile 모두 B1/B4 strict 진단·유한 출력·Top-1 일치성을 확보했다.** 부분 Compile은 실행 가능하지만, 이번 전체 ViT 반복 호출은 전체 Compile이 더 짧았다. 단일 trial의 예비 결과다.

## 왜 CPU fallback이 발생했는가

[실행 확인/소스 확인] 기본 Eager는 ViT의 eval/inference_mode self-attention에서 `aten::_native_multi_head_attention` fastpath를 선택한다. 설치 torch 2.10.0 + SDK 2026.4.0의 해당 연산에는 PrivateUse1 전용 kernel이 없고 SDK backend fallback이 등록돼 있다. CPU fallback을 금지한 실제 B1 호출은 이 연산에서 RuntimeError를 냈고, 기본 경로 CPU profiler는 native MHA 12회를 기록했다. 이는 Attention 전체가 NPU에서 불가능하다는 뜻이 아니다. `need_weights=False`도 native fastpath를 항상 막지는 않는다.

[소스 확인] PyTorch 2.10 Dynamo는 tracing 중 `_is_make_fx_tracing=True`로 취급해 native 경로를 건너뛴다. 실제 FX graph는 기존 functional MHA를 호출하며 Furiosa SDK에는 SDPA math 분해가 있다. [추정] CPU 계산·내부 복사·동기 대기가 큰 Eager 지연에 기여한다. 이번 ablation은 해당 실행 경로의 영향을 지지하지만 비용별 비중은 측정하지 않았다. CPU host scope를 순수 CPU/NPU kernel 시간으로 읽거나 중첩 시간을 더해 원인별 비율을 만들지 않는다.

[소스/실행 확인] 기본 `self_attention.compile()`만 적용한 첫 시도는 backend 진입 0회로 동일하게 실패했다. 설치 Dynamo는 기본 MHA 메서드를 top frame에서 생략하고 다른 사용자 코드 안에서는 포함한다. 따라서 기존 MHA를 그대로 호출하는 `AttentionBoundary`를 준비 단계에 넣고 그 경계를 compile했다. 12개 모두 적용했고 원래 Parameter 객체 집합·q/k/v·kwargs를 유지했다. wrapper는 수식을 바꾸지 않으며 나머지 ViT는 Eager다. 이 방식은 strict B1/B4에서 성공했다.

## 전체 ViT 배치 호출 시간

실제 사진 4장, BF16, 224×224, 기본 1 PE, 각 조건 새 worker, 독립 trial 1회, warmup 5회 뒤 repeat 20회다. CPU dispatch부터 공식 current-stream 완료까지 ms/batch-call로 측정했다. 외부 준비·전처리·입출력 전송은 제외하고 SDK 내부 복사·dispatch·대기는 포함한다. 첫 호출과 반복은 별도이며 순수 compiler/NPU kernel 시간이 아니다. 표준편차는 ddof=1이고 CI가 아니다.

| 조건 | B | setup ms | first ms | repeat 평균 ± SD ms | 중앙값 ms | images/s |
| --- | --- | --- | --- | --- | --- | --- |
| 기본 Eager (CPU fallback 포함) | 1 | 0.0013 | 61894.5772 | 1046.9703 ± 33.2683 | 1054.0137 | 0.9551 |
| Attention 경계만 Compile | 1 | 149.1720 | 113301.6091 | 40.8857 ± 0.1725 | 40.9425 | 24.4584 |
| Eager, MHA fastpath 해제 | 1 | 0.0011 | 97833.8038 | 44.1492 ± 0.0957 | 44.1681 | 22.6505 |
| 전체 Compile | 1 | 148.2916 | 204166.3484 | 3.8174 ± 0.7713 | 3.6454 | 261.9563 |
| 기본 Eager (CPU fallback 포함) | 4 | 0.0011 | 90765.5893 | 1095.1236 ± 37.3389 | 1094.7518 | 3.6526 |
| Attention 경계만 Compile | 4 | 149.0097 | 208352.8310 | 118.1715 ± 0.0719 | 118.1593 | 33.8491 |
| Eager, MHA fastpath 해제 | 4 | 0.0013 | 166638.7932 | 122.6479 ± 0.1476 | 122.6615 | 32.6137 |
| 전체 Compile | 4 | 147.3442 | 437272.8579 | 10.4538 ± 0.0073 | 10.4536 | 382.6378 |

각 worker는 처음 없던 전용 SDK cache를 사용했다. 시스템 전체 cold와 실제 hit/miss는 미확인이다. 조건별 실행은 순차였으며 Attention 경계 조건은 실패 조사 뒤 별도 run에서 실행했다. 순서 효과를 통제한 혼합 workload나 통계적 우열 검정은 아니다. 기본 Eager는 CPU/NPU 혼합 실행이므로 이 표를 NPU-only Eager 대 Compile 배속으로 해석하지 않는다. throughput은 순차 반복 처리량이다. 원래 직접 MHA compile 실패·B4/성능 보류는 첫 run의 manifest와 summary에 보존했다.

[실행 확인] 전체 Compile B1는 `not_stable`이다. repeat 17이 7.094043ms로 가장 길었다. 느린 정상 호출도 모두 평균/SD에 포함했다. 시간 증가만으로 재컴파일 또는 cache miss라고 판정하지 않는다. `figures/02_repeat_order.png`에 호출 순서를 제시한다.

## 경로와 출력 확인

| 진단 | strict | 상태 | CPU native MHA 횟수 | NPU task events | backend 진입 |
| --- | --- | --- | --- | --- | --- |
| diagnostic_eager_default_b1_strict | True | error | — | — | 0 |
| diagnostic_eager_default_b1 | False | ok | 12 | 125 | 0 |
| diagnostic_attention_compile_b1_strict | True | error | — | — | 0 |
| diagnostic_eager_fastpath_off_b1_strict | True | ok | 0 | 185 | 0 |
| diagnostic_eager_default_b4 | False | ok | 12 | 125 | 0 |
| diagnostic_attention_compile_b4_strict | True | held | — | — | — |
| diagnostic_eager_fastpath_off_b4_strict | True | ok | 0 | 184 | 0 |
| diagnostic_attention_compile_b1_strict (경계) | True | ok | 0 | 137 | 1 |
| diagnostic_attention_compile_b4_strict (경계) | True | ok | 0 | 137 | 1 |

대표 호출의 native CPU MHA 부재, strict 성공, NPU task를 함께 확인한다. strict의 SDK 계약을 넘어 전체 연산 배치를 완전히 증명했다고 표현하지 않는다. backend 진입, NPU task, 실제 codegen, disk hit/miss는 서로 다르며 뒤의 두 항목은 null/unknown이다. 상세 로그·profiler는 별도 진단 worker에서만 사용했다. 전체 Compile의 이전 strict/trace 근거는 기존 followup run에 있고 이번 control 성능 호출에는 profiler를 넣지 않았다.

| 조건 | B | 최대 절대오차 | 평균 절대오차 | 최소 cosine | Top-1 일치 |
| --- | --- | --- | --- | --- | --- |
| 기본 Eager (CPU fallback 포함) | 1 | 0.0000 | 0.0000 | 1.00000000 | 1/1 |
| Attention 경계만 Compile | 1 | 0.0312 | 0.0057 | 0.99991040 | 1/1 |
| Eager, MHA fastpath 해제 | 1 | 0.0420 | 0.0062 | 0.99989062 | 1/1 |
| 전체 Compile | 1 | 0.0215 | 0.0042 | 0.99995221 | 1/1 |
| 기본 Eager (CPU fallback 포함) | 4 | 0.0000 | 0.0000 | 1.00000000 | 4/4 |
| Attention 경계만 Compile | 4 | 0.0469 | 0.0051 | 0.99993405 | 4/4 |
| Eager, MHA fastpath 해제 | 4 | 0.0312 | 0.0053 | 0.99991918 | 4/4 |
| 전체 Compile | 4 | 0.0625 | 0.0047 | 0.99994270 | 4/4 |

참조는 이번 control run에서 저장한 같은 배치 기본 Eager logits다. CPU FP32로 변환한 뒤 비교했으며 모든 출력은 [B,1000] BF16 유한값이다. atol/rtol 미합의로 tolerance_status=pending이다. 고유 입력은 4장이고 B1의 cat은 B4에도 포함된다. Top-1 일치는 정답 accuracy가 아니며 정답 라벨이 없어 accuracy/loss를 계산하지 않았다.

## 해결 방법과 다음 최소 조치

1. **전체 Compile:** 이번 조건에서는 가장 짧은 전체 호출 시간을 관측했다. 기존 코드에서 사용하는 해결 경로다.
2. **Attention 경계만 Compile:** 실행 가능하다. 사용자 코드 경계를 두고 실제 backend 진입과 출력을 확인해야 한다. 나머지 Eager 실행과 여러 프로그램 호출은 남는다.
3. **Eager MHA fastpath 해제:** worker 내부의 공개 `torch.backends.mha.set_fastpath_enabled(False)`로 native 연산을 피했다. 기본 Eager와 구분한 조건이며 원래 설정을 복원한다. 수치 차이는 별도로 검증해야 한다.
4. **SDK 지원 개선:** 공급자에게 native MHA PrivateUse1 kernel 지원 또는 지원 여부를 고려한 dispatch 경로를 문의할 직접 재현 근거가 마련됐다. SDK·torch·드라이버·커널은 수정하지 않았다.

**판단: 조건부 진행.** fallback 회피 방법과 소수 입력 출력 일치성은 확보했으나, 출력 허용 오차·전체 연산 배치·실제 codegen/cache 관측과 단일 trial 한계가 남는다. 먼저 BF16 출력 허용 기준을 합의하고, 선택한 실행 경로의 독립 trial을 확대한다. B8/B16과 세부 모듈 벤치마크는 이번에 실행하지 않았다. 첫 피드백 질문은 ① SDK native MHA 지원/dispatch 개선 가능 여부 ② BF16 출력 atol/rtol 기준의 두 가지다.

## 원본과 재현

- Control/직접 compile 실패: `/root/ViT/vit_pilot/attention_runs/20261104T153417Z-33d046`
- 사용자 경계 compile: `/root/ViT/vit_pilot/attention_runs/20261104T160225Z-35e28e`
- 실제 입력: `/root/ViT/data/vit_real_images_4/images/`, 재사용 텐서: `/root/ViT/vit_pilot/followup_runs/20261104T132538Z-c031a6/inputs/preprocessed_bf16.pt`
- 누적 worker 시간 2320.4초 / 7200초. 개별 상한 1800초. 모든 원본·성공 결과·실패 로그와 기존 프로젝트 파일을 보존했다.

[원인 상세](../20261104T153417Z-33d046/mechanism.md), [control 보고서](../20261104T153417Z-33d046/report.md), [새 경계 코드](../../attention_boundary_worker.py), [재현 문서](../../ATTENTION_README.md). [PyTorch MHA 문서](https://docs.pytorch.org/docs/2.10/generated/torch.nn.MultiheadAttention.html), [MHA backend API](https://docs.pytorch.org/docs/2.10/backends.html#torch-backends-mha).
