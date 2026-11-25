# 실측 상태 — 2026-11-04 KST

최종 보고서 초안: [FINAL_REPORT.md](FINAL_REPORT.md). 사용자 요청에 따라 환경·모델 정보와 BN/Conv 오류 해결 과정만 먼저 작성했다. 성능 결과 표와 최종 분석은 실험 종료 후 작성한다.

## 현재 실행: 배치 1 모듈별 세부 latency 4개 케이스

사용자 요청에 따라 기존 전체 모델 grid 실험을 종료하고, 2026-11-04 17:32:45 KST에 배치 1 모듈별 세부 측정을 새 캐시로 시작했다. **Eager cold / Eager reuse / Compile cold / Compile reuse**를 순차 진행한다. 기존 18:29:35 KST 제한을 연장하지 않고 남은 시간만 사용한다. 새 supervisor의 종료 제한은 2026-11-04T18:29:32.778651+09:00이다.

- **진행:** [PROGRESS.md](results/b1-module-latency-20261104-173245/PROGRESS.md), [manifest](results/b1-module-latency-20261104-173245/manifest.json), [시간 제한](results/b1-module-latency-20261104-173245-budget.json).
- **측정 결과:** [모듈별 통계](results/b1-module-latency-20261104-173245/module_summary.csv), [Eager/Compile 비교](results/b1-module-latency-20261104-173245/module_comparison.csv). 실행 중 완료된 지점부터 갱신한다.
- **범위:** 케이스당 FX node 187곳. Conv 62, ReLU 49, MaxPool 1, add 16, concat 3, AvgPool 1, flatten 1, FC 1의 후보 134곳과 folding된 BN Identity 53곳을 구분한다. Compile flatten은 단독 binary를 만들 수 없어 `metadata_only_no_binary`로 기록하며, Identity도 latency를 0으로 만들지 않는다.
- **방법:** 실제 CPU activation을 캡처한 독립 연산 실행, 입력 전송 후 동기화, 연산 실행 후 NPU 완료까지 계측. 첫 호출 1회·워밍업 3회·본 측정 10회. Compile은 node마다 따로 컴파일하므로 전체 모델 fusion 성능과 구분한다.
- **문서 및 설정:** [MODULE_LATENCY.md](MODULE_LATENCY.md), [config-b1-module-latency.json](config-b1-module-latency.json), [시작 기록](evidence/latest-b1-module-launch.json).
- **검증:** 테스트 26개 통과, 모든 연산 종류에 대해 NPU 사전 검증 완료. CSV 원자적 저장과 runner 비정상 종료 시 worker 정리 검증 포함.
- **프로세스:** supervisor 74995, runner 74996. 작업 경로·가중치·캐시·임시 파일은 `/root/Resnet/` 내부.

## 이전 전체 모델 본실험 — 2시간 제한 실행 (사용자 중단)


**2026-11-04T17:17:28.810463+09:00 사용자 요청으로 종료.** 성공한 5개 job의 결과와 진행 중 중단한 job의 로그를 보존했다.

2026-11-04 16:29:35 KST에 새 캐시와 새 결과 폴더로 재시작했다. **전체 종료 제한은 2026-11-04 18:29:35 KST (7200초)**다. 독립 supervisor가 전체 실행 시간을 감시하며, 제한 10초 전부터 runner와 별도 세션 worker/하위 프로세스를 종료한다. 먼저 완료되면 즉시 끝낸다. 제한 도달 시 완료된 결과는 보존하고 실행 중 조건은 `time_budget_exceeded`로 기록한다. 모든 조건의 완료를 보장하는 시간 예측이 아니라 전체 실행 상한이다.

- **최신 진행:** [PROGRESS.md](results/bn-folded-split128-2h-20261104-162935/PROGRESS.md), [manifest](results/bn-folded-split128-2h-20261104-162935/manifest.json).
- **시간 제한 감시:** [budget 기록](results/bn-folded-split128-2h-20261104-162935-budget.json), [실행 로그](results/bn-folded-split128-2h-20261104-162935-budget.console.log), [시작 기록](evidence/latest-2h-launch.json).
- **설정:** [config-bn-folded-split128-2h.json](config-bn-folded-split128-2h.json). BN folding 53쌍, layer4 conv2 출력 128채널 분할, BF16, Eager/Compile, 배치 1·4·8·16, mixed, cold/reuse 및 별도 profile을 기존 순서대로 수행한다.
- **작업 경로:** `/root/Resnet` 내부. 가중치는 `.cache/torch`, 임시 파일은 `.tmp`, compiler 캐시는 새 결과 폴더에 둔다. 설치 SDK 소스 파일을 별도로 읽어 복사하는 환경 수집은 비활성화했다.
- **재시작 검증:** 테스트 21개 통과. 별도 세션 worker를 포함한 시간 제한 종료와 정상 조기 종료를 검증했다. 내부 가중치 SHA256는 기존 CPU 검증의 `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`와 같다.
- **실행 PID:** supervisor 59766, runner 59767.

## 이전 BN-folded + output-split128 본실험 (사용자 중단)

**2026-11-04T13:19:21.706840+09:00 사용자 요청으로 실험 중단.** Resnet runner/worker 모두 종료했고 RNGD 사용 프로세스 없음, 메모리 0.00 GiB를 확인했다. Eager 배치 1 cold는 성공했으며 Compile 배치 1 cold는 진행 중 중단했다. 기존 측정 결과와 로그는 보존했다.

2026-11-04 13:05 KST에 독립 runner를 시작했다. 양 모드에 동일한 Conv–BN folding 53쌍과 `layer4.0/1/2.conv2` 출력 128채널 분할을 적용한다. 기본 compiler 설정과 CPU fallback=False를 유지한다. 배치 1의 전체 모델 최소 조건을 통과하면 고정 배치 1·4·8·16, Mixed Shapes, 새 캐시/재사용 캐시, 별도 profiling을 순차 진행한다.

- **이전 실행 상태:** [PROGRESS.md — 자동 갱신](results/bn-folded-split128-20261104/PROGRESS.md), [manifest](results/bn-folded-split128-20261104/manifest.json), [console 로그](evidence/bn-folded-split128-console.log).
- **설정:** [config-bn-folded-split128.json](config-bn-folded-split128.json). 표의 변형명은 `bn_folded_output_split128`이다.
- **전체 모델 CPU 검증:** 배치 1·4 모두 folded 모델 대비 FP32/BF16 출력 최대 절대오차 0, 예측 일치율 1.0. [검증 결과](evidence/full-model-split128-validation/validation.json). 데이터셋 정확도 평가는 아니다.
- **모델 상태 SHA256:** `f4ffc8727491492e195b9c024f145293f422696cdbeed6d8c064016158753469`.
- **준비 검증:** 테스트 19개 통과. [테스트 로그](evidence/tests-split128.log). worker 소스는 실행 폴더에 사본으로 고정했다.

성공한 job의 원시값과 통계는 실행 폴더에 누적한다. 아래 내용은 원본·BN-folded-only 모델의 실패와 단독 Conv 진단 이력이다.

**이전 BN-folded-only 실험은 두 모드 모두 BF16 Conv lowering 오류로 실패했다.** Conv–BN 53쌍을 FP32에서 결합한 뒤 BF16으로 변환했으며, Eager / Compile에 동일한 folded 모델을 사용했다. BatchNorm 연산은 모두 제거됐다.

- 설정: [config-bn-folded.json](config-bn-folded.json), 실행: [bn-folded manifest](results/bn-folded-20261104/manifest.json), [실행 로그](evidence/bn-folded-console.log).
- CPU FP32 변환 검증 통과: max absolute error `1.6093e-6`, class agreement 1.0. 원본 BF16과 folded BF16 차이는 max absolute error `0.06689`, cosine `0.9999227`, class agreement 1.0이다. 이 값은 synthetic batch 1에 대한 변환 점검이며 데이터셋 정확도는 아니다. [변환 검증](evidence/bn-folding-validation.json).
- Eager / Compile의 folded state SHA256는 모두 `73d260e221fc017b5acd7066fbdfe3948195ce25e2c03916ef6d045a0abc68bb`이다. CPU fallback=False와 기본 SDK compiler 설정을 유지했다.

| BN-folded 실행 | 결과 |
| --- | --- |
| Eager 전체 모델 | `layer4.0.conv2`에서 `UnsupportedOpError`: BF16 `TransposedLower` → `LowLevelTailCrop` 검증 실패 |
| Compile 전체 그래프 | `[512,512,3,3]` BF16 Conv 가중치의 `TransposedLower` → `LowLevelTailCrop` 검증 실패 |
| Eager 실패 Conv 단독 재현 | 실제 CPU activation으로 같은 오류 재현; input `[1,512,14,14]`, weight `[512,512,3,3]`, bias `[512]`, stride 2, padding 1 |
| 본실험 | 최소 조건 실패로 나머지 성능 조건 18개와 profiling 2개 미실행 |
| 전체 모델 유효 추론 샘플 / 모드 간 출력 비교 | 0 / 미측정 |

오류는 `input_shape.slice_volume()`에 대한 byte 크기가 `u16::MAX` 이하여야 한다는 compiler 내부 검증에서 발생한다. 가중치의 layout lowering 실패이며 BatchNorm fallback 오류와 구분한다. Compile 전체 그래프에서 보고된 정확한 원본 module 경계는 확정하지 않았다.

[Eager 실패](results/bn-folded-20261104/jobs/eager-b1-cold/result.json), [Compile 실패](results/bn-folded-20261104/jobs/compile-b1-cold/result.json), [단독 Conv 재현 결과](results/bn-folded-conv-repro-20261104/result.json), [단독 재현 스크립트](reproduce_conv.py), [실패 요약](results/bn-folded-20261104/failures.csv)에 근거를 보존했다. 실패한 첫 호출의 경과 시간을 추론 지연시간이나 순수 컴파일 비용으로 보고하지 않았다. 테스트 15개가 통과했다. [테스트 로그](evidence/tests-bn-folded.log).

현재 재개 조건은 위 BF16 Conv lowering 오류가 해결된 환경에서 folded 모델의 양 모드 전체 실행이 통과하는 것이다. `config-bn-folded.json`으로 새 run-dir을 지정하면 최소 조건 확인 후 본실험을 자동 진행한다. 아래 원본 모델과 Conv1 진단은 변환 전 실험 이력이다.

## Conv1 / Conv2 lowering 원인 검토

핵심은 **전체 가중치 크기가 아니라, 선택된 layout의 `LowLevelTailCrop` 입력 slice 바이트 수**다. 같은 SDK·BF16·CPU fallback=False로 Conv1을 단독 실행하고 성공한 postlower IR을 추출했다. Conv1에도 TailCrop은 존재하며, 그 입력 slice가 작아서 검증을 통과한다.

| 항목 | `layer4.0.conv1` 단독 성공 | `layer4.0.conv2` 실패 |
| --- | ---: | ---: |
| 가중치 shape | `[512,1024,1,1]` | `[512,512,3,3]` |
| 논리 BF16 가중치 | 1,048,576 B = 1 MiB | 4,718,592 B = 4.5 MiB |
| 관측된 weight SRAM 표현 | slice당 16,384 B | 실패한 weight 재배치의 crop 입력 slice당 98,304 B |
| 실제 TailCrop 입력 | slice당 8,960 B, activation 변환 | slice당 98,304 B, weight 재배치 |
| 검증 한도 | `<=65,535 B` 통과 | `>65,535 B` 실패 |

실패 layout의 slice 크기는 `32×16×4×3×8×2 = 98,304 B`다. 64개 partition 위치를 합친 중간 텐서는 6,291,456 B이며, crop 이후에는 slice당 49,152 B로 줄어든다. **검증 대상은 crop 이전 입력**이므로 출력이 작아져도 실패한다. 논리 가중치 4.5 MiB가 6 MiB의 중간 표현으로 확장되는 것은 IR에 기록된 끝축 3의 4 정렬과 일치한다. 설치 target의 SRAM slice 용량은 524,288 B이므로, 이 오류를 SRAM 전체 용량 부족으로 설명하지 않는다. [공식 SRAM 구조](https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/memory-performance.html).

아래는 본실험과 분리한 단독 진단 결과다. 새 전용 캐시를 썼고 기본 본실험 설정은 변경하지 않았다.

| 진단 | Eager | Compile | 의미 |
| --- | --- | --- | --- |
| Conv1 원형 | 성공 | 미실행 | 큰 가중치 자체가 실패 기준은 아님 |
| Conv2의 stride만 1로 변경 | 같은 오류 | 미실행 | stride=2만의 문제로 설명할 수 없음 |
| 출력 256채널씩 2개 Conv 후 concat | 같은 오류 | 같은 오류 | compiler 재배치 후 입력 slice가 여전히 98,304 B |
| `use_dma_bridges_only=True` | 같은 오류 | 미실행 | 이 옵션만으로 해당 weight 변환을 우회하지 못함 |
| `ForVisionModel` hint | 같은 오류 | 미실행 | 다른 tactic에서도 같은 slice 한도 초과 |
| **출력 128채널씩 4개 Conv 후 concat** | **성공** | **성공** | 관측된 TailCrop 입력 최대 49,152 B로 한도 내 |

128분할은 각 분기에 입력 512채널 전체를 전달하고 가중치·bias의 출력 축만 나눈다. 출력 채널별 계산이 독립적이므로 원래 Conv와 수학적으로 동일하다. 실제 activation에 대한 CPU FP32 동등성 검증을 통과했고 CPU BF16 분할 전후 최대 오차는 0이었다. NPU의 CPU 기준 최대 오차는 양 모드 모두 `0.00048828125`; NPU Eager / Compile 간 최대 오차는 `9.536743e-7`이었다. Synthetic batch 1의 해당 Conv에 대한 결과이며, 전체 모델·데이터셋 정확도는 미검증이다.

해결방안의 우선순위는 다음과 같다.

1. **기본 그래프를 유지하려면 compiler 수정이 우선이다.** `TransposedLower`가 crop 이전 slice 크기를 검사해 잘못된 tactic을 제외하거나, crop을 발행하기 전에 시간 축을 나눠야 한다. 예를 들어 위 layout의 32 크기 축을 16씩 두 조각으로 나누면 입력은 각 49,152 B로 한도 안에 든다. 구체적인 compiler 패치는 설치 환경에 Rust 구현 소스가 없어 검증하지 못했다. 이 수정이 포함된 SDK에서 최소 재현을 다시 통과시키는 절차가 필요하다.
2. **단독 Conv에서 검증된 모델 측 우회는 output-channel 128분할이다.** 이 검토 당시에는 본실험 모델에 적용하지 않았으며, 이후 사용자 요청으로 위의 새 본실험 설정에 연결했다. `layer4.0/1/2.conv2`에 같은 변환을 적용하고 양 모드의 전체 forward와 배치 1·4·8·16을 확인한다. 분할은 Conv 수와 concat 경계를 바꾸므로 `BN-folded + output-split128` 변형으로 구분한다.

실패 결과도 `output.bin`에 직렬화되어 있음을 확인했다. 파일 존재를 컴파일 성공 증거로 삼지 않으며, SDK나 workaround 재검증에는 새 전용 캐시를 사용한다. 단순 캐시 재시도만으로 위 layout 한도가 해결되지는 않는다.

근거: [8개 진단 결과](results/conv-layout-review-20261104/diagnostics.csv), [postlower slice 계산과 출력 비교](results/conv-layout-review-20261104/lowering-comparison.json), [실행 명령](results/conv-layout-review-20261104/manifest.json), [진단 스크립트](conv_layout_diagnostics.py).

**변환 전 원본 ResNet50의 Eager / Compile 전체 비교는 SDK의 BatchNorm 미지원으로 중단됐다.** 화면과 최종 보고서는 작성하지 않았다.

## 원본 모델

| 항목 | 실제 결과 |
| --- | --- |
| 환경 | Python 3.12.3, torch 2.10.0+cpu, torchvision 0.25.0+cpu, furiosa-torch / kernels / driver 2026.4.0 |
| 가중치 | `ResNet50_Weights.IMAGENET1K_V2`, SHA256 `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca` |
| 최소 입력 | `[1,3,224,224]`, BF16, eval / inference_mode, CPU fallback=False |
| Eager | `bn1`에서 `CPU fallback is disabled for aten::native_batch_norm` |
| Compile | `_native_batch_norm_legit_no_training`의 DFG import 실패, `UnsupportedOpError` |
| 전체 모델 지연시간·처리량·모드 간 오차·손익분기 | 미측정 |
| 나머지 배치·Mixed·캐시 비교·전체 모델 profiling | 최소 조건 실패로 진행하지 않음 |

[정식 실행 manifest](results/baseline-20261104/manifest.json), [실패 요약](results/baseline-20261104/failures.csv), [Eager traceback](results/baseline-20261104/jobs/eager-b1-cold/result.json), [Compile traceback](results/baseline-20261104/jobs/compile-b1-cold/result.json)에 근거를 보존했다. 상세 초기 관측 로그도 [Eager](evidence/preflight-eager.log), [Compile](evidence/preflight-compile.log)에 있다. 실패한 시도의 경과 시간을 모델 추론 성능으로 사용하지 않았다.

설치 SDK의 지원 조회에서도 두 BatchNorm 연산은 `is_importable=False`, `is_supported_aten=False`였다. `aten::conv2d` 자체의 조회는 False지만 SDK 기본 decomposition이 `aten::convolution`으로 변환하여 실제 실행은 성공했다. 지원 목록 조회만으로 모델 전체의 실행 가능성을 판단하지 않았다. [SDK 조회 및 장치 설정](evidence/sdk-operator-support.json).

동일 가중치·입력의 CPU BF16 기준 실행은 `[1,1000]`의 유한 출력을 반환했다. 이는 모델 구성·가중치의 실행 확인이며 성능 비교 결과는 아니다. [기준 실행](evidence/cpu-reference-check.json).

## Conv1 별도 진단

원본 모델의 `conv1`만 사용했다. 입력 `[B,3,224,224]`, 출력 `[B,64,112,112]`이며 **ResNet50 전체 성능으로 해석할 수 없다.** 기본 `furiosa:0`은 Fusion=1 설정이고, 실제 trace는 물리 `npu1pe0`의 Task를 기록했다. 카드 전체 PE를 결합한 성능을 측정한 것은 아니다.

고정 배치와 Mixed, 새 캐시와 재사용 시도 등 **20개 측정 job**, 별도 **2개 profiling job**이 성공했다. 성능 job에는 첫 호출·워밍업·본 측정·별도 forward를 합쳐 776개 원시 샘플이 있다. [진단 manifest](results/conv1-diagnostic-20261104/manifest.json).

아래는 **새 전용 캐시로 시작한 프로세스의 워밍업 이후 E2E 10회**, 단위 ms다. H2D와 D2H가 포함된다.

| Batch | Eager Median | Eager Min–Max | Compile Median | Compile Min–Max |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.719 | 0.708–0.745 | 0.663 | 0.639–0.717 |
| 4 | 1.430 | 1.360–3.277 | 1.263 | 1.192–1.448 |
| 8 | 4.052 | 2.170–6.007 | 2.085 | 1.955–5.764 |
| 16 | 3.511 | 3.409–14.257 | 3.387 | 3.190–13.887 |

일부 조건의 편차가 크다. 이 소규모 진단의 Median만으로 일반적인 우열이나 전체 모델 손익분기를 결론 내리지 않는다. 전체 통계·CV 및 재사용 조건 값은 [summary.csv](results/conv1-diagnostic-20261104/summary.csv), 실제 누적값은 [workload_totals.csv](results/conv1-diagnostic-20261104/workload_totals.csv)에 있다.

모드 간 최종 출력 16쌍의 max absolute error / mean absolute error / RMSE는 모두 0이었다. FP64 누적으로 계산한 cosine은 약 1이다. Conv feature map에는 class agreement를 적용하지 않았다. [수치 비교](results/conv1-diagnostic-20261104/correctness.csv).

Mixed 최초 주기의 Compile backend 누적 진입 수는 배치 순서 `1,4,1,8,4,16,1`에서 **`1,2,2,3,3,4,4`**였다. Eager compile LRU 누적 miss도 **`1,2,2,3,3,4,4`**였다. 이전 Shape로 복귀할 때 해당 진입·miss가 증가하지 않는 것을 관측했다. 별도 계측의 [Compile guard 로그](results/conv1-diagnostic-20261104/observability/compile.log)에는 배치 4·8·16 전환 시 실제 `Recompiling function`과 size mismatch가 기록됐다. 기본 SDK import 후 `automatic_dynamic_shapes=False`이며 직접 옵션을 변경하지 않았다.

새 프로세스에서 같은 디스크 경로를 쓰는 실험도 수행했다. 실제 disk hit / miss와 compiler codegen 횟수는 확정하지 않았다. [호출별 LRU 관측](results/conv1-diagnostic-20261104/jobs/eager-mixed-cold/observations.json), [관측 실행 manifest](results/conv1-diagnostic-20261104/observability/manifest.json).

별도 프로파일은 Eager / Compile 각각 NPU Task 1개를 기록했다. [Trace 요약](results/conv1-diagnostic-20261104/trace_notes.json), [span 목록](results/conv1-diagnostic-20261104/trace_spans.csv). 단독 Conv 실행으로 전체 모델의 fusion 효과나 병목을 확정하지 않았다.

## 검증 및 재개 조건

통계·손익분기 동률 처리·출력 수치·실패 샘플 제외 등 자동 테스트를 실행했다. [테스트 로그](evidence/tests.log).

변환하지 않은 원본 모델의 비교에는 위 두 BatchNorm 실행 경로의 지원이 필요하다. 사용자 요청으로 별도 추가한 BN-folded 실험은 위의 새 결과 폴더에서 진행한다. CPU fallback은 계속 비활성화한다. 변환 전 원본 전체 모델의 정확도 합격 판정과 손익분기는 산출하지 않았다.
