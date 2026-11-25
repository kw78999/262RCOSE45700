# RNGD NPU에서 ResNet50 Eager / Compile 성능 비교 — 최종 보고서

작성일: 2026-11-04 KST  
문서 상태: 실험 진행 중 초안. 현재는 **1. 실험 환경 및 모델 정보**, **2. 오류와 해결 과정**만 작성했다. 성능 결과 표와 최종 분석·인사이트는 실험 종료 후 추가한다.

본 실험은 Furiosa RNGD에서 동일한 ResNet50 모델을 Eager와 `torch.compile`로 실행하고, 초기 준비 비용·반복 추론·배치 변화·캐시 재사용 조건을 비교한다. 원본 모델의 실행을 가로막는 오류를 우회하기 위해 양 모드에 동일한 BN folding과 출력 채널 분할을 적용했다. 최종 비교 대상의 변형명은 `bn_folded_output_split128`이다.

## 1. 실험 환경 및 모델 정보

### 1.1 하드웨어 및 실행 장치

| 항목 | 설정 및 확인 내용 |
| --- | --- |
| NPU | Furiosa RNGD, `furiosa-smi`의 물리 장치명 `npu1` |
| 장치 메모리 | `furiosa-smi` 표시 기준 47.50 GiB |
| PyTorch 실행 장치 | `furiosa:0` |
| 장치 구성 | `Fusion(num_fusion=1, num_pe_per_cluster=1, num_chips=1, num_clusters_per_chip=1)`, `Single` |
| 사용 범위 | 기존 Conv 진단 trace에서 `npu1pe0` 작업 확인. 카드 전체 PE를 결합하는 구성으로 측정하지 않음 |
| 드라이버 | 2026.4.0, revision `32ffd4e` |
| 펌웨어 | 2026.4.0, revision `93d67f3` |
| OS / 커널 | Linux, `6.8.0-124-generic`, x86_64, glibc 2.39 |
| CPU 스레드 설정 | `torch.set_num_threads(1)`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` |
| 실행 방식 | 동일 장치에서 조건별 직렬 실행. 각 조건은 새 프로세스에서 수행 |

`furiosa:0`은 SDK가 노출하는 논리 장치명이다. 환경 기록에는 `furiosa:0`부터 `furiosa:7`까지 표시되지만, 이를 물리 RNGD 카드 8개로 해석하지 않는다. `furiosa-smi`의 `Utilization`은 `-`로 표시되어 이 값으로 연산 사용률을 수치화하지 않았다.

근거: [현재 실행 환경](results/bn-folded-split128-2h-20261104-162935/evidence/environment.json), [기존 Conv 진단 상태](STATUS.md#conv1-별도-진단).

### 1.2 소프트웨어 환경

| 구성 요소 | 버전 / 설정 |
| --- | --- |
| Python | 3.12.3 |
| 실행 인터프리터 | `/root/venv/bin/python` |
| PyTorch | `2.10.0+cpu` |
| torchvision | `0.25.0+cpu` |
| furiosa-torch | `2026.4.0`, revision `5800e2ba24` |
| furiosa-kernels | `2026.4.0` |
| NumPy / Pillow | `2.5.4` / `13.0.0` |
| Eager 진입점 | NPU로 이동한 모델을 직접 호출 |
| Compile 진입점 | `torch.compile(model, backend=counter)`; counter는 호출 횟수만 세고 `ft.backend`에 그대로 위임 |
| CPU fallback | worker에서 `ft.config.set_cpu_fallback(False)`로 명시적으로 비활성화 |
| 오류 처리 | `torch._dynamo.config.suppress_errors=False`; 컴파일 실패를 숨기지 않음 |
| Compiler 설정 | 설치 SDK 기본값 유지. `dynamic`, `fullgraph`, `mode` 등의 추가 Compile 옵션을 지정하지 않음 |
| Dynamic shapes 관측 | SDK import 후 `torch._dynamo.config.automatic_dynamic_shapes=False` |

PyTorch 버전의 `+cpu`는 설치 빌드 표기다. 실제 모델·입력은 Furiosa backend의 `furiosa:0`으로 이동하고, CPU fallback을 끈 상태에서 실행한다. SDK의 fallback 기본값은 True지만 본 실험의 worker 설정은 False다. CPU에서 수행하는 모델 준비와 결과 비교는 NPU 연산의 CPU fallback과 구분한다.

Eager 모드도 처음 사용하는 NPU 연산을 SDK 내부에서 컴파일·로딩한다. 따라서 Eager의 첫 호출에도 긴 초기 준비 시간이 포함될 수 있다. 코드의 backend counter는 TorchDynamo backend 진입 횟수이며, SDK 내부 전체 compiler 호출 횟수나 캐시 적중 횟수가 아니다.

근거: [환경 기록](results/bn-folded-split128-2h-20261104-162935/evidence/environment.json), [공통 실행·측정 코드](../runtime.py), [worker 소스 사본](results/bn-folded-split128-2h-20261104-162935/source/Resnet/Phase_1/worker.py).

### 1.3 모델과 입력

| 항목 | 설정 |
| --- | --- |
| 기본 모델 | `torchvision.models.resnet50` |
| 가중치 | `ResNet50_Weights.IMAGENET1K_V2` |
| 체크포인트 | `resnet50-11ad3fa6.pth` |
| 추론 상태 | `eval()` 및 `torch.inference_mode()` |
| 정밀도 | 모델·입력 BF16 (`torch.bfloat16`) |
| 변환 순서 | FP32 가중치 로드 → Conv–BN folding → layer4 Conv 출력 분할 → BF16 변환 |
| BN folding 범위 | Stem·Bottleneck·downsample의 Conv–BN 53쌍, BN은 Identity로 대체 |
| 출력 분할 범위 | `layer4.0.conv2`, `layer4.1.conv2`, `layer4.2.conv2` |
| 분할 방식 | 출력 512채널을 128채널씩 4개 Conv로 분리하고 채널 축으로 concat |
| 입력 | `[B, 3, 224, 224]`, seed로 고정한 정규분포 합성 입력 |
| seed | 2026. 배치 크기·입력 크기에 따라 결정되는 동일 입력을 양 모드에 사용 |
| 출력 | `[B, 1000]`, 유한한 BF16 텐서인지 확인 |
| 최종 변형명 | `bn_folded_output_split128` |

공식 원본 체크포인트 SHA256:

```text
11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca
```

BN folding과 출력 분할 후 BF16 모델 state SHA256:

```text
f4ffc8727491492e195b9c024f145293f422696cdbeed6d8c064016158753469
```

모드 간 비교 시 설정·가중치·모델 변환 정보를 확인한다. 합성 입력으로 지연시간과 출력 수치 차이를 측정하며, ImageNet 검증 데이터셋에 대한 정확도 평가는 수행하지 않는다.

근거: [모델 준비 코드](../models.py), [CPU 변환 검증](evidence/full-model-split128-validation/validation.json), [현재 고정 설정](results/bn-folded-split128-2h-20261104-162935/config.json).

### 1.4 실험 조건 및 측정 범위

아래 표는 **계획한 조건**이다. 실행 완료 결과와 구분한다.

| 구분 | 조건 조합 | 실행 수 |
| --- | --- | ---: |
| 고정 배치 성능 측정 | 배치 1·4·8·16 × Eager/Compile × 새 캐시/재사용 캐시 | 16 |
| 혼합 배치 성능 측정 | `[1,4,1,8,4,16,1]` 순서 × Eager/Compile × 새 캐시/재사용 캐시 | 4 |
| 별도 프로파일링 | 배치 1, 재사용 캐시 × Eager/Compile | 2 |
| 합계 | 성능 측정 20개 + 별도 프로파일링 2개 | 22 |

고정 배치 조건에서는 첫 호출 1회, 워밍업 3회, E2E 본 측정 10회와 별도 forward 측정 10회를 기록한다. 혼합 배치는 7개 배치 순서를 한 사이클로 하여 첫 사이클 1회, 워밍업 3사이클, 본 측정 10사이클을 수행한다. 별도 프로파일링 값은 성능 통계에 섞지 않는다.

| 측정 범위 | 포함 내용 |
| --- | --- |
| E2E | CPU 입력의 H2D 전송 + 모델 forward + stream 완료 대기 + 출력 D2H 전송 및 완료 대기 |
| Forward | 입력이 NPU에 이미 있는 상태에서 forward + stream 완료 대기. 호스트 경과 시간이며 순수 커널 시간이 아님 |
| 초기 호출 | 지연 컴파일·모듈 로딩·첫 실행·전송 등이 포함될 수 있음. 순수 compiler codegen 시간으로 해석하지 않음 |
| 모델 준비 | 가중치 로드·folding·분할·모델 전송·wrapper 준비 시간을 별도로 기록 |

동기화에는 `ft.current_stream(device).synchronize()`를 사용한다. 합성 입력 생성은 E2E 구간 밖에서 수행한다. 통계는 성공한 성능 job의 본 측정만 사용하고, 첫 호출·워밍업·실패한 부분 샘플은 분리한다.

Cold는 **해당 실험의 전용 compiler 캐시 디렉터리가 빈 상태**를 의미한다. 장치·OS·공유 시스템 캐시 전체의 초기화를 뜻하지 않는다. Reuse는 성공한 cold job의 같은 디스크 경로를 새 프로세스에서 다시 지정하는 조건이다. 실제 disk hit 여부는 별도 증거 없이 단정하지 않는다.

### 1.5 현재 실행의 재현 정보와 시간 제한

이 절의 표는 전체 모델 grid 실행의 재현 정보다. 해당 실행은 이후 사용자 요청으로 중단했고, 배치 1의 네 케이스를 대상으로 한 독립 모듈별 세부 계측으로 전환했다. 최신 계측 범위·해석은 [MODULE_LATENCY.md](MODULE_LATENCY.md), 현재 실행은 [STATUS.md](STATUS.md)를 따른다.

| 항목 | 값 |
| --- | --- |
| 실행 폴더 | `results/bn-folded-split128-2h-20261104-162935/` |
| 설정 파일 | [config-bn-folded-split128-2h.json](config-bn-folded-split128-2h.json) |
| Supervisor 시작 | 2026-11-04 16:29:35 KST |
| 전체 실행 상한 | 7,200초, 종료 제한 2026-11-04 18:29:35 KST |
| 정리 시간 | 제한 10초 전부터 runner와 하위 프로세스 종료 수행 |
| 조건별 timeout | 고정 배치 1,800초, 혼합 배치 7,200초. 전체 실행 상한이 우선 적용됨 |
| 작업 디렉터리 | `/root/Resnet/` 내부 |
| 가중치 / 임시 경로 | `.cache/torch/` / `.tmp/` |
| Compiler 캐시 | 실행 폴더의 `caches/<mode>-<condition>/` |
| 재현 자료 | 고정된 `config.json`, worker용 `source/Resnet/` 사본, `source_hashes.json`, 환경 기록 |

배치 1의 Eager·Compile 전체 실행을 먼저 확인한 후 나머지 조건을 수행한다. 시간 제한에 도달하면 완료된 결과를 보존하고 실행 중인 조건을 `time_budget_exceeded`로 기록한다. 22개 실행 모두의 완료를 보장하는 설정은 아니다.

진행 상황은 [PROGRESS.md](results/bn-folded-split128-2h-20261104-162935/PROGRESS.md), 시간 제한은 [budget 기록](results/bn-folded-split128-2h-20261104-162935-budget.json), 실행 시작 정보는 [launch 기록](evidence/latest-2h-launch.json)에서 확인한다.

## 2. 실험 과정의 오류와 해결 방법

### 2.1 BatchNorm 연산 지원 문제

#### 증상 및 원인

원본 ResNet50을 BF16, `eval()`, CPU fallback=False로 실행했을 때 두 모드 모두 BatchNorm에서 중단됐다.

| 모드 | 실패 내용 | 확인된 원인 |
| --- | --- | --- |
| Eager | `bn1`에서 `CPU fallback is disabled for aten::native_batch_norm` | 해당 SDK 경로에서 BatchNorm을 NPU로 처리하지 못하고 CPU fallback을 요구 |
| Compile | `aten::_native_batch_norm_legit_no_training`의 DFG import 실패, `UnsupportedOpError` | 그래프를 compiler 내부 표현으로 가져오는 단계에서 해당 연산을 처리하지 못함 |

설치 SDK의 지원 조회에서도 두 BatchNorm 연산은 `is_importable=False`, `is_supported_aten=False`였다. 이 문제는 학습 모드를 켰기 때문에 발생한 것이 아니다. 추론 상태에서도 BN 연산 자체가 그래프에 남아 지원되지 않는 경로로 진입했다.

근거: [Eager 오류](results/baseline-20261104/jobs/eager-b1-cold/result.json), [Compile 오류](results/baseline-20261104/jobs/compile-b1-cold/result.json), [지원 조회](evidence/sdk-operator-support.json).

#### 적용한 해결 방법: Conv–BN folding

추론 상태의 BatchNorm은 고정된 running mean·variance를 사용하므로 바로 앞 Conv의 가중치와 bias에 흡수할 수 있다. 출력 채널별로 다음과 같이 변환한다.

```text
s  = gamma / sqrt(running_var + eps)
W' = s * W
b' = s * (b - running_mean) + beta

BN(Conv(x; W, b)) = Conv(x; W', b')
```

원래 Conv에 bias가 없으면 `b=0`으로 처리한다. 수식은 실수 연산에서 동등하며, 부동소수점 연산 순서 변화로 수치 차이가 생길 수 있다.

공통 모듈 [folding.py](../folding.py)에서 `torch.nn.utils.fuse_conv_bn_eval`로 53쌍을 결합했다. 모든 모듈이 eval 상태이고 FP32인 것을 확인한 뒤 folding하고, BN을 Identity로 대체했다. 미결합 BN이 남으면 실패하도록 했다. 그 후 BF16으로 변환하여 양 모드에 같은 변환 모델을 사용한다.

#### 변환 검증과 해결 범위

아래는 합성 배치 1의 **CPU 수치 검증**이며 성능 결과가 아니다.

| 비교 | 최대 절대오차 | 예측 클래스 일치율 |
| --- | ---: | ---: |
| FP32 원본 ↔ FP32 folded | `1.609325e-6` | 1.0 |
| BF16 원본 ↔ BF16 folded | `0.06689453125` | 1.0 |

FP32 비교는 `rtol=1e-4`, `atol=1e-4` 검증을 통과했다. BF16 결과는 원본과 비트 단위로 동일하지 않으며, 이 표만으로 데이터셋 정확도 보존을 주장하지 않는다.

Folding 후 기존 BatchNorm 오류를 지나 다음 Conv 오류까지 진행했다. 이는 BN 문제의 우회가 작동했음을 보여주지만, folding만으로 전체 모델 실행이 가능해진 것은 아니다.

근거: [BN folding 검증](evidence/bn-folding-validation.json), [folded 모델 실행 이력](results/bn-folded-20261104/manifest.json).

### 2.2 `layer4.0.conv2`의 `LowLevelTailCrop` 오류

#### 증상과 재현 조건

BN을 제거한 후 Eager 전체 실행은 `layer4.0.conv2`에서 실패했다. 해당 지점의 실제 CPU activation을 사용한 단독 실행에서도 같은 오류를 재현했다.

| 항목 | 값 |
| --- | --- |
| 입력 | `[1,512,14,14]` |
| Conv 가중치 | `[512,512,3,3]`, BF16 |
| Bias | `[512]`; BN folding으로 생성 |
| Stride / Padding / Groups | 2 / 1 / 1 |
| 실패 단계 | BF16 `TransposedLower` → `LowLevelTailCrop` 검증 |
| 오류 | `UnsupportedOpError`, crop 입력 slice 바이트 수가 `u16::MAX`를 초과 |

Compile 전체 그래프에서도 같은 `[512,512,3,3]` 가중치의 lowering 오류를 확인했다. 다만 전체 그래프의 오류가 가리키는 원본 module 경계를 하나로 확정하지는 않았으므로, Compile 실패를 반드시 `layer4.0.conv2` 한 곳으로만 한정하지 않는다.

근거: [folded Eager 오류](results/bn-folded-20261104/jobs/eager-b1-cold/result.json), [folded Compile 오류](results/bn-folded-20261104/jobs/compile-b1-cold/result.json), [실제 activation 단독 재현](results/bn-folded-conv-repro-20261104/result.json).

#### 원인: 전체 가중치 용량보다 crop 이전 slice 크기가 중요

Compiler 오류에 기록된 검증 조건은 다음과 같다.

```text
element_type.size_in_bytes_with_length(input_shape.slice_volume())
    <= u16::MAX   # 65,535 bytes
```

실패한 layout에서 crop 전 입력 slice는 다음 크기였다.

```text
32 × 16 × 4 × 3 × 8 × 2 bytes = 98,304 bytes
98,304 > 65,535 → 검증 실패
```

64개 partition 위치를 합친 중간 텐서는 6,291,456 B다. Crop 후에는 slice당 49,152 B가 되지만 검증 대상은 **crop 이전 입력**이므로 오류가 발생한다. 원래 가중치의 3 크기 축이 중간 표현에서 4로 정렬되는 것이 오류 layout에 관측된다.

`layer4.0.conv1` 역시 큰 가중치를 사용하지만 다음 차이가 있다.

| 비교 항목 | `layer4.0.conv1` | `layer4.0.conv2` |
| --- | ---: | ---: |
| 가중치 shape | `[512,1024,1,1]` | `[512,512,3,3]` |
| 논리 BF16 가중치 용량 | 1,048,576 B (1 MiB) | 4,718,592 B (4.5 MiB) |
| 단독 진단 결과 | 성공 | 실패 |
| 관측된 TailCrop 입력 slice | 8,960 B | 98,304 B |
| TailCrop의 해당 대상 | Activation 변환 | 가중치 재배치 |
| 65,535 B 제약 | 충족 | 초과 |

Conv1에도 TailCrop이 존재하지만 해당 입력 slice가 작아 통과했다. 따라서 가중치 총량만으로 실패 여부를 판단할 수 없다. 또한 이 오류의 직접 원인은 compiler 내부 연산의 slice 크기 제약이며, NPU 전체 메모리가 부족하다는 증거로 해석하지 않는다.

근거: [postlower IR 및 slice 분석](results/conv-layout-review-20261104/lowering-comparison.json).

#### 검토한 우회 방법

아래 표는 본실험과 분리한 Conv 단독 진단이다.

| 시도 | Eager | Compile | 판단 |
| --- | --- | --- | --- |
| Stride를 2에서 1로 변경 | 같은 오류 | 미실행 | Stride=2만의 문제로 설명할 수 없음 |
| 출력 256채널씩 2개 Conv + concat | 같은 오류 | 같은 오류 | 이 분할로는 slice 한도를 만족시키지 못함 |
| `use_dma_bridges_only=True` | 같은 오류 | 미실행 | 해당 옵션만으로 우회되지 않음 |
| `ForVisionModel` tactic hint | 같은 오류 | 미실행 | 해당 hint만으로 우회되지 않음 |
| **출력 128채널씩 4개 Conv + concat** | **성공** | **성공** | 관측된 TailCrop 입력 slice의 최댓값 49,152 B |

SDK 옵션 변경은 원인 진단에만 사용했다. 현재 본실험은 SDK 기본 compiler 설정을 유지하고 모델의 출력 채널 분할을 적용한다.

근거: [진단 결과 8개](results/conv-layout-review-20261104/diagnostics.csv), [진단 코드](conv_layout_diagnostics.py).

#### 적용한 해결 방법: 출력 채널 128개 단위 분할

가중치의 출력 채널 축과 bias를 128개씩 나누고 각 Conv에 입력 512채널 전체를 전달한다. 출력은 원래 채널 순서대로 결합한다.

```text
원래 Conv:
  y = Conv(x, W[0:512], b[0:512])

분할 후:
  y0 = Conv(x, W[  0:128], b[  0:128])
  y1 = Conv(x, W[128:256], b[128:256])
  y2 = Conv(x, W[256:384], b[256:384])
  y3 = Conv(x, W[384:512], b[384:512])
  y  = cat([y0, y1, y2, y3], dim=1)
```

각 출력 채널의 convolution 계산은 독립적이므로 수학적으로 같은 연산을 구성한다. 입력 채널을 그룹별로 나누거나 분기 결과를 합산하는 방식은 사용하지 않았다. Stride·padding·dilation·bias 및 가중치의 채널 순서를 보존한다.

공통 모듈 [partitioning.py](../partitioning.py)에서 같은 가중치 shape를 사용하는 `layer4.0/1/2.conv2`에 적용했다. `layer4.0.conv2`는 stride 2, 나머지 두 Conv는 stride 1이다. Stride 1 단독 조건에서도 오류가 재현됐고 전체 Compile 오류의 module 경계가 확정되지 않은 점을 고려해 세 Conv를 동일하게 처리했다.

#### 해결 검증과 남은 확인 범위

| 검증 범위 | 비교 대상 | 확인 내용 |
| --- | --- | --- |
| 단독 Conv, CPU | 분할 전 ↔ 128분할 | FP32 동등성 검증 통과, BF16 최대 절대오차 0 |
| 단독 Conv, NPU | CPU 기준 ↔ NPU Eager / Compile | 양 모드 실행 성공, 최대 절대오차 각각 `0.00048828125` |
| 단독 Conv, NPU | Eager ↔ Compile | 최대 절대오차 `9.536743e-7` |
| 전체 ResNet50, CPU, 배치 1·4 | Folded 모델 ↔ folded+128분할 | 각 배치에서 FP32·BF16 모두 최대 절대오차 0, 클래스 일치율 1.0 |
| 현재 전체 ResNet50, NPU, 배치 1 | Eager / Compile 실행 지원 | 양 모드 전체 실행 성공, 동일 변환 state·설정 및 `[1,1000]` 유한 출력 확인; 최소 조건 통과 |

단독 Conv 검증 수치를 전체 모델 오차로 대체하지 않는다. 배치 1의 전체 모델 최소 조건 통과는 실행 지원과 기본 출력 검증을 의미하며, 모든 배치의 실행·성능 우열·데이터셋 정확도 합격을 뜻하지 않는다. 더 큰 배치와 혼합 배치의 완료 여부는 진행 중인 본실험에서 확인한다.

출력 분할은 Conv 노드 수와 concat 경계를 바꾸므로 성능 결과는 **변환한 모델의 Eager / Compile 비교**로 보고해야 한다. 원본 ResNet50 그대로의 성능이나 compiler 자체가 수정된 결과로 해석하지 않는다.

실패한 컴파일도 캐시에 `output.bin`을 남길 수 있었다. 캐시 파일 존재만으로 성공을 판단하지 않고, 성공한 cold job의 캐시만 reuse에 사용한다. 오류 우회 재검증에는 새 전용 캐시를 사용했다.

근거: [단독 Conv 분석](results/conv-layout-review-20261104/lowering-comparison.json), [전체 모델 CPU 검증](evidence/full-model-split128-validation/validation.json), [현재 NPU 최소 조건 검증](results/bn-folded-split128-2h-20261104-162935/minimum_condition.json).
