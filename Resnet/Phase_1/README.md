# Phase 1 실험 절차

목적은 **동일한 ResNet50과 입력에서 RNGD Eager / Compile의 초기 비용, 반복 성능, Shape 전환, 캐시 재사용 및 출력 일치도를 비교**하는 것이다. 기본 설정은 원본 모델이고, 사용자 요청으로 추가한 folding 설정은 양 모드에 같은 사전 변환을 적용한다. 실제 실행 현황은 [STATUS.md](STATUS.md)에 둔다.

## 기획안 반영

| 요구 | 구현 |
| --- | --- |
| LLM에서 ResNet50으로 변경 | torchvision ResNet50 V1.5, `IMAGENET1K_V2`, BF16, eval / inference_mode |
| 두 모드 초기 비용 분리 | CPU 모델 로드, NPU 전송, mode wrapper, 첫 E2E, 워밍업, 반복값 각각 기록 |
| Fixed Shape | 입력 `[B,3,224,224]`, B = 1·4·8·16 |
| Mixed Shapes | 배치 순서 `1→4→1→8→4→16→1`; 동일 callable·프로세스 유지 |
| Cold / Warm Cache | 모드·조건별 새 전용 캐시, 그 경로를 쓰는 새 프로세스의 재사용 시도 |
| RNGD Eager 내부 준비 비용 | 연산별 export / compile / load LRU 진입·hit·miss 관측 |
| 재컴파일 관찰 | PyTorch backend 진입 횟수·증분 및 별도 SDK/guard 로그 |
| E2E 및 모듈 분석 | E2E + 별도 forward 시간, 별도 프로파일 프로세스, 원본 `conv1` 진단 |
| 수치 검증 | max / mean absolute error, RMSE, cosine, 분류 출력일 때만 class agreement |
| 기본 Compile 설정 | `torch.compile(model, backend=...)`; 래퍼는 입력을 그대로 `ft.backend`에 전달 |

배치 그리드는 기존안의 1·4·8·16을 유지하고 공간 크기는 224로 고정했다. 이 단계의 Shape 변화는 배치 차원을 뜻한다. LLM 전용 Prefill·Decode·토큰 처리량 지표를 제거하고 이미지 처리량을 기록한다. Synthetic normal 입력을 사용하며 이미지 로드나 전처리는 측정하지 않는다. 설정 원본은 [config.json](config.json)이다.

## 측정 경계

`e2e`는 **H2D 입력 전송 → forward → stream 완료 대기 → D2H 출력 전송 → 완료 대기**를 포함한다. 입력 생성, 모델 생성·전송, 수치 검증, CSV 기록은 이 구간 밖이다. 첫 E2E에는 lazy tracing, export, cache 탐색, 컴파일, 로드 및 첫 추론이 섞일 수 있으므로 이를 순수 컴파일 시간이라고 부르지 않는다.

`forward`는 NPU 상주 입력에 대해 `perf_counter_ns()`로 forward 및 stream 대기를 잰 별도 host wall-clock 값이다. 순수 NPU kernel 시간은 아니다. 모든 시간 측정 직전과 완료 시 `ft.current_stream(device).synchronize()`를 호출한다. 기본 단일 stream 경로를 사용한다.

각 고정 조건은 첫 호출 1회, 추가 워밍업 3회, 본 측정 10회, 별도 forward 측정 10회로 구성한다. Mixed는 최초 순서 1주기, 워밍업 3주기, 본 측정 10주기이며 모든 호출 순서와 누적 시간을 남긴다. Median / Min / Max / Mean / 표준편차 / CV를 함께 기록한다. 3회 워밍업만으로 수렴을 보장하지 않는다.

성능 측정 프로세스에서는 상세 로깅과 NPU profiling을 끈다. 프로파일링과 관측 로그는 별도 실행에 저장하며 성능 요약에서 제외한다. 두 모드를 순차 실행하고, 고정 배치 조건마다 모드 순서를 교차한다. cold와 reuse는 각각 독립 프로세스이므로 프로세스 내 LRU 및 Dynamo 캐시가 다시 시작한다.

`empty_project_cache`는 해당 실행의 전용 디렉터리가 비어 있다는 뜻이다. 공유 캐시나 시스템·장치 전체를 초기화하지 않는다. `disk_reuse_attempt`는 같은 경로를 다시 지정했다는 뜻이며, 시간 단축·파일 변화·backend 진입만으로 실제 disk hit 또는 compiler codegen 횟수를 확정하지 않는다. `.cache_info()`는 설치 버전에 종속된 읽기 전용 내부 API이며, 캐시 크기는 바꾸지 않는다.

CPU fallback을 명시적으로 끄고, PyTorch compiler 오류도 숨기지 않는다. SDK 기본 compiler config를 기록한다. Dynamic Compile 옵션, 연산 분해 재정의, 커널 수정은 적용하지 않는다.

추가 설정 [config-bn-folded.json](config-bn-folded.json)의 `bn_folded=true`는 공식 FP32 pretrained 모델의 Conv–BN 53쌍을 eval 상태에서 결합한 뒤 BF16으로 변환한다. Stem, 16개 Bottleneck, 4개 downsample의 BN을 모두 Identity로 바꾼다. Eager / Compile 양쪽의 변환 목록·정밀도·folded state SHA256가 같아야 결과를 비교한다. Folding은 모델 준비 시간에 포함하며 순수 compiler 시간과 구별한다. 원본 설정과 이전 결과는 유지한다. Folded 전체 그래프의 빌드를 고려해 이 설정의 조건별 timeout은 900초다.

본실험 설정 [config-bn-folded-split128.json](config-bn-folded-split128.json)은 53쌍 folding 후 FP32에서 `layer4.0/1/2.conv2`의 출력 채널을 128개씩 나누고 BF16으로 변환한다. 각 분기에 입력 512채널 전체를 전달하며, 가중치와 bias의 출력 축만 나눈다. 변형명은 `bn_folded_output_split128`이고 원시값·통계·수치 비교 표에 기록한다. CPU 배치 1·4의 전체 모델 비교에서 FP32와 BF16 모두 folded 모델과 출력 오차 0을 확인했다. [CPU 변환 검증](evidence/full-model-split128-validation/validation.json).

이 설정은 fixed 조건당 1,800초, Mixed 조건당 7,200초의 timeout을 둔다. 이는 종료 한도이며 예상 소요 시간은 아니다. 실패 결과도 `output.bin`에 저장될 수 있어 reuse는 직전 cold job이 실제 성공한 경우만 진행한다. 각 worker는 run-dir의 `source/Resnet` 소스 사본을 사용한다. 장시간 작업의 현재 단계·PID·성공/실패는 30초마다 갱신되는 `PROGRESS.md`와 `manifest.json`에서 확인한다. 성공한 job의 결과 표는 실행 중에도 갱신된다.

CPU FP32 변환 보존은 `rtol=1e-4, atol=1e-4`로 점검하고, 원본 BF16과 folded BF16의 차이는 별도 수치로 기록한다. 원본 BF16과 folded BF16의 비트 단위 일치는 요구하지 않는다. [변환 검증](evidence/bn-folding-validation.json), [공식 folding API](https://docs.pytorch.org/docs/2.10/generated/torch.nn.utils.fuse_conv_bn_eval.html).

## 결과 파일

각 run-dir에는 다음이 생성된다.

| 파일 / 폴더 | 내용 |
| --- | --- |
| `manifest.json` | 전체 완료·실패·미측정 상태, 실행 명령, 모드 순서, 실행 시간, 환경 override |
| `config.json`, `source_hashes.json` | 실행 설정과 시작 시점 코드 해시 |
| `evidence/` | 버전, 장치 시작·종료 상태, SDK 소스 |
| `jobs/*/result.json` | 조건별 상태, 실패 단계·traceback, 준비 시간, 가중치 SHA256, 캐시 파일 목록 |
| `jobs/*/samples.csv` | 첫 호출·워밍업·본 측정·forward 원시값 |
| `jobs/*/observations.json` | 각 호출 전후 Eager LRU 및 backend 진입 횟수 |
| `jobs/*.log` | 해당 프로세스 전체 로그 |
| `summary.csv` | 성공한 job의 본 측정 통계만 포함 |
| `failures.csv` | 실패와 중단 원인 |
| `cumulative.csv`, `workload_totals.csv` | 실제 측정 순서의 E2E 누적값과 전체 합계 |
| `correctness.csv`, `correctness.json` | 동일 입력·가중치·설정의 모드 간 수치 비교 |
| `amortization_estimates.csv` | 고정 조건의 초기 비용 추정 및 손익분기 경계 |
| `profiles/` | 별도 계측 프로세스, Chrome trace |
| `trace_spans.csv`, `trace_notes.json` | 별도 trace 요약 실행으로 생성하는 span 목록 |

초기 비용 추정은 `모델 로드 + 모델 전송 + wrapper + max(0, 최초/워밍업 E2E 합 − 해당 호출 수 × 반복 Median)`이다. 분리된 compiler codegen 시간을 얻었다는 의미가 아니다. 이 추정치로 기존안의 손익분기식을 계산하되, 올림 경계와 **Compile이 엄격히 빨라지는 첫 정수 호출 수**를 구분한다. 동률에서는 기존안대로 Eager를 택한다. 반복 이득이 없으면 별도 상태를 기록한다. Mixed는 Shape 빈도가 반영된 실제 누적 측정값으로 비교한다.

BF16 허용 오차의 합격 기준은 기획안에 없으므로 수치만 보고한다. 데이터셋 정확도 또는 정확도 통과 판정을 만들지 않는다. 독립 Conv 결과를 전체 ResNet50 성능으로 환산하거나 합산하지 않는다. Eager module hook trace도 별도 계측이며 Compile fusion 경계와 동일하다고 가정하지 않는다.

## 실행

```bash
cd /root
# 원본 모델: 최소 조건을 통과한 경우 전체 grid와 별도 profiling 실행
/root/venv/bin/python -m Resnet.Phase_1.run \
  --run-dir Resnet/Phase_1/results/run-002 --profiles

# BN-folded 전체 모델: 양 모드 최소 실행 통과 시 같은 본실험 grid 진행
/root/venv/bin/python -m Resnet.Phase_1.run \
  --config Resnet/Phase_1/config-bn-folded.json \
  --run-dir Resnet/Phase_1/results/folded-run-002 --profiles

# BN-folded + output-split128 본실험
/root/venv/bin/python -m Resnet.Phase_1.run \
  --config Resnet/Phase_1/config-bn-folded-split128.json \
  --run-dir Resnet/Phase_1/results/split128-run-002 --profiles

# 별도 진단: 원본 conv1만 측정. 전체 ResNet50의 대체 결과가 아니다.
/root/venv/bin/python -m Resnet.Phase_1.run --target conv1 \
  --run-dir Resnet/Phase_1/results/conv1-002 --profiles

# 저장된 원시값에서 통계·오차·누적 시간 다시 생성
/root/venv/bin/python -m Resnet.Phase_1.aggregate \
  Resnet/Phase_1/results/run-002

# 완료된 진단에서 별도 재컴파일 / guard 로그와 trace 요약 생성
/root/venv/bin/python -m Resnet.Phase_1.observe \
  Resnet/Phase_1/results/conv1-002
/root/venv/bin/python -m Resnet.profiling \
  Resnet/Phase_1/results/conv1-002

# Folded 전체 모델에서 실패한 layer4.0.conv2만 새 캐시로 재현
FURIOSA_TORCH_LOG=error FURIOSA_TORCH_PROFILE=error \
  /root/venv/bin/python -m Resnet.Phase_1.reproduce_conv \
  --run-dir Resnet/Phase_1/results/conv-repro-002
```

조건별 timeout은 기본 180초다. 초과 시 프로세스 그룹을 종료하고 timeout으로 기록한다. 기존 run-dir 덮어쓰기와 기존 캐시 삭제는 거부한다. 원본 모델 지원이 실패하면 나머지 조건을 `manifest.json`의 미측정 항목으로 남기며 성능·오차·손익분기를 생성하지 않는다.
