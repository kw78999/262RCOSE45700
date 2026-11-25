# 배치 1 모듈별 세부 latency 계측

## 목적과 비교 범위

기존 배치 1·4·8·16 전체 모델 실험을 사용자 요청으로 중단하고, **Eager cold / Eager reuse / Compile cold / Compile reuse** 네 케이스의 모듈별 세부 측정을 수행한다. 기존 성공 결과는 별도 실행 폴더에 보존한다.

전체 모델의 Compile fusion 결과에서 원래 모듈마다 시간을 분리하는 계측은 아니다. FP32 BN folding과 layer4 출력 분할을 적용한 BF16 모델을 CPU에서 실행해 실제 중간 입력을 캡처하고, 각 FX 연산 지점을 NPU에서 **독립 실행**한다. Compile도 각 지점을 따로 컴파일하므로 결과는 `isolated_operation_forward_with_device_synchronization`으로 표시한다. 동일한 CPU activation을 사용하므로 NPU 전체 파이프라인의 오차 누적을 측정하는 실험도 아니다.

## 계측 항목

| 종류 | 호출 지점 수 | 처리 |
| --- | ---: | --- |
| Conv2d | 62 | layer4 분할 분기 12개를 포함해 각각 측정 |
| ReLU | 49 | 공유 ReLU 모듈의 반복 호출을 FX node별로 구분 |
| MaxPool2d | 1 | 독립 실행 측정 |
| Residual add | 16 | 함수 연산도 별도 측정 |
| Concat | 3 | layer4 분할 결과 결합을 별도 측정 |
| AdaptiveAvgPool2d | 1 | 독립 실행 측정 |
| Flatten | 1 | Eager 측정. Compile은 단독 binary 생성 불가 시 `metadata_only_no_binary`로 기록 |
| Linear | 1 | FC 독립 실행 측정 |
| Identity | 53 | Folding된 BN. `identity_no_kernel`로 표시하고 latency를 만들지 않음 |
| 합계 | 187 | 후보 계측 134곳 + Identity 53곳 |

모든 연산 종류의 Eager/Compile NPU 사전 실행을 확인했다. Flatten의 단독 Compile에서는 `RunnableSequence requires at least one runnable with a binary`가 발생했다. 이는 독립 실행 가능한 binary가 없는 그래프에 대한 제한으로 기록하며, latency를 0으로 채우거나 임의의 추가 연산을 넣지 않는다. 다른 예외는 실패로 기록한다.

## 측정 방법

1. 기존과 동일한 pretrained 가중치·변형 SHA256를 검증한다. 모델은 `bn_folded_output_split128`, 입력은 배치 1의 `[1,3,224,224]`, BF16, seed 2026이다.
2. `torch.fx.Interpreter`로 각 연산의 입력과 CPU 기준 출력을 캡처한다. In-place ReLU의 입력이 바뀌지 않도록 **호출 전에 clone**한다. BN Identity와 함수 연산도 inventory에 남긴다.
3. 각 연산의 모듈·가중치를 NPU로 옮기고 매 호출마다 입력을 새로 전송한다. H2D 전송과 사전 동기화는 timer 밖에서 수행한다.
4. 연산 호출 직전부터 `ft.current_stream(device).synchronize()` 완료까지의 호스트 경과 시간을 측정한다. D2H 전송·수치 검증·파일 저장은 timer 밖에서 수행한다. 순수 NPU 커널 시간과는 구분한다.
5. 첫 호출 1회, 워밍업 3회, 본 측정 10회를 기록한다. 본 측정의 median·min·max·mean·표준편차·CV를 저장한다. 첫 호출에는 compiler·모듈 로딩 등의 비용이 들어갈 수 있다.
6. Compile은 각 연산 전에 TorchDynamo 상태를 초기화해 서로 다른 모듈이 공유 Python frame의 재컴파일 제한에 걸리지 않게 한다. SDK compiler 옵션은 기본값을 사용하고 CPU fallback은 False로 유지한다.

Cold는 **케이스 시작 시** 해당 모드의 전용 디스크 compiler 캐시가 비어 있다는 의미다. 같은 케이스 안에서는 연산 순서대로 실행하므로 동일 shape/연산이 이미 준비됐을 수 있다. 각 행을 독립적인 완전 cold compile이라고 해석하지 않는다. Reuse는 같은 캐시를 사용하는 새 worker 프로세스다. Cold에서 성공한 지점만 reuse하며, 실패 지점과 metadata-only 지점은 사유를 남긴다. 캐시 적중을 시간 단축만으로 단정하지 않는다.

독립 실행 시간이므로 모듈별 값을 합산해 기존 전체 모델 E2E나 Compile fusion 성능을 재구성하지 않는다. 모델 stage/Bottleneck container 전체 시간은 직접 측정하지 않으며, 내부 Conv·ReLU·add 등의 호출 지점을 구분한다.

## 결과 파일과 진행 확인

최신 경로는 [실행 시작 기록](evidence/latest-b1-module-launch.json)과 [STATUS.md](STATUS.md)에 기록한다.

| 파일 | 내용 |
| --- | --- |
| `PROGRESS.md`, `manifest.json` | 네 케이스의 진행 상태, 현재 node·module, 처리 개수, 종료 제한 |
| `jobs/<case>/module_inventory.json` | FX node, 원래 module 경로, operator, 입력·출력 shape, 소속 module stack |
| `jobs/<case>/samples.csv` | 각 지점의 첫 호출·워밍업·본 측정 원시값 |
| `jobs/<case>/modules.csv` | 지점별 통계, CPU 기준 최대 절대오차/RMSE, 성공·실패·미계측 사유 |
| `module_summary.csv` | 네 케이스의 모듈별 통계 모음. 완료된 지점은 실행 중에도 갱신 |
| `module_comparison.csv` | 같은 node·cache 조건의 Eager/Compile 비교. 양 모드 성공 시에만 속도비 계산 |
| `source/Resnet`, `source_hashes.json` | worker 소스 사본과 해시 |

CSV와 JSON은 임시 파일에 완전히 작성한 뒤 교체하여 진행 중 집계에서 일부만 기록된 행을 읽지 않도록 한다. 케이스 전체가 중단돼도 이미 성공적으로 측정한 독립 연산의 행은 보존한다.

기존 전체 실험의 **2026-11-04 18:29:35 KST 종료 제한을 연장하지 않는다.** 새 supervisor는 남은 시간만 사용하고, 시간 제한에 도달하면 runner와 worker를 종료한다. 작업 파일·가중치·캐시·임시 파일은 `/root/Resnet/` 내부에 둔다.

구현: [공통 캡처·측정 모듈](../module_profiling.py), [worker](module_worker.py), [runner](module_run.py), [시간 감시](budget.py). 검증 자료: [준비 검증](evidence/b1-module-preparation.json), [NPU 사전 검증](evidence/b1-module-npu-smoke.json).
