# RNGD ResNet50 Eager / Compile 실험

공통 구현은 이 폴더에 두고, 실험 설정·실행·결과는 [Phase_1](Phase_1/README.md)에서 관리한다. 화면과 최종 보고서는 만들지 않는다.

**현재 실험:** `BN-folded + output-split128` 전체 모델 실험을 시작했다. `layer4.0/1/2.conv2`를 출력 128채널씩 4분기 후 concat으로 변환한다. 배치 1 최소 조건을 통과하면 본실험 grid를 자동 진행한다. [자동 갱신 진행 상태](Phase_1/results/bn-folded-split128-20261104/PROGRESS.md), [실측 상태와 이전 실패 이력](Phase_1/STATUS.md).

**Phase 2 (배치 스케일링):** 1-PE 실행(1시간 예산)에서 eager 배치 2·4가 sequencer 부족으로 실패해, PE 4개 fusion(`Phase_2/config-fusion4.json`, `furiosa:1` = 물리 PE 4–7)으로 `eager-b2-t1`부터 재개한다. BERT 실험(PE 0–3)과 PE를 공유하지 않으며, runner는 자기 PE와 겹치는 프로세스만 기다린다. 실행 폴더는 `Phase_2/results/LATEST`. [절차](Phase_2/README.md), [보고서](Phase_2/REPORT.md).

| 공통 모듈 | 역할 |
| --- | --- |
| `models.py` | 공식 ResNet50 V1.5, 명시적 ImageNet V2 가중치, 고정 시드 BF16 입력 |
| `folding.py` | eval Conv–BN 쌍을 FP32에서 결합하고 변환 후 모델 해시 기록 |
| `partitioning.py` | 지정 Conv의 출력 채널만 분할하고 전체 입력 채널·bias·stride 유지 |
| `runtime.py` | Eager / Furiosa Compile 경로, 공식 stream 동기화, E2E·forward 측정, backend 진입 횟수 |
| `environment.py` | Python·패키지·SDK·장치 상태와 설치 소스 보존 |
| `analysis.py` | 반복 통계, 출력 비교, 초기 비용 추정에 기반한 손익분기 경계 |
| `profiling.py` | Furiosa Chrome trace의 span 및 NPU Task 목록 정리 |
| `artifacts.py` | JSON·CSV 저장 |

작업 기준 디렉터리는 `/root`이며, 확인된 Python은 `/root/venv/bin/python`이다. 버전은 [requirements-observed.txt](requirements-observed.txt)와 각 실행의 `evidence/environment.json`에 기록한다. 이번 작업에서는 누락된 torchvision 및 Pillow를 추가했으며, 기존 torch·Furiosa SDK·드라이버 버전은 그대로 사용했다.

```bash
cd /root
/root/venv/bin/python -m unittest discover -s Resnet/tests -v
/root/venv/bin/python -m Resnet.Phase_1.run \
  --run-dir Resnet/Phase_1/results/run-002 --profiles

# Conv–BN folding 적용 모델
/root/venv/bin/python -m Resnet.Phase_1.run \
  --config Resnet/Phase_1/config-bn-folded.json \
  --run-dir Resnet/Phase_1/results/folded-run-002 --profiles

# 이번 본실험 변형: BN folding + layer4의 Conv2 출력 split128
/root/venv/bin/python -m Resnet.Phase_1.run \
  --config Resnet/Phase_1/config-bn-folded-split128.json \
  --run-dir Resnet/Phase_1/results/split128-run-002 --profiles
```

실행 폴더는 매번 새로 지정한다. 첫 실행 시 공식 가중치 약 98 MiB를 다운로드한다. 선택한 모델 변형의 최소 조건이 두 모드 모두 성공해야 나머지 성능 실험을 진행하며, 실패 시 원인과 미측정 항목을 남긴다. SDK가 달라진 뒤 같은 명령으로 다시 확인할 수 있다.

설계 근거: [기존 기획안](https://boatneck-bath-d60.notion.site/3fbbc586c4078036b9f0f2131eaf2b8d), [피드백 반영 수정안 B](https://boatneck-bath-d60.notion.site/B-403bc586c4078033af18e9100f90bcc3), [Furiosa 실행 경로](https://furiosa.ai/developers), [PyTorch / torchvision 버전 대응](https://pytorch.org/get-started/previous-versions/), [ResNet50 가중치 정의](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html).
