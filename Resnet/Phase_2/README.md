# Phase 2 — ResNet50 Eager / Compile 배치 스케일링

Phase 1에서 NPU 전체 실행이 확인된 `bn_folded_output_split128` 변형(BF16, CPU fallback=False)을 그대로 사용해,
**정상상태(steady-state) 지연시간이 배치 크기에 따라 어떻게 변하는지**를 Eager / Compile로 비교한다.
Phase 1 코드와 결과는 수정하지 않고 공통 모듈(`models.py`, `runtime.py`, `analysis.py`, `artifacts.py`, `environment.py`)과
`Phase_1/run.py`의 프로세스 timeout 처리를 재사용한다. 결과 해석은 [REPORT.md](REPORT.md)에 둔다.

## 측정 절차

| 항목 | 설정 |
| --- | --- |
| 조건 1개 = 프로세스 1개 | (모드, 배치, 회차)마다 새 프로세스 + **새 compiler 캐시 디렉터리** |
| 호출 순서 | 첫 호출 1회(컴파일 포함, 기록만) → 워밍업 10회 → 반복 50회 |
| 기준 측정 범위 `forward_to_cpu` | 입력을 NPU에 미리 올리고 동기화한 뒤, forward 호출부터 출력 `.to("cpu")` 반환까지 |
| 보조 측정 (별도 루프 50회) | H2D \| forward+동기화 \| D2H 세 구간, 유휴 stream 동기화 비용 |
| 배치 스케일링 | 배치 1·2·4·8·16·32 × 3회차, 이후 여유가 있으면 64·128·256 |
| 기존 결과 재검증 | 배치 1을 5개 독립 세트(회차 1–5)로 측정해 Phase 1 값과 비교 |
| 실행 순서 | 회차 우선(회차 1의 모든 배치 → 회차 2 → …), 배치·회차마다 모드 순서 교차 |
| 실패 처리 | 로그·traceback을 `errors/`에 보존하고 다음 조건 진행. 한 번 실패·timeout한 (모드, 배치)는 재시도하지 않음 |

통계는 성공한 job의 반복 구간만 사용한다. 첫 호출·워밍업은 별도 표에만 나온다.
표준편차는 표본 표준편차, P95는 선형 보간이다. 처리량 = `배치 × 1000 / 평균 지연`,
배속 = `Eager 평균 / Compile 평균`, 스케일링 효율 = `(처리량(B) / 처리량(1)) / B`.

## 실행

```bash
cd /root
# 환경 점검 (작은 Conv의 새 캐시 컴파일까지 확인, 지연 측정 아님)
XDG_CACHE_HOME=/root/Resnet/.cache /root/venv/bin/python -m Resnet.Phase_2.preflight
# 본 실험 (새 run-dir 필수; 순차 실행, 장치 lock 사용)
XDG_CACHE_HOME=/root/Resnet/.cache TMPDIR=/root/Resnet/.tmp \
  /root/venv/bin/python -u -m Resnet.Phase_2.run --run-dir Resnet/Phase_2/results/<새 이름>
# PE 4개 fusion으로 이전 실행의 실패 지점(eager-b2-t1)부터 새 run-dir에서 이어 하기
#   1-PE 실행에서 eager 배치 2·4가 "insufficient sequencer entries"로 실패했다.
#   furiosa:1 = 물리 PE 4–7 (BERT는 furiosa:0 = PE 0–3). 시작점 이전 job은 skipped로 남고, 배치 1 기준값은 회차 2–5에서 fusion 4로 새로 측정된다.
XDG_CACHE_HOME=/root/Resnet/.cache TMPDIR=/root/Resnet/.tmp \
  /root/venv/bin/python -u -m Resnet.Phase_2.run --config Resnet/Phase_2/config-fusion4.json \
  --run-dir Resnet/Phase_2/results/<새 이름> --start-from eager-b2-t1 \
  --previous-run Resnet/Phase_2/results/batch-scaling-1h-20261119-103155
# 중단된 실행 이어서 하기 (중단된 job의 폴더·캐시는 보존만 하고 새로 실행)
/root/venv/bin/python -u -m Resnet.Phase_2.run --run-dir <run-dir> --resume
# 원시값에서 표 다시 만들기 (NPU 사용 없음)
/root/venv/bin/python -m Resnet.Phase_2.analyze <run-dir>
# 그래프 (matplotlib가 있는 인터프리터 사용)
/root/ViT/vit_pilot/.plot-venv/bin/python Resnet/Phase_2/plots.py <run-dir>
# 단위 테스트 (CPU만 사용)
/root/venv/bin/python -m unittest discover -s Resnet/tests
```

runner는 시작 전에 cross-compiler가 aarch64 libc 헤더를 찾는지 확인한다. 헤더(`libc6-dev-arm64-cross`)가 없으면
SDK의 모든 신규 컴파일이 실패하므로 실행하지 않는다.

## 결과 파일 (run-dir)

| 파일 | 내용 |
| --- | --- |
| `raw_latency.csv` | 모든 job의 원시 샘플 (첫 호출·워밍업·반복·보조 구간 포함) |
| `trial_summary.csv` | 회차(프로세스)별 Mean / Std / Median / P95, 첫 호출 시간, 전송 구간 |
| `condition_summary.csv` | (모드, 배치, 측정 범위)별 통합 통계와 회차 평균의 분산 |
| `batch_scaling.csv` | 배치별 지연·처리량·배속·스케일링 효율 |
| `revalidation_b1.csv` | 배치 1의 세트별·통합 통계와 Phase 1 값 |
| `initial_costs.csv` | 첫 호출 시간 (기록용, 비교 제외) |
| `correctness.csv` | Eager↔Compile, 같은 모드의 회차 간, Phase 1 CPU 기준 출력 비교 |
| `failures.csv`, `errors/` | 실패 job과 보존한 로그·traceback |
| `coverage.csv` | 계획한 모든 job의 실행 여부 (`ok` / `failed` / `timeout` / `skipped` / `not_run`) |
| `figures/` | 지연·처리량·배속·효율·첫 호출·전송 그래프 |
| `jobs/`, `caches/`, `source/`, `evidence/`, `manifest.json`, `PROGRESS.md` | job별 원본, 캐시, 실행 당시 소스 사본, 환경 기록, 진행 상태 |
