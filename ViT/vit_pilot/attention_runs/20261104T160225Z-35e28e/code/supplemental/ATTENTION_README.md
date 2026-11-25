# Attention CPU fallback 연구

사용자의 추가 요청에 따라 기존 pilot와 followup 범위에 Attention 부분 compile 연구를 추가한다. 기존 코드·결과·캐시는 보존하고 `attention_runs/<고유 run_id>/`에 새 결과를 쓴다. SDK, torch, 전역 설정, 드라이버는 변경하지 않는다.

```bash
cd /root/ViT
/root/venv/bin/python -u vit_pilot/attention.py run
# 기존 전처리 자산을 명시할 때
/root/venv/bin/python -u vit_pilot/attention.py run \
  --assets-dir /root/ViT/vit_pilot/followup_runs/20261104T132538Z-c031a6
# NPU forward 없이 저장 결과 재집계
/root/venv/bin/python vit_pilot/attention.py analyze --run /ABS/ATTENTION_RUN
# 기본 MHA 자체의 top-frame skip을 피하는 사용자 코드 경계만 별도 연구
# 앞선 worker 실행 예산을 차감한 초를 --total-budget으로 전달한다.
/root/venv/bin/python -u vit_pilot/attention.py run \
  --only-attention --attention-boundary --total-budget REMAINING_SECONDS
```

기존 실제 사진 4장의 전처리 완료 BF16 텐서를 그대로 읽고 파일/배치/가중치 해시를 확인한다. 새 이미지 검색·생성·전처리를 하지 않는다. 실제 이미지 폴더는 `/root/ViT/data/vit_real_images_4/images/`이며 순서는 cat, coffee, astronaut, rocket이다. B1은 첫 사진, B4는 네 사진이다. 원본 출처와 순서는 복사한 image_manifest.json에 보존한다. 정답 라벨은 없고, category 이름은 예측값이며 accuracy/loss를 계산하지 않는다.

조건은 모두 ViT-B/16 V1, BF16, eval/inference_mode, furiosa:0, 기본 Fusion 1 PE다.

| 조건 | 실행 경로 변경 |
| --- | --- |
| eager_default | 기존 nn.MultiheadAttention 기본 fastpath 유지. CPU fallback을 포함할 수 있는 진단 기준선 |
| attention_compile | 12개 encoder의 기존 self_attention에 `Module.compile(backend=ft.backend)` 적용. 나머지는 Eager |
| eager_fastpath_off | 해당 worker 안에서만 `torch.backends.mha.set_fastpath_enabled(False)` 후 기존 MHA 실행. 종료 때 원래 값 복원 |
| full_compile | 기존 전체 모델 torch.compile 기본 backend 설정을 대조 조건으로 사용 |

첫 실행에서 기본 MHA에 직접 `Module.compile`을 적용하면 backend 진입 없이 native fallback이 재현됐다. Dynamo의 top-frame skip 규칙을 조회한 뒤 `--attention-boundary` 조건을 추가했다. 이 조건은 준비 단계에 `AttentionBoundary`를 두며, forward는 전달받은 q/k/v와 kwargs로 원래 MHA를 그대로 호출한다. 기존 파라미터 객체 집합이 같은지 검사하고 그 사용자 코드 경계를 compile한다. state_dict 이름에 `original_attention` 경로가 추가되지만 가중치 값은 바꾸지 않는다. `attention_boundary_worker.py`는 isolated worker 안에서만 기존 ViT 준비 함수를 감싸 재사용하며, SDK/Torch 함수·파일을 수정하지 않는다. 성능 실행의 wrapper는 실험 대상 실행 경계이고 진단 hook이 아니다.

Attention 수식, need_weights, 파라미터, 가중치, 해상도, 정밀도를 바꾸지 않는다. fastpath-off는 명시적인 실행 경로 ablation이다. dynamic/fullgraph/mode/recompile_limit, CPU 스레드 설정을 튜닝하지 않는다.

먼저 기본 Eager strict B1 실패를 재현하고 기본 경로 profile을 수집한다. 변경 조건의 strict B1이 성공하고 logits 구조·유한값·Top-1 참조 일치가 확인되면 B4로 진행한다. strict는 공식 `ft.config.cpu_fallback(False)`를 별도 worker에서 적용한다. 실패한 조건은 반복 확대로 숨기지 않는다. 기본 Eager의 혼합 실행은 fallback 원인 연구용으로 명시하고 측정하며 NPU-only 결과로 표현하지 않는다.

성능은 새 프로세스에서 조건당 B1/B4 각각 독립 trial 1회, first 1 + warmup 5 + repeat 20이다. CLI의 `--warmup`, `--repeats`, `--worker-timeout`(기본 1800초), `--total-budget`(기본 7200초)으로 조절할 수 있다. worker는 직렬 실행하고 사용 중인 NPU가 있으면 보류한다. timeout은 미지원 판정이 아니며 이번 실행의 프로세스 그룹만 종료한다.

원래 common.py의 타이머를 재사용한다. 공식 current_stream 사전 동기화 → perf_counter_ns → 전체 모델 호출 → stream 동기화 → 종료 시각 순서다. ms/batch-call이며 CPU dispatch, 내부 전송·대기를 포함한다. 외부 모델/입력 준비와 출력 CPU 복사는 제외한다. profiler와 상세 로그·backend observer는 별도 진단 worker에서만 켠다. 각 worker의 전용 SDK cache 경로는 처음 없어야 하며 기존 공용 캐시를 삭제하지 않는다. 시스템 전체 cold, 실제 disk hit/miss, 순수 compile 시간은 미확인이다.

보고서와 CSV는 관측된 원본 시간으로만 만든다. 모든 성능 조건이 완료되면 8 workers × 26 = 208개 timing 행이며 진단 호출은 별도다. 반복 표준편차는 ddof=1, 처리량은 B × 유효 반복 수 / 시간 합계다. 기본 Eager 대비 비율은 CPU fallback을 포함한 전체 호출 시간 비율이다. 합의된 허용 오차는 없으므로 출력 수치와 Top-1 일치가 있어도 tolerance=pending으로 남긴다. 이전 3 trial 결과와 합쳐 독립 trial 수를 늘리지 않는다.

설치 소스의 핵심 근거는 PyTorch activation.py의 native fastpath 선택과 PrivateUse1 장치 허용, Dynamo variables/torch.py의 `_is_make_fx_tracing: True`, Furiosa local_decomp.py의 SDPA math 분해다. 실행별 소스 사본/해시, dispatch table, strict 오류, CPU/NPU trace와 원본 logits를 보존한다. backend 진입, PyTorch 재컴파일, NPU task 수, 실제 codegen, cache hit/miss는 각각 다른 관측이다.
