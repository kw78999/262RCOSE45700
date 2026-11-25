# RNGD ViT 사전 실험

`torchvision.models.vit_b_16`, `ViT_B_16_Weights.IMAGENET1K_V1`, BF16, eval/inference_mode, 입력 `[B,3,224,224]`와 logits `[B,1000]`만 다룬다. B=1·4, 이미지당 내부 토큰 197개다. 정확도 검증이나 일반적인 성능 우열을 확정하는 실험이 아니다.

실측 완료: [첫 피드백 요약](runs/20261104T113745Z-8fccff/feedback_summary.md), [상세 보고서](runs/20261104T113745Z-8fccff/report.md), [원본 재계산 점검](runs/20261104T113745Z-8fccff/artifact_audit.json). 2026-11-04 20:37–20:54 KST, 기본 1 PE, BF16, 합성 고유 입력 4개, 독립 trial 1회. 고정 타이밍 104/104 및 shape 전환 12/12 완료. 진행 판단은 조건부다. 실제 이미지/출력 오차 기준/전체 연산 실행 경로/실제 codegen·disk cache 관측이 후속 확인 사항이다.

## 실행

저장소 루트 `/root/ViT`에서 기존 SDK 가상환경을 사용한다. 매 실행은 UTC 시각과 임의 suffix로 고유 디렉터리를 생성한다. 기존 결과를 덮어쓰지 않는다.

```bash
/root/venv/bin/python -u vit_pilot/pilot.py run
# 고정 이미지가 있을 때: 정확히 서로 다른 파일 4장, 나열한 순서 유지
/root/venv/bin/python -u vit_pilot/pilot.py run --images a.jpg b.jpg c.jpg d.jpg
# 원본 결과 재집계 (NPU forward 없음)
/root/venv/bin/python vit_pilot/pilot.py analyze --run-dir vit_pilot/runs/RUN_ID
/root/venv/bin/python vit_pilot/pilot.py --help
```

기본 CLI는 worker당 1800초, 자산 준비를 포함한 전체 운영 예산 7200초다. 컴파일 예상 시간이 아니다. `--worker-timeout`, `--total-budget`으로 조정한다. 성능 기본값은 `--warmup 5 --repeats 20 --trials 1`. `--device furiosa:0` 사용. 합의된 기준이 생기면 `--atol VALUE --rtol VALUE`를 둘 다 지정한다. 미지정 시 수치만 보고하며 합격을 선언하지 않는다. 타임아웃은 미지원 판정이 아니다.

실제 실행 명령·순서·PID·코드 해시·시각은 각 run의 manifest.json, 로그의 process.json 및 events.jsonl에 보존한다. 기존 프로젝트의 preflight/ 코드를 읽고 SDK API와 측정 경계를 재사용했으며 해당 디렉터리는 변경하지 않았다. /root/ViT는 Git 저장소가 아니므로 commit 대신 스크립트 SHA256을 기록한다. 커밋/푸시하지 않는다.

## 환경과 의존성

2026-11-04 확인: `/root/venv/bin/python` 3.12.3, torch 2.10.0+cpu, torchvision 0.25.0+cpu, furiosa-torch 및 furiosa-kernels 2026.4.0. torch/torchvision 실제 import 성공. CPU 빌드 torch에 Furiosa PrivateUse1 장치 확장이 등록되어 NPU를 사용한다. torchvision metadata는 `torch==2.10.0`을 요구한다. [공식 호환 표](https://github.com/pytorch/vision), [공식 CPU 빌드 안내](https://pytorch.org/get-started/previous-versions/)도 확인했다. torchvision 추가 설치는 필요하지 않았다. 과거 preflight/의 torchvision 부재 기록은 현재 설치 상태와 구분한다.

RNGD npu1, PCI 0000:04:00.0, firmware 2026.4.0 rev93d67f3. 실제 버전·드라이버·사용 상태는 실행별 environment.json과 device snapshot이 기준이다. 별도 Python distribution furiosa-sdk/furiosa-runtime 부재는 SDK 전체 부재를 뜻하지 않는다. 동시 사용자 작업은 실행하지 않으며, 각 worker 전 furiosa-smi ps에 사용 PID가 있거나 상태를 확인할 수 없으면 실행을 보류한다. 외부 사용자와의 원자적인 장치 예약까지 제공하는 runner는 아니다.

설치는 runner가 자동 수행하지 않는다. 기존 환경이 공용인지 미확인이므로 torch/SDK를 변경하지 않았다. 그림 패키지는 `vit_pilot/.plot-venv`에만 설치한다. dry-run 및 설치 명령·버전은 plot_install_dry_run.log/json, plot_requirements.txt, plot_install.log/json에 보존한다. 해당 환경에는 torch/Furiosa를 설치하지 않는다.

```bash
python -m venv vit_pilot/.plot-venv
vit_pilot/.plot-venv/bin/python -m pip install --dry-run --report vit_pilot/plot_install_dry_run.json -r vit_pilot/plot_requirements.txt
# dry-run 목록에 torch/Furiosa가 없는지 확인한 뒤 그림 전용 환경에만 추가
vit_pilot/.plot-venv/bin/python -m pip install --no-deps -r vit_pilot/plot_requirements.txt
```

## 고정 조건과 타이머

공식 enum의 URL에서 표준 `torch.hub.get_dir()/checkpoints/vit_b_16-c867db91.pth`를 사용한다. 없으면 공식 표준 다운로드를 수행한다. URL hash prefix와 전체 SHA256을 확인·저장한다. 오프라인이면 이 파일을 해당 경로에 준비해야 한다. 가중치 준비/모델 로드 시간은 추론 시간에서 제외된다. weights=None 점검·모델 변형·Attention 교체는 하지 않는다.

기본 입력은 CPU generator seed=42의 FP32 정규분포 텐서 4개를 한 번 만들어 inputs.pt로 저장한 뒤 두 모드 모두 BF16으로 변환한다. B1은 첫 샘플, B4는 전체 4개. 입력 파일·BF16 sample·batch 해시와 순서를 assets.json에 저장한다. 합성 입력은 synthetic_input=true이며 실제 이미지 검증을 수행했다는 뜻이 아니다. `--images`를 주면 각 이미지를 RGB로 읽어 `weights.transforms()`를 적용한다.

`common.py`는 모델 독립 timer/저장, `vit.py`는 모델·입력, `environment.py`는 환경/소스 근거, `worker.py`는 smoke/고정 shape/shape 전환, `pilot.py`는 순차 실행·시간 제한, `analyze.py`는 집계·출력 비교, `plots.py`는 독립 그림 생성이다. 다른 모델 adapter는 없다.

모델·입력을 NPU로 전송한 뒤 공식 `furiosa.torch.current_stream(device).synchronize()` 및 query()로 준비 완료를 확인한다. 사전 forward는 없다. SDK 설치 소스 `_C/stream/__init__.pyi`는 이전 stream 작업 완료까지 host가 대기한다고 문서화한다. 같은 current stream을 모든 호출에 유지한다.

```python
stream.synchronize()              # timer 밖, 이전 작업 완료
t0 = time.perf_counter_ns()
output = runner(inputs)           # Eager 또는 한 번 만든 Compile runner
stream.synchronize()
t1 = time.perf_counter_ns()
```

CPU에서 관측한 배치 모델 호출~완료 시간으로 CPU 디스패치를 포함한다. 입출력 전송 제외이며 순수 kernel 시간이 아니다. CPU 출력 복사/유한값 확인/파일 저장/print는 timer 밖이다. 모든 forward는 동일 eval/inference_mode. runner_setup_ms는 torch.compile wrapper 생성 등을 별도 기록한다. 첫 호출 1회, warmup 5회, repeat 20회를 기록한다. SDK compiler 기본 implicit_type_casting도 보존하므로 BF16 표기는 모델·입출력 dtype이며 모든 내부 연산이 BF16이라고 보장하지 않는다.

SDK 기본 Eager CPU fallback=True를 변경하지 않는다. Compile backend의 eager_fallback=False, torch suppress_errors=False를 확인한다. fallback을 새로 활성화하거나 실패를 숨기지 않는다. 진단에서 fallback/graph 분할이 발견되면 별도 표시한다. output.device만으로 전체 NPU 실행을 주장하지 않는다. 성능 로그만으로 fallback/추가 컴파일 부재를 확정하지 않는다.

## 순서·캐시·진단

Smoke: 새 프로세스로 Eager B1, Compile B1. 둘 다 성공한 뒤 Eager B4, Compile B4. B1 실패 시 B4를 실행하지 않는다. 모두 성공하면 고정 성능 Eager B1 → Compile B1 → Eager B4 → Compile B4. 각 케이스·trial은 새 프로세스지만 worker 안에서는 runner를 재사용한다. 이어 별도 새 worker에서 모드별 `[1,1,4,4,1,4]`를 사전 forward/shape 워밍업 없이 같은 runner로 실행한다.

설치 SDK에서 확인한 `ft.config.cache_context(run/'compiler-cache', summary=False)`만 사용한다. 새 run에서 해당 경로가 없음을 확인하고 기존 공용 캐시는 삭제하지 않는다. 이 전용 캐시는 smoke와 성능/진단 사이에 공유하므로 후속 첫 호출은 완전 cold compile이 아니다. 상태는 fresh_process_persistent_cache_possible / same_process_reuse로 남기며 실제 hit/miss와 구분한다.

진단 worker 시작 전에 설치 registry가 지원하는 TORCH_LOGS=recompiles,graph_breaks와 확인된 FURIOSA_TORCH_LOG=debug, Python furiosa_torch DEBUG를 설정한다. 성능에는 상세 로그를 끈다. backend 관측 wrapper는 진단에만 있으며 SDK backend에 그래프를 그대로 전달한다. stderr VIT_CALL_BEGIN/END와 stdout events.jsonl로 요청·로그를 연결한다. PyTorch graph 재컴파일, SDK backend 진입, 실제 compiler codegen, disk cache hit/miss는 별도 값이다. 미관측은 null/unknown이고 시간 변화로 추측하지 않는다. LRU 정보는 private 설치 심볼의 읽기 전용 관측이며 프로세스 내 ATen cache 정보다. compiler 실제 실행 횟수로 해석하지 않는다.

## 산출물과 해석

각 runs/RUN_ID에 environment/config/manifest/assets.json, inputs.pt, timings/summary/comparisons/shape_switch.csv, output_comparison.json, cases/, outputs/, logs/, figures/, report.md, feedback_summary.md를 만든다. 부분 결과도 케이스별 보존한다. 정상 고정 실행이면 104개(4×26) timing, shape 전환이면 12개 완료 호출이다. 미실시 shape 요청도 상태표에는 포함하고 완료 측정으로 세지 않는다. JSON은 allow_nan=False, 미측정은 null, CSV는 빈 필드다.

repeat mean/median/sample SD(ddof=1), `throughput_images_s = B * valid_repeat_count / sum(valid_repeat_seconds)`를 원본에서 집계한다. 정상 완료·유효 반복·출력 구조 조건이 충족되어야 처리량을 제공한다. 같은 batch끼리 `compile_speedup=Eager mean/Compile mean`; >1이면 Compile이 빠르다. 서로 다른 batch의 배속을 평균내지 않는다. `first_call_extra_ms=first-repeat_mean`은 순수 컴파일 시간이 아니다. 실제 SDK compile 구간이 없으면 compiler_time_ms=null이다.

warm_state는 10회 이상 repeat에서 CV≤0.10 및 앞/뒤 절반 평균 상대차≤0.10인 경우 preliminary_stable, 아니면 not_stable, 자료 부족 시 unknown이다. 이는 이번 코드의 투명한 잠정 기준이며 steady state 증명이 아니다. 느린 정상 호출을 삭제하지 않는다. 단일 trial 20회로 P95/P99나 우열의 일반화를 주장하지 않는다.

저장 logits를 CPU FP32로 비교한다. shape/NaN/Inf, max/mean abs error, 이미지별 cosine 및 평균/최소, Top-1 일치 분자/분모/비율, 불일치 ID·클래스·logit을 저장한다. 영벡터 cosine은 null+이유. 기본 허용 오차는 미정이다. Top-1 일치는 정답 라벨 accuracy가 아니다. B1/B4를 합쳐 고유 표본 5개로 세지 않는다.

그림은 first와 repeat mean±sample SD를 분리한다. shape 진단 그림의 x 표시는 로그로 관측한 backend 진입이며 실제 codegen으로 표시하지 않는다. 측정이 하나도 없으면 가상의 차트를 생성하지 않는다. 정확한 조건·캐시·출력 검증이 미확인인 배속은 예비 수치다.

## 분석 코드 점검

```bash
/root/venv/bin/python -m unittest discover -s vit_pilot/tests -v
```

테스트의 합성 로그/출력은 vit_pilot/tests/ 아래 임시 경로에만 저장한다. NPU 실측 run과 합치지 않는다. 전체 모델 실행은 위 `pilot.py run`으로 별도 검증한다. B8·B16, 독립 trial 확대, 대표 모듈 분석은 첫 피드백 이후 단계이며 자동 수행하지 않는다.
