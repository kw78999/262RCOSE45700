# ViT 후속 실험: 실행 경로·실제 이미지·독립 trial

Run `20261104T132538Z-c031a6`, 이전 합성 run은 `/root/ViT/vit_pilot/runs/20261104T113745Z-8fccff`. 모든 시간 원본은 UTC, 사용자 일정 표기는 KST(UTC+9). **예비 결과**, BF16·기본 1 PE, B1/B4 유지.

1. **기존 약 1초 Eager 지연 [문서·소스/실행 확인]**: 명시적 모델 로드·입력 준비·출력 저장은 타이머 밖이었다. 기본 Eager에서 native attention CPU fallback을 확인했고, 대표 호출에서 native attention 12회의 호스트 inclusive 합계는 B1 약 1017ms, B4 약 1006ms였다. 복사·대기를 포함할 수 있어 원인별 비율이나 순수 CPU 계산 시간으로 표현하지 않는다. 기존 합성 배속을 NPU 전용 두 모드 비교로 해석할 수 없다. [실행 경로 근거](path_evidence.md).
2. **CPU fallback 및 NPU 사용 [실행 확인/미확인]**: 4/4 경로에서 NPU 사용 근거, 2/4 경로에서 실제 CPU tensor fallback을 관측했다. 전체 연산 배치는 미확인이다. strict 진단과 기본 설정 상태를 분리했다.
3. **실제 이미지 [실행 확인]**: 제공된 cat → coffee → astronaut → rocket 순서, 파일 해시 검증 후 weights.transforms()를 한 번 적용해 저장했다. synthetic_input=false, 고유 이미지 4개이며 B1은 B4 첫 사진을 재사용한다. 정답 라벨이 없어 accuracy/loss를 계산하지 않았다.
4. **독립 실행 재현성 [실행 확인/미확인]**: 성능 worker 6/12 완료, 6개 보류. 유효 시간 336개; 전체 설계는 672개이며 보류 제외 예정 수는 336개다. 각 worker 첫 1 + warmup 5 + repeat 50. Eager 확대가 보류되면 paired speedup도 계산하지 않는다. 독립 trial 평균의 중앙값/최소/최대를 아래에 제시하며 반복 150회를 독립 실험 150개로 세지 않는다.
5. **본실험 판단: Compile은 유효 조건에서 조건부 진행 / Eager 기본 경로는 문제 해결 우선 [실행 확인/미확인]**. 출력 허용 기준은 pending, 전체 연산 배치 및 실제 codegen/disk cache hit는 미확인이다. batch 8·16/모듈 분석으로 확대하지 않았다.

| B | mode | 독립 trial 수 | trial mean ms | 중앙값 [최소, 최대] ms | trial images/s |
| --- | --- | --- | --- | --- | --- |
| 1 | eager | 0 | 보류/미실시 | 미측정/미확인 [미측정/미확인, 미측정/미확인] | 보류/미실시 |
| 1 | compile | 3 | [3.6726, 3.6738, 3.6723] | 3.6726 [3.6723, 3.6738] | [272.28, 272.2, 272.31] |
| 4 | eager | 0 | 보류/미실시 | 미측정/미확인 [미측정/미확인, 미측정/미확인] | 보류/미실시 |
| 4 | compile | 3 | [10.4565, 10.5795, 10.4539] | 10.4565 [10.4539, 10.5795] | [382.54, 378.09, 382.63] |


| B | trial | 예비 paired speedup | 상태/보류 이유 |
| --- | --- | --- | --- |
| 1 | 1 | 미측정/미확인 | Eager CPU fallback 조건의 성능 확대 보류 |
| 1 | 2 | 미측정/미확인 | Eager CPU fallback 조건의 성능 확대 보류 |
| 1 | 3 | 미측정/미확인 | Eager CPU fallback 조건의 성능 확대 보류 |
| 4 | 1 | 미측정/미확인 | Eager CPU fallback 조건의 성능 확대 보류 |
| 4 | 2 | 미측정/미확인 | Eager CPU fallback 조건의 성능 확대 보류 |
| 4 | 3 | 미측정/미확인 | Eager CPU fallback 조건의 성능 확대 보류 |


| B | max/mean abs | cosine 평균/최소 | Top-1 일치 | 허용 기준 | 출력 구조·예측 일치 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.0215/0.0042 | 0.99995221/0.99995221 | 1/1 | pending | True |
| 4 | 0.0625/0.0047 | 0.99995068/0.99994270 | 4/4 | pending | True |


위 출력 점검과 모드별 확대 조건은 별개다. 출력이 일치해도 CPU fallback이 확인된 Eager 성능 확대는 보류했다.

## 실제 입력 및 전처리

| 순서/ID | 실제 절대 경로 | 원본 W×H | 제공된 출처/조건 |
| --- | --- | --- | --- |
| 01_cat | /root/ViT/data/vit_real_images_4/images/01_cat.png | [451, 300] | Stefan van der Walt / CC0-1.0 |
| 02_coffee | /root/ViT/data/vit_real_images_4/images/02_coffee.png | [600, 400] | Rachel Michetti; courtesy of Pikolo Espresso Bar / CC0-1.0 |
| 03_astronaut | /root/ViT/data/vit_real_images_4/images/03_astronaut.png | [512, 512] | NASA Great Images database / Public domain, as documented by scikit-image |
| 04_rocket | /root/ViT/data/vit_real_images_4/images/04_rocket.jpg | [640, 427] | SpaceX / Public domain, as documented by scikit-image |


원본 제공 manifest 사본: inputs/provided_image_manifest.json. 출처·설명·순서는 보존했고 설명을 정답으로 사용하지 않았다. [image_manifest.json](image_manifest.json)은 파일 SHA256과 FP32/BF16 tensor SHA256을 구분한다. 전처리 정보: `{'weights_id': 'ViT_B_16_Weights.IMAGENET1K_V1', 'transform': 'ImageClassification(\n    crop_size=[224]\n    resize_size=[256]\n    mean=[0.485, 0.456, 0.406]\n    std=[0.229, 0.224, 0.225]\n    interpolation=InterpolationMode.BILINEAR\n)', 'rgb_conversion': True, 'random_augmentation': False, 'torchvision': '0.25.0+cpu', 'pillow': '13.0.0', 'fp32_tensor_file': '/root/ViT/vit_pilot/followup_runs/20261104T132538Z-c031a6/inputs/preprocessed_fp32.pt', 'bf16_tensor_file': '/root/ViT/vit_pilot/followup_runs/20261104T132538Z-c031a6/inputs/preprocessed_bf16.pt', 'batch_fp32_tensor_sha256': {'1': '9209694292fcfea140cc5dde051e7d34e40860cdc42e10da65e77501da61a82d', '4': '790922cf969aed0b12942e9edf24dcdeec664f0dc5fef9d7dea7131a472d7bb4'}, 'batch_bf16_tensor_sha256': {'1': '08b52c5a2a27fe6d0e20fa7d2d99a1c8cf03671ba9c0d51f7ff2490732c33d23', '4': 'e943285ad3e9668d6e3a6466aae88f6e2697109336cbfe4aed83cae62579b7f7'}, 'file_hashes_are_not_tensor_hashes': True, 'prepared_once': True, 'effective_transform_settings': {'crop_size': [224], 'resize_size': [256], 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'interpolation': 'InterpolationMode.BILINEAR', 'antialias': True}}`. 모든 worker가 같은 `/root/ViT/vit_pilot/followup_runs/20261104T132538Z-c031a6/inputs/preprocessed_bf16.pt`를 로드하고 batch별 해시를 확인한다. 전처리를 반복 수행하지 않는다. 원본 FP32 텐서는 `/root/ViT/vit_pilot/followup_runs/20261104T132538Z-c031a6/inputs/preprocessed_fp32.pt`.

| B | 이미지 | Eager 예측 | index | logit | Compile 예측 | index | logit | 일치 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 01_cat | Egyptian cat | 285 | 8.3750 | Egyptian cat | 285 | 8.3750 | True |
| 4 | 01_cat | Egyptian cat | 285 | 8.3750 | Egyptian cat | 285 | 8.3750 | True |
| 4 | 02_coffee | espresso | 967 | 9.3125 | espresso | 967 | 9.3125 | True |
| 4 | 03_astronaut | bobsled | 450 | 8.3750 | bobsled | 450 | 8.3750 | True |
| 4 | 04_rocket | drilling platform | 540 | 8.3750 | drilling platform | 540 | 8.3125 | True |


클래스 이름은 weights.meta.categories의 예측 이름이다. 점수는 raw logit이며 정답/accuracy/loss가 아니다. 허용 기준을 결과에 맞춰 정하지 않았다. 영벡터 cosine은 null 및 이유, 이미지별 오차/예측과 의심스러운 동일 출력 검사는 real_output_comparison.json에 보존했다.

## 실제 이미지 성능 trial별 원본 요약

| trial | mode | B | 상태 | setup ms | first ms | repeat mean±SD ms | images/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | eager | 1 | held | 미측정/미확인 | 미측정/미확인 | 미측정/미확인±미측정/미확인 | 미측정/미확인 |
| 1 | compile | 1 | ok | 242.5709 | 202951.0407 | 3.6726±0.0161 | 272.2843 |
| 1 | compile | 4 | ok | 264.1939 | 440363.6284 | 10.4565±0.0212 | 382.5374 |
| 1 | eager | 4 | held | 미측정/미확인 | 미측정/미확인 | 미측정/미확인±미측정/미확인 | 미측정/미확인 |
| 2 | compile | 1 | ok | 242.2101 | 201809.6887 | 3.6738±0.0177 | 272.1983 |
| 2 | eager | 1 | held | 미측정/미확인 | 미측정/미확인 | 미측정/미확인±미측정/미확인 | 미측정/미확인 |
| 2 | eager | 4 | held | 미측정/미확인 | 미측정/미확인 | 미측정/미확인±미측정/미확인 | 미측정/미확인 |
| 2 | compile | 4 | ok | 263.5601 | 437383.2055 | 10.5795±0.7671 | 378.0902 |
| 3 | eager | 4 | held | 미측정/미확인 | 미측정/미확인 | 미측정/미확인±미측정/미확인 | 미측정/미확인 |
| 3 | compile | 4 | ok | 263.9567 | 442180.5749 | 10.4539±0.0170 | 382.6311 |
| 3 | compile | 1 | ok | 241.8305 | 197405.2881 | 3.6723±0.0203 | 272.3095 |
| 3 | eager | 1 | held | 미측정/미확인 | 미측정/미확인 | 미측정/미확인±미측정/미확인 | 미측정/미확인 |


표준편차는 worker 내 반복의 표본 SD(ddof=1), 독립 trial 사이 변동은 across_trial_summary.csv에 별도 기록한다. 처리량=B×유효 반복 수/유효 반복 초 합계이며 서버 최대 처리량이 아니다. 첫 호출은 새 SDK 캐시 경로 조건에서 별도 관측했고 first_call_extra는 순수 compile 시간이 아니다. compiler_time_ms=null.

## 이전 합성 결과: 별도 표

| 이전 합성 / mode | B | repeat 평균 ms | 반복수 |
| --- | --- | --- | --- |
| compile | 1 | 3.7102 | 20 |
| compile | 4 | 10.4741 | 20 |
| eager | 1 | 1027.9730 | 20 |
| eager | 4 | 1124.3351 | 20 |


이전 run은 반복 20회·trial 1회·smoke와 SDK cache 공유, 새 run은 실제 이미지·반복 50회·trial 3회·worker별 새 SDK cache다. 입력 내용 하나의 효과로 차이를 해석하지 않는다.

## 실행·캐시·코드·보호 규칙

환경은 environment.json, 실제 worker 순서·PID·시각·명령·cache 경로·초기 상태는 manifest.json 및 cases/*/result.json에 기록했다. 성능 worker별 torch thread/inter-op thread, SDK compiler/device config, CPU fallback 기본값, 입력·가중치 해시와 측정 경계가 저장된다. 원래 common.py/vit.py는 그대로 재사용하고 새 followup.py/followup_worker.py만 진입점을 추가했다. 변경 전후 해시와 code/ 사본을 보존한다. 기존 합성 run과 기존 SDK/torch/드라이버/공용 캐시를 수정하거나 삭제하지 않는다.

진단 코드 수정 이력: 최초 diagnostic_compile_b1은 관측 wrapper의 중복 utc 인자로 실패했다. SDK 미지원 오류가 아니다. 원본 실패를 보존하고 새 retry case ID에서 수정된 관측 wrapper로 재실행했다. 수정·재개를 위해 이 실행의 orchestrator에만 SIGINT를 보냈고 당시 strict_compile_b4의 자기 worker만 중단했다. 이 중단은 timeout/미지원으로 해석하지 않는다. strict_compile_b4도 새 attempt에서 재개하며 기존 성공 결과·사용 예산을 유지한다. 성능 worker의 모델·입력·timer·backend 설정은 이 수정으로 바뀌지 않았다. 최초/재개 명령과 해시는 manifest 및 resume_history에 남겼다.

공식 current_stream.synchronize 준비/호출 전/후 완료 대기를 사용한다. 준비에만 query=True를 확인하고 timer 안에는 runner + 완료 대기만 있다. 모델/입력 전송·로그·파일 저장·CPU 출력 복사는 timer 밖이다. 기본 torch.compile(backend=ft.backend)에서 dynamic/fullgraph/mode/options를 지정하지 않는다. 성능 runner에 진단 backend wrapper/profiler/hook/strict 옵션을 넣지 않는다. eval/inference_mode와 BF16을 유지한다.

trace의 `npu...::Task::compile`은 이름이 compile인 NPU 실행 프로그램 task이며 실제 compiler codegen 이벤트가 아니다. 진단 Profiler는 자체 profile 결과 D2H 전송도 추가할 수 있으므로 trace의 모든 copy 이벤트를 모델 내부 전송으로 합산하지 않는다. native attention CPU 구간과 copy 함수 호출의 포함 관계도 단순 더하지 않는다.

NPU worker는 직렬, 매 worker 전 device usage를 재확인한다. 기존 사용자 PID가 보이면 실행을 보류한다. worker 기본 1800초, 전체 NPU 예산 7200초(설정값은 config.json). 운영상 상한이며 timeout은 미지원 판정이 아니다. 이 실행에서 생성한 process group만 정리한다. 실제 worker 중에는 외부 작업이 새로 시작할 수 있어 원자적 장치 예약을 보장하지 않는다.

## 산출물 일관성

`{'utc': '2026-11-04T14:33:13.309776+00:00', 'status': 'pass', 'errors': [], 'expected_performance_records': 672, 'actual_performance_records': 336, 'completed_valid_performance_records': 336, 'completed_workers': 6, 'expected_workers': 12, 'all_workers_complete': False, 'held_workers': 6, 'expected_records_for_unheld_workers': 336, 'all_unheld_workers_complete': True, 'note': 'Only performance rows counted; diagnostics and output verification excluded; repeats are not independent trials'}`

미측정/실패/보류는 0ms로 채우지 않는다. JSON nonfinite 값은 허용하지 않는다. 느린 정상 호출은 삭제하지 않는다. input/weight/dtype/timer identity, worker별 cache path 고유성 및 초기 부재, CSV를 다시 읽은 평균/표본 SD/throughput/paired speedup을 검사했다. [원본 시간](timings.csv), [trial별 요약](trial_summary.csv), [배속](comparisons.csv), [trial 사이 요약](across_trial_summary.csv), [원본 audit](artifact_audit.json). 그림은 figures/ 아래 PNG이며 로그 축이면 축과 caption에 표시한다.

## 재현·단계별 재개

정확한 실행 명령: `['/root/venv/bin/python', 'vit_pilot/followup.py', 'run', '--images', '/root/ViT/data/vit_real_images_4/images/01_cat.png', '/root/ViT/data/vit_real_images_4/images/02_coffee.png', '/root/ViT/data/vit_real_images_4/images/03_astronaut.png', '/root/ViT/data/vit_real_images_4/images/04_rocket.jpg']`. `followup.py run --images FILE FILE FILE FILE`는 새 고유 run을 만든다. `--stop-after diagnostic` 또는 `--stop-after validation`으로 단계 경계에서 멈춘 뒤 `followup.py resume --run-dir RUN`으로 미실시 단계만 진행한다. 이미 실행한 worker는 덮어쓰지 않는다. 재집계: `/root/venv/bin/python vit_pilot/followup.py analyze --run-dir /root/ViT/vit_pilot/followup_runs/20261104T132538Z-c031a6`. 기존 CLI 형식도 `pilot.py --images`가 nargs=4임을 확인했다. 모델당 첫 호출을 검증 forward로 소비하지 않는다.

최종 판단: **Compile은 유효 조건에서 조건부 진행 / Eager 기본 경로는 문제 해결 우선**. 실제 장애물은 raw 오류/보류 이유를 참고하고, 추가 본실험은 출력 기준·실행 경로 확인 범위를 합의한 뒤 검토한다. 이번 작업은 B1·B4·ViT 세 범위만 수행한다.
