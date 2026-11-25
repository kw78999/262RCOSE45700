# ViT RNGD Eager vs torch.compile 초기 점검

Run `20261104T113745Z-8fccff`. 시간 원본은 UTC, 사용자 기준일은 KST(UTC+9). **예비 결과, 독립 trial=1/케이스.**

1. **실행 가능성 [실행 확인]**: Eager/Compile B1·B4 smoke 4/4, 고정 성능 케이스 4/4 완료. 정상 완료 케이스는 BF16 logits 형태·유한값과 공식 stream 완료 대기를 확인했다. 기본 장치 구성을 유지했다.
2. **첫 호출·반복 성능 [실행 확인]**: eager B1: 첫 4530.8206 ms, 반복 1027.9730±37.2357 ms, 0.9728 images/s; compile B1: 첫 9660.3532 ms, 반복 3.7102±0.2068 ms, 269.5250 images/s; eager B4: 첫 4818.0746 ms, 반복 1124.3351±39.7859 ms, 3.5577 images/s; compile B4: 첫 9993.0565 ms, 반복 10.4741±0.0113 ms, 381.8950 images/s. 잠정 안정 기준 충족 4/4; steady state 증명은 아니다. 아래 배속은 같은 batch의 반복 평균만 사용한 예비 수치다.
3. **재컴파일·캐시 [실행 확인/미확인]**: Compile 진단에서 PyTorch 재컴파일 1회, backend 진입 2회(초기 포함)를 관측했다. B1→B4에서 tensor x의 index 0 크기 expected 1 / actual 4 guard 실패가 남았다. 실제 codegen·disk hit/miss는 **미확인**이다. SDK compile 진입 로그와 동일시하지 않는다. smoke와 전용 디스크 캐시를 공유한 후속 첫 호출은 완전 cold가 아니다.
4. **출력 [실행 확인/미확인]**: B1: max abs=0.0156, Top-1 1/1; B4: max abs=0.0312, Top-1 4/4. CPU FP32 비교이며 NaN/Inf·형태 상태는 원본 JSON에 기록했다. 허용 오차: 미정. 전체 고유 입력 4개, B1 첫 입력이 B4에도 포함된다. Top-1 일치율은 정답 accuracy가 아니다.

| mode | batch | 실행 상태 | runner setup ms | first ms | 반복 mean ± SD ms | median ms | images/s | 출력 상태 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eager | 1 | ok | 0.0028 | 4530.8206 | 1027.9730 ± 37.2357 | 1020.3526 | 0.9728 | 형태·유한값 정상 / 오차 기준 미정 |
| compile | 1 | ok | 148.8815 | 9660.3532 | 3.7102 ± 0.2068 | 3.6646 | 269.5250 | 형태·유한값 정상 / 오차 기준 미정 |
| eager | 4 | ok | 0.0029 | 4818.0746 | 1124.3351 ± 39.7859 | 1129.1695 | 3.5577 | 형태·유한값 정상 / 오차 기준 미정 |
| compile | 4 | ok | 148.8848 | 9993.0565 | 10.4741 ± 0.0113 | 10.4723 | 381.8950 | 형태·유한값 정상 / 오차 기준 미정 |


평균±표준편차는 유효 repeat만 사용하고 표본 표준편차(ddof=1)를 표시한다. 첫 호출은 1회 관측이다. 워밍업 5회, 반복 20회/worker. 실패·미실시 값은 0으로 채우지 않았다.

- B1, trial 1: speedup=277.0644; Preliminary only: output tolerance or all-op execution path unverified
- B4, trial 1: speedup=107.3445; Preliminary only: output tolerance or all-op execution path unverified

| mode/호출 | batch | ms | 상태 | graph 재컴파일/backend 진입 | 실제 compile / disk hit | 로그 |
| --- | --- | --- | --- | --- | --- | --- |
| eager/1 | 1 | 4512.6791 | ok | 0/0 | unknown / unknown | [원본](logs/switch_eager.stderr.log) |
| eager/2 | 1 | 1005.5740 | ok | 0/0 | unknown / unknown | [원본](logs/switch_eager.stderr.log) |
| eager/3 | 4 | 4678.8530 | ok | 0/0 | unknown / unknown | [원본](logs/switch_eager.stderr.log) |
| eager/4 | 4 | 1143.3352 | ok | 0/0 | unknown / unknown | [원본](logs/switch_eager.stderr.log) |
| eager/5 | 1 | 1044.3233 | ok | 0/0 | unknown / unknown | [원본](logs/switch_eager.stderr.log) |
| eager/6 | 4 | 1136.9123 | ok | 0/0 | unknown / unknown | [원본](logs/switch_eager.stderr.log) |
| compile/1 | 1 | 9688.7954 | ok | 0/1 | unknown / unknown | [원본](logs/switch_compile.stderr.log) |
| compile/2 | 1 | 3.8458 | ok | 0/0 | unknown / unknown | [원본](logs/switch_compile.stderr.log) |
| compile/3 | 4 | 9947.7879 | ok | 1/1 | unknown / unknown | [원본](logs/switch_compile.stderr.log) |
| compile/4 | 4 | 10.6263 | ok | 0/0 | unknown / unknown | [원본](logs/switch_compile.stderr.log) |
| compile/5 | 1 | 3.8106 | ok | 0/0 | unknown / unknown | [원본](logs/switch_compile.stderr.log) |
| compile/6 | 4 | 10.5373 | ok | 0/0 | unknown / unknown | [원본](logs/switch_compile.stderr.log) |


진단 시간에는 상세 로그 영향이 있다. 기본 성능 표에 합치지 않았다. 로그 행의 컴파일 진입은 실제 codegen 횟수와 다르며 프로세스 LRU hit/miss는 디스크 hit/miss가 아니다.

| batch | 입력/고유 표본 | max / mean abs | cosine mean / min | Top-1 분자/분모 | 판정 |
| --- | --- | --- | --- | --- | --- |
| 1 | synthetic; 비교 1 / 전체 고유 4개 | 0.015625/0.003264 | 0.99998997/0.99998997 | 1/1 (100%) | 허용 오차 미정 |
| 4 | synthetic; 비교 4 / 전체 고유 4개 | 0.031250/0.003441 | 0.99998863/0.99998601 | 4/4 (100%) | 허용 오차 미정 |


**진행 판단: 조건부 진행.** 고유 합성 입력 4개·허용 오차 미정·전체 연산 실행 장치 미확인·실제 codegen/disk hit 미관측. 순차 images/s는 서버 최대 처리량이 아니다. 출력 허용 오차 확정 여부와 후속 실험 확대 판단은 별개다.

## 환경·실행·측정 경계

Python `/root/venv/bin/python`, `3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0]`. OS `Linux-6.8.0-124-generic-x86_64-with-glibc2.39`. 환경 및 장치 원문: [environment.json](environment.json), [종료 장치 상태](logs/device_after.json).

| 항목 | 실제 설치/확인 |
| --- | --- |
| torch | 2.10.0+cpu |
| torchvision | 0.25.0+cpu |
| furiosa-torch | 2026.4.0 |
| furiosa-kernels | 2026.4.0 |
| furiosa-runtime | 미설치/metadata 없음 |
| furiosa-sdk | 미설치/metadata 없음 |
| numpy | 2.5.4 |
| Pillow | 13.0.0 |
| matplotlib | 미설치/metadata 없음 |


기존 공용 여부는 미확인으로 취급하고 기존 Python/torch/SDK/드라이버를 변경하지 않았다. torchvision은 import 및 패키지 metadata에서 torch==2.10.0 호환을 확인했다. 공식 [호환 표](https://github.com/pytorch/vision)와 [CPU 배포 안내](https://pytorch.org/get-started/previous-versions/)도 확인했다. 추가 설치가 없어 dry-run은 해당하지 않는다.

지침: /AGENTS.md, /root/AGENTS.md, /root/ViT/AGENTS.md 및 저장소 하위 AGENTS.md 없음. 기존 preflight/를 읽고 공식 sync/cache API와 의미를 재사용했다. 기존 preflight의 덧셈·batch2·DEBUG 포함 시간과 본 실험을 합치지 않았다. 이번 공통 timer는 query 계측을 timer에서 제외했다. 기존 파일은 변경하지 않았다.

모델·입력: [assets.json](assets.json), [config.json](config.json). 공식 가중치는 표준 torch hub 경로에 다운로드하고 URL SHA256 prefix 및 전체 파일 SHA256을 보존했다. weights=None 대체는 없다. 실제 이미지가 저장소에서 발견되지 않아 CPU FP32 seed=42 정규분포 텐서 4개를 한 번 저장하고 BF16으로 변환했다. batch별·sample별 BF16 바이트 해시를 검증한다. B1은 첫 샘플, B4는 전체 4개다. 224×224 / patch16의 내부 토큰은 이미지당 197개이며 B4를 토큰 길이 788로 표현하지 않는다.

준비에는 가중치 로드, BF16 변환, NPU 전송 및 준비 stream 완료 대기가 포함된다. 준비 중 사전 forward는 없다. runner_setup_ms는 Eager runner 참조 준비 또는 torch.compile 래퍼 생성만 별도 기록한다. 시간은 prewait → perf_counter_ns → runner(inputs) → current_stream.synchronize → perf_counter_ns 순서다. CPU 디스패치와 완료 대기를 포함하고 kernel-only 시간이 아니다. 출력 CPU 복사·파일 쓰기·print·오차 계산은 timer 밖이다. 동일 eval/inference_mode를 모든 forward에 적용한다. 기본 dynamic/fullgraph/mode/재컴파일 제한을 변경하지 않는다. compiler_time_ms는 독립 SDK 구간 미확인으로 null이다. first_call_extra_ms는 순수 compile 시간으로 부르지 않는다.

SDK 기본 Eager CPU fallback=True; 이 값을 새로 켜거나 오류 억제를 켜지 않았다. Compile backend의 eager_fallback=False와 별개다. 기본을 보존했으므로 성공한 output.device만으로 NPU 전체 실행을 주장하지 않는다. 진단의 fallback/graph 분할 로그와 실행 경로 확인 상태를 별도 남긴다.

동기화 근거: 설치 furiosa/torch/config/_stream.py의 current_stream, _C/stream/__init__.pyi의 Stream.synchronize/query; backend 근거: _backend/torch_compile.py의 backend=_Backend(), 기본 eager_fallback=False. 설치 사본과 줄 번호·해시: [installed_sources.txt](logs/installed_sources.txt), environment.json.sources. 준비 동기화 후 query=True를 실행으로 점검하며 완료 API 실패 시 시간 측정으로 진행하지 않는다. 기본 단일 current stream에 대한 계약이며 다른 스트림을 임의 생성하지 않는다.

캐시: 확인된 SDK cache_context(root_dir, summary=False)로 run/compiler-cache를 사용했다. 실행 전 경로는 없었고 worker들이 순차 공유한다. smoke가 후속 성능 캐시를 채웠을 수 있다. fresh_process_persistent_cache_possible은 worker 내부 LRU가 새로 시작함을 뜻할 뿐 disk cache hit를 증명하지 않는다. 공유 캐시 삭제는 없다. 시스템의 다른 캐시는 cold라고 주장하지 않는다.

실행 한도: worker 기본 1800초, 전체 최대 7200초(본 CLI는 자산 준비부터 계산해 더 보수적이다). 운영상 상한이며 예상 compile 시간이 아니다. timeout은 미지원 확정이 아니다. worker는 새 세션으로 생성하고 이 실행의 process group만 TERM/KILL한다. worker 전 device ps/status를 다시 기록하며 다른 PID가 보이면 실행하지 않는다. 장치 ps의 순간 스냅샷 사이에 외부 작업이 생기는 경쟁 가능성은 완전히 제거하지 못한다. utilization '-'는 0%로 추정하지 않았다.

## 진단 이벤트 해석

TORCH_LOGS 항목은 설치 torch registry에서 확인한 뒤 worker 시작 전 설정한다. FURIOSA_TORCH_LOG=debug는 설치 config/_env.py 근거를 사용한다. Python furiosa_torch DEBUG는 smoke/shape 진단에만 적용한다. 성능 worker는 warning 로그를 사용하므로 추가 컴파일 이벤트가 없었다고 단정하지 않는다. shape 진단 backend wrapper는 전달받은 gm/inputs를 SDK 기본 ft.backend로 그대로 전달한다. backend 호출의 집계 단위는 Dynamo graph이고 SDK 실제 compiler count가 아니다. stderr의 VIT_CALL_BEGIN/END와 stdout events.jsonl로 요청 귀속을 남긴다. 재컴파일 로그 행·guard 이유, backend entry, SDK compile-entry 로그, 실제 codegen, disk hit/miss를 각각 분리한다. 초기 backend entry와 후속 entry도 구분한다. 시간 증가/감소만으로 컴파일/cache hit를 판정하지 않는다.

## 실패 및 미실시

실패 worker 없음; 모든 상태는 manifest.json을 참고한다.

| case | 상태 | 이유 |
| --- | --- | --- |


정확한 실행 명령과 worker PID·시작/종료 시각: [manifest.json](manifest.json). 오류 traceback은 cases/*/result.json과 logs/*stderr.log에 원형 보존했다.

## 집계·그림 검증

{
  "utc": "2026-11-04T12:00:35.921869+00:00",
  "expected_fixed_records": 104,
  "actual_fixed_records": 104,
  "completed_valid_fixed_records": 104,
  "fixed_count_check": "pass",
  "expected_shape_calls": 12,
  "shape_status_rows": 12,
  "completed_shape_calls": 12,
  "shape_count_check": "pass",
  "csv_row_count_matches": true,
  "recalculation": [
    {
      "case_id": "fixed_eager_b1_t1",
      "valid_repeats_match": true,
      "mean_matches": true,
      "throughput_matches": true
    },
    {
      "case_id": "fixed_compile_b1_t1",
      "valid_repeats_match": true,
      "mean_matches": true,
      "throughput_matches": true
    },
    {
      "case_id": "fixed_eager_b4_t1",
      "valid_repeats_match": true,
      "mean_matches": true,
      "throughput_matches": true
    },
    {
      "case_id": "fixed_compile_b4_t1",
      "valid_repeats_match": true,
      "mean_matches": true,
      "throughput_matches": true
    }
  ],
  "script_sha256_at_analysis": {
    "pilot.py": "90c02fef0abb3f5c0196d3759fcc5898e89a0837742fa998181b172732b22a6e",
    "prepare.py": "ac70c1196e34dbfc18281cc81dce1cc8ce06d6d023c4d497765763d7c8d923be",
    "verify_artifacts.py": "72126bbf02d1e3b73518a22bc31d4ad525d10084f4dbe1761ed512ad065a8564",
    "plots.py": "f2c19018ffb5197ce1ef79461ee976943d9e4443a0776b911ef03c68a427bcfa",
    "analyze.py": "f12dc6a8a22abad0e8f7397a3b3751d2146274c76c53a4f3a955ace709098544",
    "vit.py": "e8802c4842403d12bf45a26c7fc65a36abeda166ba02e5ce9767c46409b323ee",
    "worker.py": "1677e9676b691004af197cc19c79c01adefa33d7f4aa1f6160c49116fb724d13",
    "environment.py": "8a2e154f2d64facf5218b9e8282de8605d815a908b3d3c92ffcda0ef0a3f15ab",
    "common.py": "2ca3097717b94c35aad6f339e3eb0175a8a2b4cd43ce06cd8bcf00f79962c2d9"
  }
}

그림 상태: {"status": "created", "files": ["02_shape_switch.png", "01_fixed_batch.png"]}. 값이 없으면 0으로 그리지 않는다. 정상 성능 데이터가 없으면 가상의 차트를 만들지 않는다. 원본 [timings.csv](timings.csv), [summary.csv](summary.csv), [comparisons.csv](comparisons.csv), [shape_switch.csv](shape_switch.csv), [output_comparison.json](output_comparison.json).

warm_state는 repeat CV≤0.10 및 앞/뒤 절반 평균 상대차≤0.10을 잠정 안정 기준으로 사용한다. 10회 미만이면 unknown이다. 이는 steady state 증명이나 상세 compiler 이벤트 부재 증명이 아니다. 느린 정상 호출을 이상치로 제거하지 않는다. 평균/처리량은 원시 유효 반복에서 재계산한다. 반복 미완료나 출력 구조 실패의 throughput 및 배속은 null이다. 반복간 변동과 trial간 변동은 별개다. 단일 trial 20회로 P95/P99, 일반적 우열, 원인, 손익분기점, 최적 batch를 주장하지 않는다.

## 재현과 다음 단계

저장소 루트에서 `/root/venv/bin/python -u vit_pilot/pilot.py run`을 실행하면 새 고유 run_id를 만든다. 재집계: `/root/venv/bin/python vit_pilot/pilot.py analyze --run-dir /root/ViT/vit_pilot/runs/20261104T113745Z-8fccff`. 설치 환경을 바꾸지 않으며 동일 가중치/입력·해시를 확인한다. CLI --worker-timeout, --total-budget, --warmup, --repeats, --trials, --device, --images, --atol/--rtol 지원. 독립 trial 확대와 B8·B16은 장애 해결 및 피드백 후, 그 뒤 대표 Encoder/Attention/MLP 분석을 수행한다. 이번 실행에서는 범위 확대하지 않았다.

장치 범위 [실행 확인/문서·소스 확인]: 기본 Fusion(num_pe_per_cluster=1), MeshKind.Single을 유지했다. 실행 중 smi의 worker는 npu1:0으로 표시됐다. 전체 물리 칩의 최대 처리량을 측정한 결과가 아니다. 타이머에서 제외한 전송은 명시적인 사전 입력/모델 전송과 사후 출력 CPU 복사다. 모델 내부의 DMA/프로그램 로드/장치 전환 비용은 호출 구간에 발생하면 포함된다.

진단 관측 [실행 확인]: Compile backend 진입 2회(초기 포함), PyTorch graph 재컴파일 로그 1회. SDK split 로그는 각 backend 진입 시 1개 segment를 기록했다. 진단에서 CPU fallback 및 graph break 로그는 관측되지 않았으나, 전체 연산 배치의 NPU 전용 실행 증명은 아니다. Eager process LRU와 SDK compile-entry 로그 집계는 shape_switch.csv의 별도 필드이며 실제 codegen/disk cache hit와 구분한다.

그림 축: 01의 first-call 패널은 선형, repeat 패널은 로그 y축이다. 02는 두 모드 모두 같은 로그 y축을 쓴다. 각 그림에 축 종류·단위·관측값을 명시했다. 오차막대는 repeat 표본 표준편차이며 신뢰구간이 아니다.

코드 근거: manifest.json의 초기 해시 및 worker별 해시, consistency_checks.json의 분석 시점 해시를 참고한다. code/ 사본과 provenance가 있으면 실행 중 완성한 진단 로그 표식/재현 CLI의 버전 차이를 함께 확인할 수 있다. 모델·입력·측정 timer를 바꾸어 성능을 유도하지 않았다.

- 피드백 질문 1: 이 SDK에서 실제 codegen과 disk cache hit/miss를 구분하는 공식 관측 방법은 무엇인가?
- 피드백 질문 2: BF16 Eager/Compile logits 비교의 atol/rtol 기준을 어떻게 합의할 것인가?
- 피드백 질문 3: 기본 Eager CPU fallback=True에서 전체 연산의 NPU 실행 범위를 확인할 최소 공식 방법은 무엇인가?

최종 진행 판단: **조건부 진행** [실행 확인/미확인]. 고유 합성 입력 4개·허용 오차 미정·전체 연산 실행 장치 미확인·실제 codegen/disk hit 미관측. 최소 조치는 실제 이미지 4장 확보, BF16 출력 기준 합의 및 전체 연산 실행/컴파일 관측 방법 확인이다. 이후 독립 trial 확대 → B8·B16 → 대표 Encoder·Attention·MLP 분석 순서를 검토하며 이번에는 추가 실행하지 않았다.
