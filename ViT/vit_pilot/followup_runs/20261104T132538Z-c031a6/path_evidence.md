# ViT 실행 경로·Eager 지연 근거

이전 run: `/root/ViT/vit_pilot/runs/20261104T113745Z-8fccff`. 합성 원본 입력을 그대로 사용했다. 파일/텐서 해시 재확인은 [previous_audit.json](previous_audit.json), 원본 시간 재계산도 같은 파일에 있다. 기존 common.py timer와 vit.py 모델 준비 함수는 수정하지 않았다. +cpu 버전 표기는 tensor CPU fallback 증거가 아니다.

| 케이스 | 상태 | NPU 사용 | CPU fallback 관측 | 전체 연산 배치 |
| --- | --- | --- | --- | --- |
| eager B1 | ok | True | True | unverified |
| compile B1 | ok | True | False | unverified |
| eager B4 | ok | True | True | unverified |
| compile B4 | ok | True | False | unverified |


| strict case | 상태 | CPU fallback 설정 | 출력 유효 |
| --- | --- | --- | --- |
| strict_eager_b1 | error | False | 미측정/미확인 |
| strict_compile_b1 | ok | False | True |
| strict_eager_b4 | held | 미측정/미확인 | 미측정/미확인 |
| strict_compile_b4 | interrupted | False | 미측정/미확인 |
| strict_compile_b4_retry1 | ok | False | True |


| 기본 Eager | 대표 CPU native attention 호출 수 | 해당 호스트 구간 inclusive ms | 공식 strict 오류 |
| --- | --- | --- | --- |
| B1 | 12 | 1017.2890 | B1 strict: CPU fallback is disabled for aten::_native_multi_head_attention |
| B4 | 12 | 1005.9565 | B1 strict 오류를 참고; B4 strict는 보류 |


[실행 확인] 기본 Eager의 native attention 12회와 CPU 계산(mm/bmm/softmax 등), native attention의 PrivateUse1 전용 kernel 부재, strict B1의 명시적 CPU fallback 금지 오류를 함께 확인했다. B4 strict는 B1 실패 규약에 따라 미실시했으며 B4 기본 trace에서 같은 native attention CPU 경로가 관측됐다. Python `run_by_cpu::executed_by_cpu` span 또는 `fallback to CPU` 로그가 0이어도 이 native dispatcher fallback은 존재한다. 따라서 이전 Python fallback 로그 부재를 CPU fallback 부재로 확장할 수 없다. NPU와 CPU tensor 연산이 섞인 기본 Eager 결과로 판단하며, 해당 성능 확대를 보류한다. 위 inclusive 호스트 시간은 복사·대기를 포함할 수 있고 순수 CPU kernel 시간/전체 지연 비율로 해석하지 않는다.

추가 원본 근거: logs/native_attention_dispatch.json, logs/followup_installed_sources.txt, cases/diagnostic_eager_b*/torch_cpu_summary.json 및 python_profile.txt. 반복별 compile-entry/load 로그와 프로세스 LRU는 diagnostic_observations.json에 남겼다. 해당 캐시에서 반복 compile-entry/load가 관측되지 않아도 실제 compiler 내부 이벤트의 완전한 부재를 증명하지는 않는다.

공식 제어 [문서·소스 확인]: 설치 `furiosa/torch/config/_cpu_fallback.py:cpu_fallback(False)`는 해당 worker context에서 SDK CPU fallback을 비활성화하고 원래 값으로 복원한다. Python UnsupportedOpError 분기 외에 관측한 native backend fallback에도 적용되어 strict B1에서 RuntimeError가 발생했다. Compile backend eager_fallback=False와 별개다. strict 성공은 이 계약 범위의 점검이며 전체 연산 배치 증명은 아니다. 기본 성능 worker에는 strict 설정을 넣지 않았다.

| case | first ms | 진단 repeat ms (3회) | 대표 trace NPU task | CPU fallback span |
| --- | --- | --- | --- | --- |
| diagnostic_eager_b1 | 61808.2709 | [1008.9744, 1043.2056, 1076.9756] | 125 | 0 |
| diagnostic_compile_b1 | 미측정/미확인 | [] | 미측정/미확인 | 미측정/미확인 |
| diagnostic_eager_b4 | 91687.7500 | [1062.7007, 1083.254, 1088.2921] | 125 | 0 |
| diagnostic_compile_b4 | 미측정/미확인 | [] | 미측정/미확인 | 미측정/미확인 |
| diagnostic_compile_b1_retry1 | 9984.0909 | [3.6747, 3.6689, 3.6751] | 2 | 0 |
| diagnostic_compile_b4_retry1 | 425601.9447 | [10.4589, 10.4559, 10.4581] | 2 | 0 |


| 가설 | 관측·판단 | 근거 |
| --- | --- | --- |
| 기존 타이머에 매회 모델 로드/전처리/compile 래퍼/CPU 출력 저장 포함 | [문서·소스 확인] 코드에서 해당 작업은 타이머 밖이다. SDK 내부 준비 비용은 별개다. | common.py timed_call, vit.py prepare_model_inputs, 이전 worker 소스 사본 |
| 모델 tensor 연산의 CPU fallback이 Eager 지연의 원인 | [실행 확인/미확인] 아래 fallback 로그·executed_by_cpu trace·strict 결과 범위로 판단한다. 호스트 Python 실행 자체는 tensor fallback이 아니다. | diagnostics JSON, 원본 로그·trace |
| 반복 중 실제 codegen/프로그램 재로드 | [실행 확인/미확인] 호출별 SDK compile-entry/load 로그와 LRU 전후를 분리 기록했다. 실제 codegen·디스크 hit/miss는 unknown이다. | diagnostic_observations.json per_call |
| NPU 호출 수·CPU dispatch/런타임·내부 전송·대기 | [실행 확인/추정] 대표 호출의 NPU task와 호스트 span/cProfile을 관측한다. 큰 지연 기여 후보지만 중첩 span을 더해 원인별 비율을 만들지 않는다. | cases/diagnostic_*/furiosa_trace.json, torch_cpu_trace.json, python_profile.txt |


SDK 관측 근거: 설치 `_C/profiler/__init__.pyi:Profiler/RecordFunction`, config/_env.py의 FURIOSA_TORCH_PROFILE=trace, config/_stream.py current_stream 및 `_C/stream/__init__.pyi:Stream.synchronize`. CPU fallback 실제 실행 span은 `_backend/_exported_program.py:run_by_cpu::executed_by_cpu`, Eager 분기 근거는 `_backend/eager.py`다. 설치 사본/해시: logs/installed_sources.txt 및 environment.json. 상세 로그와 profiler는 진단 worker 및 마지막 추가 대표 호출에서만 사용했다. profiler 호출은 672개 성능 시간에 포함하지 않는다. CPU profiler 시간은 NPU kernel 시간이 아니며 비동기·중첩 span을 단순 합산하지 않는다.

정확한 최초 오류가 있으면 cases/<case>/result.json 및 logs/<case>.stderr.log의 traceback과 input_metadata를 참고한다. 기본·strict 오류는 별개 상태다. 실제 codegen 시간이나 CPU/NPU 시간 비율은 독립적으로 관측하지 않아 만들지 않았다.
