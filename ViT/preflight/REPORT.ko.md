현재 환경에서는 지정 ViT의 최소 실행을 완료할 수 없습니다. `/root/venv`에 `torchvision`이 설치되어 있지 않으며 실제 실행도 `ModuleNotFoundError`로 중단됐습니다. 패키지를 설치하거나 모델을 대체하지 않았습니다. 공식 동기화와 관측 도구는 설치 소스와 작은 BF16 덧셈으로 별도 확인했습니다. 아래 덧셈 시간은 ViT 성능이나 최종 벤치마크 결과가 아닙니다.

점검일: 2026-11-03, 약 15:29–15:32 KST. 로그의 UTC 시각은 KST보다 9시간 느립니다. 표기: **① 실행으로 확인됨 / ② 문서·소스에서 확인됨 / ③ 아직 확인하지 못함**. 소스는 설치된 2026.4.0을 우선하며 [줄 번호를 포함한 사본](evidence/installed_sources.txt)을 보존했습니다. 아래 상대 소스 경로의 기준은 `/root/venv/lib/python3.12/site-packages/furiosa/torch/`입니다.

**1. 환경과 기존 실행 경로**

| 항목 | 상태 | 확인 결과 |
|---|---|---|
| Python 실행 파일 / 가상환경 | ① | `/root/venv/bin/python`, `sys.prefix=/root/venv`, `base_prefix=/usr`, `VIRTUAL_ENV=/root/venv` |
| Python | ① | 3.12.3, GCC 13.3.0 |
| torch | ① | 2.10.0+cpu. CPU 빌드 표기에도 Furiosa PrivateUse1 확장으로 NPU 사용 가능; CUDA는 unavailable |
| torchvision | ① | metadata 및 import 대상 없음. ViT 실제 import 실패 |
| furiosa-torch / furiosa-kernels | ① | 모두 2026.4.0 |
| 확장 빌드 | ① | `furiosa-torch 2026.4.0 (rev: 5800e2ba24, built_at: 2026-10-12T09:11:33Z)` |
| 장치 관리 / 드라이버 | ① | furiosa-smi 2026.4.0, device driver 2026.4.0 rev 32ffd4e; Debian 패키지 furiosa-smi 2026.4.0-4 |
| NPU | ① | RNGD 물리 장치 npu1, PCI 0000:04:00.0, firmware 2026.4.0 rev 93d67f3 |
| 시작 상태 | ① | 메모리 0.00/47.50 GiB, 사용 프로세스 없음, 약 36°C / 38W. utilization은 `-`로 반환되어 0%라고 판정하지 않음 |
| 종료 상태 | ① | 메모리 0.00/47.50 GiB, 사용 프로세스 없음 |
| 논리 장치 | ① | `is_available()=True`, furiosa:0~7; 기본 Fusion=1 PE, MeshKind.Single. 덧셈 trace의 furiosa:0 실행은 npu1pe0으로 기록 |
| 전체 SDK / 독립 runtime 버전 | ③ | furiosa-sdk, furiosa-runtime이라는 Python 배포 metadata는 없음. SDK 전체가 미설치라는 뜻은 아님. 별도의 runtime/compiler 독립 버전은 확인하지 못함; 확장 빌드와 위 구성품 버전만 확인 |
| 기존 프로젝트 | ① | 시작 시 `/root/ViT`는 빈 디렉터리. 적용할 AGENTS.md 발견 안 됨. 기존 프로젝트 실행 명령 없음 |
| 서버의 참고 코드 | ①·② | `/root/torchTest.py`에 BF16 tensor를 furiosa:0으로 옮겨 더하는 예제 있음. 코드를 읽었으며 해당 파일 자체는 실행하지 않음 |

전체 출력: [환경 로그](logs/environment.log), [종료 장치 상태](logs/device-after.log). 초기 일반 샌드박스 명령은 `bwrap: Failed to make / slave: Permission denied`로 실행되지 않아 실행 도구의 escalation을 이용했습니다. 이 문제는 SDK 실패가 아닙니다.

**2. 지정 ViT와 실행 방법**

② 공식 torchvision 문서의 후보는 `vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)`이며 224×224 입력과 1,000개 클래스 출력을 갖습니다. 다른 가중치나 `weights=None`으로 대체하지 않습니다. [공식 builder 문서](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html?highlight=vit_b_16), [공식 구조 구현](https://docs.pytorch.org/vision/main/_modules/torchvision/models/vision_transformer.html).

② 설치 SDK의 기본 실행 경로는 다음과 같습니다. 이 경로를 [최소 점검 스크립트](check_vit.py)에 구현했습니다. backend 호출 횟수 기록용 래퍼는 전달받은 그래프를 그대로 `ft.backend(gm, inputs)`에 전달합니다.

```python
import torch
import furiosa.torch as ft
from torchvision.models import vit_b_16, ViT_B_16_Weights

device = torch.device('furiosa:0')
model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
model = model.eval().to(dtype=torch.bfloat16).to(device)
inputs = torch.randn(1, 3, 224, 224, dtype=torch.bfloat16).to(device)
compiled = torch.compile(model, backend=ft.backend)
with torch.inference_mode():
    eager_output = model(inputs)
    ft.current_stream(device).synchronize()
    compiled_output = compiled(inputs)
    ft.current_stream(device).synchronize()
```

② `__init__.py`가 `backend`를 공개하고, `_backend/torch_compile.py:125`의 `backend = _Backend()`는 기본 `compiler_config=None, eager_fallback=False`입니다. `torch.compile()`만 호출하면 기본 PyTorch backend를 선택하므로 Furiosa backend를 명시해야 합니다. `backend_eager_fallback`, `with_config`, `precompile`은 이번 실행에 사용하지 않았습니다. `dynamic`, `fullgraph`, `mode`, `options`도 지정하지 않았습니다. SDK 기본 컴파일 설정을 그대로 사용했고 커널·decomposition·모델 구조를 수정하지 않았습니다.

① BF16 tensor 전송과 덧셈은 Eager/Compile 양쪽에서 BF16 출력으로 확인됐습니다. 단, `ft.get_amp_supported_dtype()`는 빈 목록을 반환합니다. 따라서 autocast 지원을 추정하지 않고 모델과 입력을 명시적으로 BF16으로 변환합니다. ③ ViT 전체 연산의 BF16 지원은 아직 미확인입니다. 지원되지 않는 경우 dtype이나 연산을 조용히 대체하지 않고 실패합니다.

① 현재 기본 `ft.config.get_cpu_fallback()`은 True입니다. ② `_backend/eager.py:23`은 UnsupportedOpError 발생 시 이 설정에 따라 CPU로 실행할 수 있습니다. 해당 Python logger의 DEBUG 로그에 `fallback to CPU`와 예외가 남습니다. Compile backend의 Eager fallback 기본 비활성화와 Eager의 CPU fallback은 서로 다른 설정입니다. PyTorch graph break 후 Eager 실행에서도 후자가 적용될 수 있습니다. 기본값을 변경하지 않았으므로 이후 ViT가 성공해도 출력 장치만 보고 NPU 전용 실행이라고 판정하면 안 됩니다.

① 덧셈에서는 backend 로그에 `split into 1 submodules, 1 to compile`, RunnableSequence 및 NPU Task가 관측됐습니다. CPU fallback 실행 로그는 발견되지 않았습니다. 초기 `Unsupported aten function: [...]` 전체 목록은 backend 등록 시 나오는 일반 로그이므로 ViT에서 실제 발생한 미지원 연산 목록으로 해석하지 않습니다. ③ ViT의 미지원 연산, graph break, SDK 그래프 분할, CPU 전환 여부는 모델 import 전에 중단되어 미확인입니다.

① 기본 torch checkpoint 디렉터리 `/root/.cache/torch/hub/checkpoints`도 없습니다. 다른 위치의 가중치 존재는 ③ 미확인입니다. 스크립트는 정확한 enum URL의 기본 checkpoint 파일을 검사하고 없으면 중단합니다. `--allow-weight-download`를 명시할 때만 torchvision의 정상 다운로드를 허용하며 이번에는 사용하지 않았습니다. 의존성 확보 후에도 정확한 가중치가 필요합니다.

**3. 시간 측정과 완료 보장**

② 공식 공개 API는 다음과 같습니다.

| API | 보장과 사용 | 설치 근거 |
|---|---|---|
| `ft.current_stream(device)` | 장치의 현재 스트림. 별도 설정 없으면 default stream; 연산은 해당 장치 현재 스트림에 제출 | `config/_stream.py:7` |
| `stream.synchronize()` | 해당 스트림에 이전에 제출된 작업이 모두 끝날 때까지 host 대기 | `_C/stream/__init__.pyi:151` |
| `stream.query()` | 비차단 완료 확인; False이면 미완료 작업 존재 | 같은 파일 141 |
| `ft.Event(device)` / `event.record(stream)` / `event.synchronize()` | 기록 지점 이전 작업의 완료 대기 | 같은 파일 30, 59, 51 |
| `start.elapsed_time(end)` | 두 event를 기록하고 각각 synchronize한 뒤 ms 반환. 완료 timestamp 비교라는 문서상 제약 존재 | 같은 파일 79 |

① 실제 최소 실행에서는 `ft.current_stream('furiosa:0').synchronize()`와 `.query()`를 사용했습니다. Eager 첫 호출과 Compile 반복 호출에서 함수가 반환한 직후 `.query()==False`가 관측됐습니다. **반환만으로 실행 완료가 보장되지 않는다는 실제 근거**입니다. ② 스트림 문서도 비동기 제출을 명시합니다. ③ Event의 실행·시간값은 별도 시험하지 않았고, 이 점검은 wall-clock+공식 스트림 대기를 사용했습니다. device 전역 `ft.synchronize()` 같은 미확인 API는 사용하지 않았습니다.

```python
s = ft.current_stream(device)
s.synchronize()
t0 = time.perf_counter_ns()
y = fn(inputs)
t_return = time.perf_counter_ns()
ready = s.query()
s.synchronize()
t_done = time.perf_counter_ns()
# completion_ms = (t_done - t0) / 1e6
# 이후 출력 CPU 전송과 오차 계산: timed interval 밖
cpu_y = y.to('cpu', non_blocking=False)
s.synchronize()
```

① 스크립트의 `completion_ms`는 Python 호출, 첫 호출의 tracing/export/compile/cache 탐색/load, 제출, `query()` 관측 및 최종 완료 대기 비용을 포함합니다. 모델/입력의 사전 NPU 전송, 모델 생성, 가중치 로드, 출력 CPU 전송, 오차 계산은 제외합니다. 첫 호출과 반복 호출 모두 DEBUG 로깅이 켜져 있어 순수 성능 수치로 쓸 수 없습니다. `host_return_ms`는 반환까지의 참고치이며 완료 latency가 아닙니다. 순수 NPU kernel 시간도 아닙니다.

② 스트림 동기화는 다른 스트림의 무관한 작업까지 기다리지 않습니다. 이번 코드는 별도 스트림을 생성하지 않았고 실제 trace는 stream_id=0입니다. SDK 기본 단일 스트림 경로의 측정입니다. 모델이 여러 스트림을 사용하도록 바뀌면 의존성 및 완료 범위를 다시 확인해야 합니다.

② CPU 복사는 `_custom_ops/copy_from.py:22,36`에서 `non_blocking=False`를 `memory.copy`로 전달합니다. 내부 C 확장 전체의 CPU 복사 계약은 Python 소스만으로 확인하기 어렵습니다. 이번에는 이미 확인된 공식 동기화를 사용하므로 CPU 복사를 완료 대기의 대체 수단으로 삼을 필요가 없습니다. CPU 복사를 timer 안에 넣는 대안은 D2H 전송, CPU 메모리 할당 및 대기까지 포함하는 end-to-end 값이며 모델 latency와 분리해서 보고해야 합니다.

**4. 컴파일·캐시 관측**

| 구분 | 상태 | 이번에 확인한 관측 방법과 한계 |
|---|---|---|
| Eager 연산별 export/compile/load | ①·② | logger `furiosa_torch` DEBUG/INFO, `AtenOp` 로그, private LRU의 `.cache_info()` 전후. `_backend/aten.py:34,88,118` |
| Eager 프로세스 내 캐시 | ① | export/compile LRU maxsize 8192/8192, load LRU 실제 maxsize **32**. 같은 batch 반복 때 load-cache hits 증가, batch2에 misses 증가, 1 복귀 시 hit |
| 기본값 문서 차이 | ①·② | `config/_env.py`는 load 기본 2048을 설명하지만 실제 import 직후 및 실행 중 LRU는 32. 이 점검에서는 실제 `.cache_info()`를 보고하며 setter로 변경하지 않음 |
| PyTorch backend 호출 | ① | SDK backend 위의 얇은 래퍼에서 호출 수 기록. 덧셈 batch1에서 1, batch2에서 2, 1 복귀 시 여전히 2 |
| PyTorch 그래프 재컴파일 | ①·② | `TORCH_LOGS=graph_breaks,recompiles`; batch2에서 guard 실패와 `Recompiling function` 로그. 설치 torch `_logging/_registrations.py:146,159`에 항목 존재 |
| SDK 내부 그래프 분할 | ①·② | Python logger DEBUG의 graph, `split into ...`, RunnableSequence. `_backend/torch_compile.py:153` |
| Furiosa 컴파일 진입 | ① | `compile without given compiler config, use default config`, `graph compiled successfully`; 첫 실행에 4개 key 디렉터리와 output.bin/summary 생성 |
| 실제 compiler 산출물 | ① | 전용 compiler-cache에 input.fir, compile_params.yaml, key.bin, output.bin, summary/compiler_config.yaml, summary/summary.log 등 생성 |
| 디스크 캐시 기능과 기본 경로 | ①·② | `ft.config.get_cache_config()`의 `/root/.cache/furiosa/compiler`; 기존 28개 key 디렉터리 존재. `CacheConfig`, `cache_context(root_dir, summary=True)` 소스 확인 |
| 새 프로세스의 캐시 재사용 시도 | ① | 별도 프로세스에서 첫 실행의 **전용** 디스크 cache-dir를 명시. 프로세스 내부 LRU는 0부터 다시 시작함 |
| 디스크 hit/miss 정확한 판정 및 실제 compiler 실행 횟수 | ③ | 현재 DEBUG 로그에 명시적 disk hit/miss 레코드는 찾지 못함. 같은 경로의 두 번째 실행이 빨라도 hit로 확정하지 않음. compile 진입/성공 로그만으로 실제 재컴파일 횟수를 세지 않음 |
| 프로세스 종료 후 모든 캐시 초기화 | ③ | 보장되지 않음. 디스크 캐시가 남으며 다른 내부/시스템 캐시 전부의 초기화 여부는 미확인 |

② Rust 측 로그는 `FURIOSA_TORCH_LOG=debug` 또는 target=level 형태로 제어하며 `config/_env.py:33`이 근거입니다. Python 측은 별개로 `logging.getLogger('furiosa_torch').setLevel(logging.DEBUG)`를 설정했습니다. `RUST_LOG=warn`이 기존 환경에 있으나 변경하지 않았으며, 이를 Furiosa disk hit/miss 관측 설정이라고 주장하지 않습니다. 임의의 캐시 삭제 환경변수나 추측한 compiler 로그 API는 사용하지 않았습니다.

PyTorch 그래프 재컴파일 → backend 호출 → Furiosa 컴파일 경로 진입 → compiler 디스크 hit 또는 실제 codegen은 서로 다른 단계입니다. 한 backend 호출에서 여러 SDK segment를 컴파일할 수도 있고 디스크 cache를 읽을 수도 있습니다. Python과 Rust logger가 같은 메시지를 각각 출력하는 경우도 있어 로그 행 수를 compiler 실행 횟수로 세면 안 됩니다.

공유 캐시는 삭제·변경하지 않았습니다. 실행 시 `cache_context`로 프로젝트 내부 전용 경로만 지정했고 compiler 옵션은 바꾸지 않았습니다. `summary=True`는 산출물 관측을 위해 사용했습니다. 전용 디렉터리가 처음 없었다는 사실은 확인했지만 시스템 전체의 cold state라고 부르지 않습니다. 모델 비교 시에는 두 모드 실행 순서와 캐시 경로를 함께 보고해야 합니다. 이번 스크립트는 Eager 다음 Compile 순서이며 캐시를 공유하므로 모드별 독립 cold 비교가 아닙니다.

**5. 프로파일링 지원과 관측 범위**

② `furiosa.torch.profiler.Profiler(path)` 및 `RecordFunction(name)`를 설치 `_C/profiler/__init__.pyi:24,51`에서 확인했습니다. Profiler context 종료 시 Chrome trace JSON을 기록합니다. `config/_env.py:52` 기준으로 `FURIOSA_TORCH_PROFILE=trace`는 operator/DMA 상세, `info`/`debug`는 task 수준, `warn`/`error`는 NPU profiling 비활성입니다. 환경변수는 import/초기화 전에 설정해야 합니다.

```python
from furiosa.torch.profiler import Profiler, RecordFunction
with Profiler('new-trace.json'):
    with RecordFunction('ViT'):
        output = model(inputs)
        ft.current_stream(device).synchronize()
```

① 추가 프로파일 호출은 시간 측정 뒤 별도로 수행했습니다. [Furiosa trace](substrate-first/furiosa-trace.json)는 총 50개 이벤트이며 `npu1pe0::Task::aten::add.Tensor`, `npu1pe0::Task::compile`, launch 및 profile_d2h 이벤트가 있습니다. Task 이벤트에 begin_cycle/end_cycle/cycle_actual, stream_id가 남아 CPU 구간과 NPU 실행을 구분할 수 있습니다. 이는 기본 활성화 여부를 가정한 결과가 아니라 실제 JSON 검사 결과입니다. ③ 이 단순 덧셈에서 복잡한 모델 내부의 모든 operator/DMA 상세가 노출되는지는 검증하지 못했습니다.

① `torch.profiler.supported_activities()`는 CPU와 PrivateUse1을 반환하지만, 둘 다 요청한 실제 프로파일의 event device는 **DeviceType.CPU만**입니다. [torch trace](substrate-first/torch-trace.json)는 `aten::add`, Python annotation만 관측했고 NPU Task 이벤트는 없습니다. 따라서 이 환경에서 torch.profiler의 activity 목록만 근거로 NPU kernel 시간 지원을 주장하지 않습니다. ③ 모든 모델·연산에서 NPU 이벤트가 불가능하다고 일반화할 수도 없습니다.

② Eager는 ATen 및 compile/load span을 관측할 수 있고 Compile은 SDK segment/RunnableSequence 수준으로 묶일 수 있습니다. ③ ViT 전체, encoder layer, attention, MLP의 이름과 device 이벤트 대응은 미확인입니다. 모듈 경계와 compiler fusion 경계가 일치한다는 보장도 없습니다.

후속 대안: ViT 전체 최소 실행이 성공한 뒤에만 같은 가중치의 `model.encoder.layers.encoder_layer_0`, 그 안의 `self_attention`, `mlp`를 대표 블록으로 별도 점검합니다. 공식 구조상 block 입력은 `[B,197,768]`, MLP 입력은 hidden=768이며, attention에는 원래 forward의 ln_1 결과를 Q/K/V로 전달하고 `need_weights=False` 등 원래 인수를 유지해야 합니다. 이는 [공식 torchvision 구현](https://docs.pytorch.org/vision/main/_modules/torchvision/models/vision_transformer.html)에 근거한 제안이고 현재 설치 버전에서는 ③ 미확인입니다. 실제 전체 forward의 입력 activation을 별도 비측정 실행에서 확보하고, 원본 submodule을 두 모드에서 같은 BF16/eval/inference 설정으로 동기화 전후 측정할 수 있습니다. 인공 입력을 쓰면 그 사실을 명시합니다. 블록 단독 Compile은 fusion 경계가 달라지므로 전체 ViT 내부 기여 시간으로 합산하지 않습니다. Compile 경로에 hook을 추가하면 graph capture가 달라질 수 있어 전체 측정과 구분해야 합니다. 이번에는 수행하지 않았습니다.

**6. 최소 검증 결과와 남은 단계**

| 검증 | 상태 | 결과 |
|---|---|---|
| 지정 ViT batch1 실행 진입 | ① | import 시 `ModuleNotFoundError: No module named 'torchvision'`, exit=1 |
| ViT Eager/Compile 출력 형태·시간 | ③ | 모델 구성 이전에 중단. 예상 `[1,1000]`을 실행 결과로 보고하지 않음 |
| ViT 동일 입력 출력 오차 | ③ | 출력 없음. 허용 기준도 미정이므로 통과 판정 없음 |
| ViT batch2 및 1→2→1 | ③ | batch1 미성공으로 진행하지 않음 |
| BF16 덧셈 API 점검 | ① | 양 모드 `[1,16]` 및 `[2,16]` BF16, furiosa:0; 1→2→1 완료. 모델 대체 실험이 아닌 작은 API 점검 |
| 덧셈 출력 오차 | ① | 세 단계 모두 max_abs=mean_abs=RMSE=0. 정확도 합격 기준으로 사용하지 않음 |

아래는 **첫 전용 캐시 실행의 덧셈 점검값(ms)**입니다. 로그·계측·컴파일을 포함하고 반복이 2회뿐이며 ViT 성능, 정상상태 성능 또는 모드별 우열을 의미하지 않습니다.

| 덧셈 단계 | Eager 완료 시간 | Compile 완료 시간 |
|---|---:|---:|
| batch1 첫 호출 | 596.014 | 450.197 |
| batch1 반복1 | 0.213 | 0.141 |
| batch1 반복2 | 0.182 | 0.095 |
| batch2 첫 호출 | 417.215 | 422.843 |
| batch1 복귀 | 0.218 | 0.122 |

새 프로세스에서 같은 전용 디스크 경로를 지정한 덧셈 첫 호출은 Eager 378.147ms / Compile 251.947ms였습니다. 차이는 disk hit 증명이 아닙니다. 모든 원시값, LRU 변화, backend 호출 횟수는 [첫 실행 결과](substrate-first/results.json), [새 프로세스 결과](substrate-reuse/results.json), CSV 및 전체 로그에 있습니다.

실행 스크립트는 batch1에서 Eager/Compile 각 첫 호출+2회 반복, 그 다음 batch2 각 1회, batch1 복귀 각 1회를 순서대로 수행합니다. 어떤 단계가 예외로 실패하면 즉시 중단합니다. 프로파일링은 선택 옵션이며 성공한 측정 뒤에만 수행합니다. RNG seed와 CPU 입력을 고정하고 같은 NPU 입력을 두 모드에 전달합니다. 상대 오차는 eager 절댓값의 분모를 1e-8로 제한한 진단값이며 임의 허용 오차는 적용하지 않습니다. 입력은 synthetic BF16이며 ImageNet 정확도를 측정하지 않습니다.

현재 필요한 후속 조건은 같은 venv에 torch 2.10과 호환되는 torchvision을 확보하고 지정 checkpoint를 준비하는 것입니다. 이번 범위에서는 설치·업그레이드·드라이버 변경·캐시 삭제·장시간 실험을 하지 않았습니다. 준비 후 아래처럼 **새 run-dir**로 재실행할 수 있습니다. timeout도 사전 점검 한도이며 초과 시 성공으로 해석하지 않습니다.

```bash
cd /root/ViT
FURIOSA_TORCH_LOG=debug TORCH_LOGS=graph_breaks,recompiles \
  timeout 120s /root/venv/bin/python -u preflight/check_vit.py \
  --run-dir preflight/vit-next > preflight/logs/vit-next.log 2>&1
# 정확한 가중치 다운로드가 필요한 경우에만 --allow-weight-download 추가
# trace 별도 확인은 성공 후 새 run-dir와 --profile 사용
```

산출물: [최소 실행 스크립트](check_vit.py), [환경 수집 스크립트](collect.py), [사용 명령과 종료 상태](commands.sh), [ViT 실패 로그](logs/vit-attempt.log), [BF16 첫 실행 로그](logs/substrate-first.log), [캐시 경로 재사용 로그](logs/substrate-reuse.log), [trace 요약](evidence/trace-summary.json), [설치 소스 근거](evidence/installed_sources.txt). 외부 릴리스 문서는 [Furiosa SDK 2026.4.0](https://developer.furiosa.ai/latest/en/whatsnew/release-2026.4.0.html)이며, 실제 실행 경로의 판단은 설치 구현을 우선했습니다.
