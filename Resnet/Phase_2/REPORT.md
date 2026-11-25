# Phase 2 보고서 — ResNet50 Eager / Compile 배치 스케일링 (RNGD)

작성: 2026-11-19 KST, 갱신: 2026-11-20 KST
**문서 상태: 본 실험(PE 4 fusion, 58개 계획) 완료, 결과는 8절. Eager 배치 2·16 이상 실패의 원인 진단·수정(배치 분할 변형)과 그 측정(배치당 1회차) 완료.**
1–7절은 실험 전에 쓴 준비 기록이라 당시 시점의 서술(예: "미실행")을 그대로 둔다.

## 1. 요약

- **Phase 1 분석:** 전체 ResNet50은 `bn_folded_output_split128` 변형에서만 NPU 실행이 된다. 유효한 전체 모델 측정은 배치 1(양 모드)과 배치 4(Compile만)뿐이고, 조건당 프로세스 1개·반복 10회라 배치 스케일링 결론을 낼 수 없었다.
- **Phase 2 준비:** 요청 사양(첫 호출 1회 → 워밍업 10회 → 반복 50회, 조건마다 새 프로세스·새 캐시 3회, 배치 1~32 이상, H2D/D2H 분리, 출력 일치 검증, CSV·그래프)을 ResNet50에 맞춰 구현했다. 단위 테스트 35개 통과.
- **환경 문제 1건 해결:** 컨테이너 재생성(2026-11-18 18:31 UTC) 후 aarch64 libc 헤더가 없어 SDK의 모든 신규 컴파일이 실패했다. 사용자 승인 후 `libc6-dev-arm64-cross`를 설치했고, 새 캐시에서 작은 Conv의 Eager·Compile 컴파일·실행을 확인했다.
- **결과 (8절):** Compile은 배치 1–256(2 제외)에서 2.06 → 34.99 ms, 처리량은 배치 128에서 약 7,400 img/s로 포화. Eager는 배치 1–8에서 13.0–14.3 ms로 거의 일정해 배속이 6.3 → 5.1배로 줄었다.
- **Eager 실패와 수정 (8.5절):** op별 컴파일에서 SDK가 일부 (op, shape)를 거부해 b2·b16 이상이 실패했다. 실패 op만 배치 분할한 변형으로 b2·16·32·64·128 모두 실행·출력 일치를 확인했지만, 처리량은 427–531 img/s로 b8(560 img/s)을 넘지 못한다.
- **Eager 지연 원인 (8.6절):** 대부분 forward당 op 130회의 호스트 디스패치(약 11 ms)다. SDK 로드 캐시(32칸 < 고유 op 44개)의 재로드는 측정 조건에서 많아야 약 1 ms로 추정된다.

## 2. 사양 해석

붙여 주신 사양은 BERT(`textattack/bert-base-uncased-SST-2`)용으로 작성된 것이고, 같은 틀의 BERT 측정은 `/root/262RCOSE45700-msi/BERT/RESULTS.md`에 이미 있다. 요청이 `Resnet/`의 Phase_1 분석과 Phase_2 진행이므로 **그 사양을 ResNet50에 적용**했다. BERT 실험을 다시 돌리려던 의도였다면 이 해석이 틀린 것이다.

| 사양 항목 | Phase 2 구현 | 비고 |
| --- | --- | --- |
| 모델 | ResNet50 `IMAGENET1K_V2`, `bn_folded_output_split128` | Phase 1에서 NPU 전체 실행이 확인된 유일한 변형. state SHA256 `f4ffc872…3469` 일치 시에만 실행 |
| RNGD 1장, furiosa-torch 2026.4.0, torch 2.10.0, BF16, `cpu_fallback=False` | 동일 | Compile은 `torch.compile(model, backend=ft.backend)`만 지정, SDK 기본 compiler 설정 |
| Sequence Length 32/128/256 | **해당 없음** | 이미지 크기 스윕 등으로 대체하지 않았다. 입력은 `[B,3,224,224]` 고정 |
| CPU에서 Attention Mask 생성 후 NPU 전달 | **해당 없음** | ResNet50에는 mask 입력이 없다 |
| 기존 실험 재검증: 배치 1, 독립 5세트 | 배치 1을 새 프로세스·새 캐시 5회(회차 1–5) | Phase 1의 forward·E2E 값과 같은 표에서 비교 |
| 첫 호출 1회 별도 기록 → 워밍업 10회 → 반복 50회 | 동일 | 첫 호출·워밍업은 통계에서 제외 |
| Mean, Median, P95, Std | 동일 + Min/Max/CV | 표본 표준편차, P95는 선형 보간 |
| Batch 1, 2, 4, 8, 16, 32, 여유 시 확대 | 1~32 × 3회차 후 64·128·256 | 확장 배치는 직전 배치가 성공한 모드만 시도 |
| Latency, Throughput, Speedup, Scaling Efficiency | `batch_scaling.csv` | 효율 = `(처리량(B)/처리량(1))/B` |
| 미지원 배치는 오류 로그 저장 후 계속 | `errors/`, `failures.csv`, `coverage.csv` | 한 번 실패·timeout한 (모드, 배치)는 재시도하지 않음 |
| NPU 입력 미리 준비, forward부터 `.cpu()` 완료까지 | 기준 범위 `forward_to_cpu` | 타이머 직전에 stream 동기화 |
| H2D/D2H 별도 측정 | 별도 루프 50회: H2D \| forward+동기화 \| D2H | 기준 통계와 섞지 않음 |
| Eager/Compile 출력 일치 검증 | `correctness.csv` | 합격 기준은 정의되지 않아 수치만 보고 |
| 조건마다 새 프로세스 3회, 캐시는 매번 새 경로 | 동일 | 이미 파일이 있는 캐시 경로는 worker가 거부 |
| CSV, 그래프, Markdown 보고서 | 표 9종, 그림 7종, 이 문서 | 그림은 CSV만 읽는 별도 스크립트 |

## 3. Phase 1 분석

### 3.1 무엇이 측정됐나

Phase 1은 원본 모델의 BatchNorm 미지원, folding 후 `layer4.*.conv2`의 `LowLevelTailCrop` 한도 초과를 차례로 우회해 `bn_folded_output_split128` 변형을 만들었다. 전체 모델 grid(배치 1·4·8·16, mixed, cold/reuse)는 사용자 요청으로 중단돼 **5개 job만 성공**했다. 아래는 워밍업 3회 뒤 10회 반복의 중앙값이다.

| 조건 (새 캐시) | Eager | Compile | 배속 |
| --- | ---: | ---: | ---: |
| 배치 1 forward (ms) | 16.100 | 6.679 | 2.41 |
| 배치 1 E2E, H2D·D2H 포함 (ms) | 16.393 | 7.069 | 2.32 |
| 배치 4 forward (ms) | 미측정 (첫 호출 중 중단) | 7.334 | — |
| 배치 4 E2E (ms) | 미측정 | 7.811 | — |
| 배치 1 첫 호출 (s) | 456.7 | 418.6 | — |
| 배치 4 첫 호출 (s) | 450초 경과 시점에 중단 | 1172.4 | — |
| 배치 1, 같은 캐시 재사용 시 첫 호출 (s) | 14.0 | 3.4 | — |

배치 1의 Eager↔Compile 출력은 최대 절대오차 0.0156, Top-1 일치였다(합성 입력 1장).

### 3.2 Phase 2 설계에 반영한 관찰

1. **Compile은 배치 4에서 지연이 거의 늘지 않았다.** forward 6.68 → 7.33 ms(+10%)로 이미지는 4배다. 처리량으로는 150 → 545 img/s다. Eager 쪽 배치 4 값이 없어 배속의 배치 의존성은 Phase 1에서 알 수 없었다. Phase 2의 핵심 질문이 이것이다.
2. **첫 호출 비용이 크고 배치에 따라 커진다.** Compile 첫 호출이 배치 1에서 약 7분, 배치 4에서 약 20분이다. 조건마다 새 캐시를 쓰는 Phase 2는 측정보다 컴파일에 시간이 든다. 그래서 회차 우선 순서, job별 timeout(7,200초), 누적 시간 예산, 재개 기능을 넣었다.
3. **Eager는 매 forward마다 kernel을 다시 load한다.** Phase 1 관측 기록에서 Eager의 고유 kernel은 42개인데 SDK의 `_compile_and_load_cached` LRU는 32칸이고, 워밍업 이후에도 forward 1회마다 miss가 42회씩 늘었다(24회 호출에 1,008회). 이번 스모크에서도 반복 50회 동안 forward당 42.0회였다. 이 재load가 Eager 지연의 얼마를 차지하는지는 측정하지 않았다. SDK 기본값을 유지하므로 Phase 2에서도 같은 조건이며, worker가 회차마다 miss 수를 기록한다.
4. **모듈별 독립 계측(배치 1)에서는 133개 지점 모두 Compile이 빨랐다.** 지점별 중앙값의 중앙값이 Eager 0.154 ms, Compile 0.100 ms, 독립 배속 중앙값 1.50(1.14~4.16)이다. 독립 실행 값이라 전체 모델 지연으로 합산할 수 없지만, 전체 모델 배속(2.4)이 지점별 배속보다 크다는 점은 그래프 전체를 한 번에 컴파일하는 효과가 따로 있음을 시사한다.

### 3.3 Phase 1의 한계

| 한계 | Phase 2의 대응 |
| --- | --- |
| 반복 10회, 워밍업 3회, 조건당 프로세스 1개 | 워밍업 10회, 반복 50회, 조건당 새 프로세스 3개(배치 1은 5개) |
| P95 없음 | Mean / Std / Median / P95 |
| 배치 2·8·16·32 없음, 배치 4는 Compile만 | 배치 1~32 양 모드, 이후 64 이상 |
| E2E 한 값에 H2D·D2H 포함 | 기준 범위와 전송 구간을 따로 측정 |
| 샘플마다 CSV·JSON을 기록 | 반복 루프 안에서는 파일 I/O 없음 |

## 4. 코드 구조와 RNGD API 확인

Phase 1 파일은 수정하지 않았다. 새 코드는 `Phase_2/`와 `tests/test_phase2.py`뿐이다.

| 파일 | 역할 | 재사용한 기존 코드 |
| --- | --- | --- |
| `config.json` | 배치·회차·반복 수·timeout·예산 | — |
| `worker.py` | (모드, 배치, 회차) 1개를 새 프로세스에서 측정 | `models.load_model / make_input`, `runtime.make_callable / synchronize / eager_cache_stats` |
| `measure.py` | 측정 범위 정의 | `runtime.synchronize` |
| `run.py` | 순차 실행, timeout, 실패 후 계속, 재개, 진행 기록 | `Phase_1.run.launch`, `environment.collect`, `.rngd-benchmark.lock` |
| `analyze.py`, `stats.py` | 원시값 → 표 | `analysis.compare_tensors` |
| `plots.py` | 표 → 그림 | — |
| `preflight.py` | 환경 준비 점검 | — |

설치된 furiosa-torch 2026.4.0에서 직접 확인한 항목이다.

| 사용한 API | 확인 결과 |
| --- | --- |
| `ft.config.set_cpu_fallback(False)` / `get_cpu_fallback()` | 존재, worker가 False임을 확인하고 아니면 중단 |
| `ft.config.cache_context(path)` | 존재. 새 경로에서 실제 컴파일이 일어나고(Phase 1: 419초), 같은 경로 재사용 시 3.3초 |
| `ft.current_stream(device).synchronize()` / `.query()` | 존재. `.to("cpu")` 반환 직후 `query()`가 매번 True여서, 출력 복사가 반환되면 stream이 비어 있음을 확인 |
| `torch.compile(model, backend=ft.backend)` | 존재. 반복 중 backend 재진입 0회 |
| `ft.get_device_configuration()` | `Fusion(num_fusion=1, num_pe_per_cluster=1, …)`, 즉 PE 1개 구성. 변경하지 않음 |
| Eager LRU 통계 (`furiosa.torch._backend.aten`) | 비공개 API, 읽기만 함. `set_eager_cache_size`는 존재하지만 기본값을 바꾸지 않음 |

사양에서 요구했지만 SDK가 지원하지 않아 다른 것으로 바꾼 기능은 없다.

## 5. 실제 NPU 실행분과 미실행분

| 구분 | 내용 | 상태 |
| --- | --- | --- |
| 코드 경로 스모크 | 배치 1, Eager·Compile 각 1회. **Phase 1 캐시의 사본**을 사용 | NPU 실행함. 새 캐시 조건이 아니므로 Phase 2 결과가 아님 |
| 첫 본 실험 시도 `aborted-20261119-094632-missing-libc-headers` | `eager-b1-t1`이 9.6초에 컴파일 오류, `compile-b1-t1`은 중단 | 지연 샘플 0개 |
| 환경 사전 점검 | 작은 BF16 Conv2d를 빈 캐시에서 Eager·Compile로 컴파일·실행 | NPU 실행함. 지연 측정 아님 |
| **본 실험 58개 job** (배치 1~32 × 2모드 × 3회차 36개, 배치 1 추가 세트 4개, 배치 64·128·256 확장 18개) | — | **전부 미실행** |
| 그래프 | 레이아웃만 합성값으로 확인(결과 폴더 밖) | 실제 그림 없음 |

스모크 실행의 반복 50회 값이다. 코드가 Phase 1과 같은 수준의 값을 내는지 보는 용도이며, 새 캐시·3회차 조건을 충족하지 않는다.

| 배치 1, 캐시 사본 | Eager | Compile |
| --- | ---: | ---: |
| `forward_to_cpu` 평균 ± 표준편차 (ms) | 16.278 ± 0.108 | 6.784 ± 0.010 |
| 중앙값 / P95 (ms) | 16.256 / 16.468 | 6.785 / 6.797 |
| forward + 동기화 중앙값 (ms) | 16.292 | 6.709 |
| H2D / D2H 평균 (ms) | 0.220 / 0.176 | 0.236 / 0.219 |
| E2E 중앙값 (ms) | 16.686 | 7.169 |
| 평균 기준 배속 | 2.40 | |
| 유휴 stream 동기화 1회 (ms) | 0.026 | 0.026 |

Phase 1 대비 forward 중앙값은 Eager +1.2%, Compile +0.4%다. 장치 표기가 `npu1`에서 `npu0`으로 바뀌었고 컨테이너도 다르다. Eager↔Compile 출력은 최대 절대오차 0.0156, Top-1 일치로 Phase 1과 같다. 원본: [evidence/smoke-phase1-cache-copy](evidence/smoke-phase1-cache-copy).

## 6. 오류와 조치

| 항목 | 내용 |
| --- | --- |
| 증상 | 새 캐시의 첫 호출에서 `CompileError`: 생성된 C 코드 컴파일 중 `pe.h:4: fatal error: assert.h: No such file or directory` |
| 원인 | 컨테이너가 2026-11-18 18:31 UTC에 재생성됐고, `gcc-aarch64-linux-gnu`는 설치됐지만 그 권장 패키지 `libc6-dev-arm64-cross`가 빠졌다. 캐시가 있으면 컴파일을 건너뛰므로 스모크에서는 드러나지 않았다 |
| 영향 | 모드·배치와 무관하게 모든 신규 컴파일 실패. SDK의 기능 미지원이 아니라 호스트 환경 문제 |
| 조치 | 사용자 승인 후 `apt-get install -y libc6-dev-arm64-cross` (함께 설치: `linux-libc-dev-arm64-cross`). [설치 로그](evidence/toolchain-libc-headers-install.log) |
| 확인 | [사전 점검](evidence/preflight.json) 7개 항목 통과: 헤더, 패키지 버전, 장치, 모델 변형 SHA256, 빈 캐시 Conv 컴파일(Eager 1.1초, Compile 0.9초, CPU BF16 대비 오차 0), 그림용 인터프리터 |
| 재발 방지 | runner가 시작 전에 헤더를 확인하고 없으면 실행하지 않는다. **컨테이너가 다시 만들어지면 패키지를 다시 설치해야 한다** |

전체 ResNet50의 새 캐시 컴파일은 이 컨테이너에서 아직 끝까지 돌려 보지 않았다. 본 실험의 첫 job이 그 확인을 겸한다.

## 7. 본 실험 실행 방법

```bash
cd /root
# 0) 환경 점검 (약 20초, 컨테이너가 바뀌었으면 필수)
XDG_CACHE_HOME=/root/Resnet/.cache /root/venv/bin/python -m Resnet.Phase_2.preflight

# 1) 본 실험: 터미널과 분리해 실행
RUN=Resnet/Phase_2/results/batch-scaling-$(TZ=Asia/Seoul date +%Y%m%d-%H%M%S)
XDG_CACHE_HOME=/root/Resnet/.cache TMPDIR=/root/Resnet/.tmp \
  setsid nohup /root/venv/bin/python -u -m Resnet.Phase_2.run --run-dir $RUN \
  > $RUN.console.log 2>&1 < /dev/null &

# 2) 진행 확인
cat $RUN/PROGRESS.md

# 3) 중단됐을 때 이어서 실행 / 예산 변경
/root/venv/bin/python -u -m Resnet.Phase_2.run --run-dir $RUN --resume --max-total-s 86400

# 4) 그림
/root/ViT/vit_pilot/.plot-venv/bin/python Resnet/Phase_2/plots.py $RUN
```

소요 시간은 측정해 봐야 안다. 알려진 값은 Phase 1의 첫 호출뿐이다(배치 1 양 모드 약 7분, Compile 배치 4 약 20분). 배치 8 이상의 컴파일 시간은 모른다. 기본 예산은 worker 누적 12시간(`max_total_s`)이고, 회차 우선 순서라 예산이 먼저 끝나도 완료된 회차는 온전한 배치 스윕이다. 표는 job이 끝날 때마다 갱신된다.

## 8. 결과와 분석

본 실험 `results/batch-scaling-fusion4-pe4-7-20261119-124115` (PE 4 fusion, `furiosa:1` = 물리 PE 4–7). 58개 계획 중 성공 35 / 실패 4 / 건너뜀 19, worker 누적 41,290 s. 근거: `batch_scaling.csv`, `condition_summary.csv`, `revalidation_b1.csv`, `initial_costs.csv`, `correctness.csv`, `failures.csv`, `figures/` 7종. 지연은 `forward_to_cpu`(입력을 NPU에 미리 올린 뒤 출력 `.to("cpu")` 반환까지), 회차별 반복 50회를 모은 값이다.

### 8.1 배치 스케일링

| 배치 | Eager 평균 (ms) | Compile 평균 ± SD (ms) | 배속 E/C | Compile 처리량 (img/s) | 처리량 배수 (vs b1) | 스케일링 효율 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12.94 | 2.057 ± 0.018 | 6.29 | 486 | 1.0 | 1.00 |
| 2 | 실패 | 실패 | – | – | – | – |
| 4 | 13.68 | 2.453 ± 0.021 | 5.58 | 1,631 | 3.4 | 0.84 |
| 8 | 14.27 | 2.819 ± 0.021 | 5.06 | 2,837 | 5.8 | 0.73 |
| 16 | 실패 | 3.833 ± 0.018 | – | 4,174 | 8.6 | 0.54 |
| 32 | 실패 | 5.589 ± 0.020 | – | 5,725 | 11.8 | 0.37 |
| 64 | 건너뜀 | 9.475 ± 0.016 | – | 6,755 | 13.9 | 0.22 |
| 128 | 건너뜀 | 17.33 ± 0.03 | – | 7,387 | 15.2 | 0.12 |
| 256 | 건너뜀 | 34.99 ± 0.12 | – | 7,317 | 15.1 | 0.06 |

배치 1은 이 표에서 회차 2·3만 쓴다(회차 1은 다른 설정의 1시간 실행에서 돌았다). 배치 1의 5세트 재검증은 8.3절.

1. **배속은 줄어든다.** Eager가 배치 1→8에서 12.9→14.3 ms(+10%)로 거의 그대로인 동안 Compile은 2.06→2.82 ms(+37%)로 늘어, 배속이 6.3→5.1배로 떨어졌다. BERT(9.3→1.4배)보다 완만하지만 방향은 같다. 배치 16 이상의 Eager는 8.5절의 수정 변형으로 따로 쟀다.
2. **Compile 처리량은 배치 128에서 포화**한다(7,387 img/s, 256에서 7,317). 배치 64→128에서 지연이 1.83배, 128→256에서 2.02배로, 128 이상은 지연이 배치에 정비례한다.
3. **지원 최대 배치**: Compile은 2를 뺀 256까지 통과. Compile b2는 그래프 전체 컴파일에서 `On populating O4594: Option::unwrap() on None` panic. Eager는 원래 1·4·8만 통과했다(원인과 수정은 8.5절, NOTION_REPORT.md).
4. **첫 호출 비용**: Compile 499 s(b1) → 891 → 939 → 1,031 → 1,422 → 1,777 → 2,190 → 2,457 s(b256). Eager는 460 → 491 → 573 s(b1·4·8). 회차 간 차이는 ±3% 이내.
5. **H2D 비중** (E2E 중앙값 대비, Compile): b1 9% → b32 27% → b128 32% → b256 31%. 배치가 크면 입력 전송이 forward의 약 절반 크기가 되므로, 처리량 수치는 입력이 이미 NPU에 있다는 가정에 기댄다. Eager에서는 b8에서도 4%다. D2H는 0.16–0.49 ms로 작다.

### 8.2 안정성
- Compile 회차 간 CV < 1.1%, 같은 모드의 회차 간 출력은 비트 단위로 같다(`correctness.csv`의 `same_mode_across_trials`).
- 실패 4건은 모두 첫 호출의 `UnsupportedOpError: failed to compile`이다(`failures.csv`, `errors/`). 실패한 조건은 이후 회차를 재시도하지 않는다.

### 8.3 배치 1 재검증 (5세트)
| 모드 | 세트 2–5 평균 (ms) | 묶음 평균 ± SD (ms) | Phase 1 forward (1 PE, n=10) |
| --- | --- | ---: | ---: |
| Eager | 13.06 / 12.83 / 12.92 / 13.13 | 12.98 ± 0.15 | 16.10 |
| Compile | 2.05 / 2.07 / 2.05 / 2.02 | 2.047 ± 0.021 | 6.68 |

세트 간 범위는 Eager 2.3%, Compile 2.2%로 Phase 1 결론(Compile이 배치 1에서 여러 배 빠름)을 재현한다. Phase 1은 1 PE라 절대값은 직접 비교하지 않는다. 같은 컨테이너의 1 PE 측정(Eager 16.35 / Compile 6.78 ms)은 Phase 1과 1–2% 차이다.

### 8.4 출력 일치
Eager↔Compile max abs error 0.0156(b1)–0.0313(b4·8), cosine ≥ 0.99999, Top-1 전부 일치. CPU BF16 기준(Phase 1 방식, b4)과도 max abs error 0.031, Top-1 일치. 합성 입력이라 정확도 주장은 하지 않는다.

### 8.5 Eager 수정 변형 (배치 분할) — 배치 2·16·32·64·128

**원인.** Eager는 aten op마다 단독 프로그램으로 컴파일한다. SDK 2026.4.0 컴파일러가 특정 (op, shape)을 거부해 forward 전체가 멈췄다. 모든 op을 컴파일만 해 보는 진단(`eager_op_probe.py`, 배치 8 대조군 44/44 통과, NPU 실제 에러와 일치)으로 고유 op 44개 중 b2 3개, b16 3개, b32 5개, b64 9개, b128 9개의 실패를 찾았다. 에러는 `insufficient sequencer entries: required 9`, `On populating O25` panic(maxpool b2), `TransposedLower … lowering failed`(b32 이상의 add·relu·conv·avgpool), `attempt to read incomplete tensor buffer`(b64 stem relu), DMA `completion_ids should not be empty`(b128 layer1·2 conv3), `BridgeTacticFailure`(b128 add)다. 실패는 배치에 단조롭지 않다(b2 실패, b4·8 통과). 출력 채널 분할로는 해결되지 않았다.

**수정.** 실패 op을 포함하는 가장 작은 모듈(conv, pool, 잔차 블록)만 배치 축으로 잘라 실행 후 concat한다(`partitioning.partition_batch`, 계획 `eager_batch_partition.json`). 배치 16 이상은 8씩, 배치 2는 1씩 자른다. 이 크기들은 전체 모델이 NPU에서 통과한 배치다. 분할 모듈 수는 b2 10, b16 15, b32 10, b64 11, b128 18개다. 인스턴스 `forward`만 바꾸므로 state SHA256(`f4ffc872…`)이 같고, worker가 이를 다시 확인한다. Eager에만 적용하며 변형명은 `bn_folded_output_split128+eager_batch_chunkN`이다. 수정 모델은 진단에서 다섯 배치 모두 op 100% 컴파일됐다.

**측정.** `results/eager-fix-fusion4-pe4-7-2h-20261120`: 2시간 제한 때문에 **배치당 1회차**(새 프로세스·새 캐시, 첫 호출 → 워밍업 10 → 반복 50), `furiosa:1`, 5개 모두 성공, worker 5,160 s. 1회차라 회차 간 편차는 없다. Compile 열은 8.1절의 3회차 평균이다.

| 배치 | Eager 평균 (ms) | 중앙값 / P95 (ms) | 처리량 (img/s) | Compile (ms) | 배속 E/C | 첫 호출 (s) | op 호출/forward |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 (원 모델, 참고) | 14.27 | 14.24 / 14.51 | 560 | 2.82 | 5.1 | 573 | 130 |
| 2 | 16.89 | 16.87 / 17.02 | 118 | 실패 | – | 416 | 140 |
| 16 | 32.69 | 30.01 / 30.24 | 489 | 3.83 | 8.5 (중앙값 7.8) | 851 | 145 |
| 32 | 62.71 | 62.71 / 62.93 | 510 | 5.59 | 11.2 | 1,301 | 271 |
| 64 | 120.42 | 120.42 / 120.57 | 531 | 9.48 | 12.7 | 1,273 | 662 |
| 128 | 300.06 | 300.04 / 300.48 | 427 | 17.33 | 17.3 | 1,233 | 1,375 |

- b16 평균은 반복 43회차의 이상치 1개(155 ms)를 포함한다. 나머지 49개는 29.9–38.5 ms다.
- b16 첫 호출 중 약 2.5분 동안 8.6절 진단을 `furiosa:0`·별도 CPU에서 같이 돌렸다. 측정 단계(워밍업·반복)와는 겹치지 않았다.
- **출력**: Compile 출력(같은 배치 t1) 대비 b16–128 max abs error 0.031, cosine ≥ 0.99999, Top-1 100%. b2는 CPU BF16 대비 max abs error 0.031, cosine 0.999995, Top-1 2/2.
- **해석**: 수정 변형의 Eager 처리량은 490–530 img/s로 원 모델 b8(560 img/s)을 넘지 못하고, b128에서는 427 img/s로 떨어진다. 지연이 forward당 op 호출 수에 따라 늘어난다(b8 130회 → b128 1,375회). 배치를 키워도 Eager는 이득이 없고, 배속은 b16 7.8배에서 b128 17.3배로 벌어진다. 이 수치는 SDK 2026.4.0의 컴파일 실패를 우회한 변형의 값이며, 원 모델 Eager의 성능으로 읽으면 안 된다.

### 8.6 Eager 지연이 배치 1–8에서 약 13 ms로 일정한 원인 (진단)

원 모델 Eager는 forward마다 aten op을 130회 호출하고, 그중 42회가 SDK의 로드 캐시(LRU) miss다. 고유 op은 44개인데 `_compile_and_load_cached`가 `lru_cache(maxsize=32)`로 고정돼 있기 때문이다. 문서상 기본값(`FURIOSA_TORCH_EAGER_NUM_LOAD`=2048)은 import 시 적용되지 않는다. `eager_lru_ab.py`가 한 프로세스에서 로드 캐시 크기를 32 → 2048 → 32 → 2048로 바꿔 가며 반복 50회씩 쟀다(복사한 캐시, Phase 2 측정 아님).

| 조건 | b1: 32 / 2048 (ms) | b8: 32 / 2048 (ms) | miss/forward |
| --- | --- | --- | --- |
| b16 컴파일과 동시, `furiosa:0` | 17.32·17.41 / 12.48·12.55 | 18.00·17.92 / 13.94·13.92 | 42 / 0 |
| 조용한 머신, `furiosa:1` | 18.07·18.16 / 13.04·13.18 | 18.30·18.58 / 14.23·14.16 | 42 / 0 |

- 이 진단 안에서는 miss를 없애면 forward당 약 5.0 ms(b1)·4.2 ms(b8)가 줄었다.
- 그러나 **새 캐시로 측정 worker를 그대로 돌린 b1 재확인**(`results/eager-b1-recheck-20261120`, 2026-11-20)은 miss 42회/forward 상태에서 **14.07 ms**였다. 본 실험(11-19)은 12.98 ms다. 복사한 캐시를 쓰는 진단의 32칸 값(17.5–18.2 ms)만 유독 높아, 그 5 ms는 진단 조건에서 생긴 비용으로 보인다(원인 미확인).
- 측정 조건에서 miss를 없앤 값은 직접 재지 않았다. 진단의 2048칸 값(b1 13.0 ms)과 재확인 값(14.07 ms)을 비교하면, LRU 재로드의 몫은 **많아야 약 1 ms**로 추정된다.
- **결론**: 배치 1–8에서 약 13 ms로 일정한 지연의 대부분은 op 130회의 호스트 디스패치다. NPU 계산(Compile b1 2.06 ms)을 빼면 약 11 ms가 남는다. 8.5절에서 op 호출 수에 비례해 지연이 늘어난 것도 같은 방향이다. 하루 사이 b1 Eager가 12.98 → 14.07 ms(+8%)로 달라졌으므로, Eager 절대값은 날짜 간 비교에 주의한다.
