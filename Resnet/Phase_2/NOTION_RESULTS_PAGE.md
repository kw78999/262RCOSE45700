# ResNet50 on RNGD — Eager vs Compile · PE Fusion 실험 결과 (Phase 2)

> 기록 기준: 2026-11-25 20:20 KST · PPT 작성용 정리. PE fusion 매트릭스 18개 조건 모두 측정 완료.

## 0. 핵심 메시지 (슬라이드 1장용)
- **Compile이 Eager보다 배치 1에서 6배 이상 빠르고, 배치를 키우면 격차가 더 벌어진다.** (PE 4 fusion 기준 배치 1: 6.3배, 배치 128: 17배)
- **Compile은 PE fusion에 따라 거의 선형으로 빨라진다.** PE 1→4개에서 forward 3.1–3.9배 단축(배치 64에서 3.93배). **Eager는 조금만 빨라진다**(배치 1: 16.3 → 13.1 ms, 1.24배).
- **Eager는 배치 1–8에서 지연이 약 13–14 ms로 거의 일정**하다. 원인은 op 130개를 호스트에서 하나씩 실행하는 비용(약 11 ms)이다.
- **Eager는 SDK 2026.4.0에서 배치 2·16 이상이 컴파일 실패**한다. 실패 op만 배치로 나눠 돌리는 변형으로 실행은 되지만, 처리량은 약 530 img/s에서 더 늘지 않는다.
- **Compile 처리량은 배치 128 부근에서 약 7,400 img/s로 포화**하고, 큰 배치에서는 입력 전송(H2D)이 E2E의 약 30%를 차지한다.

## 1. 실험 환경
| 항목 | 값 |
| --- | --- |
| NPU | Furiosa RNGD 1장 (npu0, PE 8개), firmware 2026.4.0 |
| SDK | furiosa-torch / kernels 2026.4.0 (rev 5800e2ba24) |
| SW | Python 3.12.3, torch 2.10.0+cpu, torchvision 0.25.0 |
| 모델 | ResNet50 IMAGENET1K_V2, BN folding + layer4 conv2 출력 128채널 분할 (`bn_folded_output_split128`), BF16 |
| 옵션 | CPU fallback 끔, SDK 기본 compiler 설정, `torch.compile(model, backend=furiosa)` |
| 입력 | synthetic normal `[B,3,224,224]` (정확도 평가 아님) |
| 호스트 | AMD EPYC 9354, 컨테이너 허용 CPU 22개, 메모리 한도 160 GiB |

## 2. 측정 방법
- 조건(모드·배치·PE fusion)마다 **새 프로세스 + 새 compiler 캐시**
- 첫 호출 1회(컴파일·로드 포함, 별도 기록) → 워밍업 10회 → 반복 50회 → 구간 분리 루프(워밍업 5 + 50회)
- 지표
  - **forward_to_cpu** (본 실험 기준 지연): 입력을 NPU에 미리 올린 뒤 forward ~ 출력 `.to("cpu")` 반환까지
  - **Forward** (PE fusion 매트릭스 기준): 입력이 NPU에 있는 상태에서 forward + stream 동기화, 전송 제외
  - **H2D / D2H / E2E**: 구간 분리 루프에서 각각 동기화로 닫아 측정
- 처리량 = 배치 × 1000 / 평균 지연(ms)
- ⚠ 본 실험(4절)과 PE fusion 매트릭스(7절)는 **지연 지표가 다르다**(forward_to_cpu는 D2H 포함). 같은 표에 섞지 말 것.

## 3. 실험 구성 (무엇을 돌렸나)
| 실험 | 내용 | 결과 위치 |
| --- | --- | --- |
| A. 본 실험 | PE 4 fusion, Eager·Compile × 배치 1–256, 3회차, 배치 1은 5세트 재검증 | `results/batch-scaling-fusion4-pe4-7-20261119-124115` |
| B. Eager 수정 변형 | 실패 op만 배치 분할한 Eager, 배치 2·16·32·64·128, 1회차 | `results/eager-fix-fusion4-pe4-7-2h-20261120` |
| C. Eager 지연 원인 진단 | SDK 로드 캐시 크기 A/B, 새 캐시 재확인 | `results/eager-lru-ab-*`, `eager-b1-recheck-20261120` |
| D. PE fusion 매트릭스 | PE fusion 1·2·4 × Eager·Compile × 배치 1·16·64 = 18조건, 1회차 | `results/matrix-f*-20261125`, 표: `fusion-matrix-20261125/MATRIX_REPORT.md` |

## 4. 결과 A — Eager vs Compile 배치 스케일링 (PE 4 fusion, forward_to_cpu 평균 ms)
| 배치 | Eager | Compile | Eager/Compile | Compile 처리량 (img/s) | Compile 첫 호출 (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12.94 | 2.06 | 6.3× | 486 | 499 |
| 2 | 실패 | 실패 | – | – | – |
| 4 | 13.68 | 2.45 | 5.6× | 1,631 | 891 |
| 8 | 14.27 | 2.82 | 5.1× | 2,837 | 939 |
| 16 | 30.01* | 3.83 | 7.8×* | 4,174 | 1,031 |
| 32 | 62.71* | 5.59 | 11.2×* | 5,725 | 1,422 |
| 64 | 120.42* | 9.48 | 12.7×* | 6,755 | 1,777 |
| 128 | 300.06* | 17.33 | 17.3×* | 7,387 | 2,190 |
| 256 | – | 34.99 | – | 7,317 | 2,457 |

\* Eager 수정 변형(결과 B), 1회차. b16은 이상치 1회(155 ms)가 있어 중앙값.

- **Compile**: 회차 간 CV < 1.1%, 같은 모드 회차 간 출력 비트 단위 동일. 배치 64→128 지연 1.83배, 128→256 2.02배 → **배치 128 이상은 지연이 배치에 정비례(포화)**
- **Eager(원 모델)**: 배치 1→8에서 12.9→14.3 ms(+10%)로 거의 일정, Compile은 +37% → **배속이 6.3→5.1배로 줄어듦**
- **Eager(수정 변형)**: 배치 16 이상에서 지연이 배치에 비례 이상으로 커져 처리량 427–531 img/s, 원 모델 배치 8(560 img/s)을 넘지 못함 → 배속이 다시 7.8→17.3배로 벌어짐
- **H2D 비중(Compile, E2E 대비)**: 배치 1 9% → 32 27% → 128 32% → 256 31%. 처리량 수치는 "입력이 이미 NPU에 있음" 가정에 기댄다
- **첫 호출(컴파일)**: Compile 499 s(b1) → 2,457 s(b256). Eager 460–573 s(b1–8)
- **배치 1 재검증(5세트)**: Eager 12.98 ± 0.15 ms, Compile 2.047 ± 0.021 ms, 세트 간 범위 약 2%
- **출력 일치**: Eager↔Compile max abs error 0.016–0.031(BF16), cosine ≥ 0.99999, Top-1 전부 일치

## 5. 결과 B — Eager 배치 2·16 이상 실패와 수정
**원인**: Eager는 aten op마다 단독 프로그램으로 컴파일한다. SDK 컴파일러가 특정 (op, shape)을 거부하면 forward 전체가 멈춘다. 같은 모델이 Compile(그래프 전체 컴파일)에서는 배치 256까지 통과한다.

| 배치 | 실패 op (고유 44개 중) | 대표 에러 | 실패 위치 |
| ---: | :-: | --- | --- |
| 2 | 3 | `On populating O25: Option::unwrap() on None` (panic), `insufficient sequencer entries: required 9` | maxpool, layer3 conv2, layer4.0 conv2 |
| 16 | 3 | `insufficient sequencer entries: required 9` | layer3 conv3·conv2, layer4.0 conv2 |
| 32 | 5 | + `TransposedLower … lowering failed` | + layer3 블록 add·relu |
| 64 | 9 | `TransposedLower…`, `attempt to read incomplete tensor buffer` | stem relu, layer3·4, avgpool |
| 128 | 9 | + `BridgeTacticFailure`, DMA `completion_ids should not be empty` | + layer1·2 conv3 |

- 실패가 배치에 단조롭지 않다(배치 2 실패, 4·8 통과, 16부터 다시 실패). 14×14·7×7 해상도(layer3·4)에 몰림
- PE 수가 적을수록 실패가 많다: PE 2 fusion은 배치 8·4·2에서도 실패 op 존재, PE 1은 배치 2부터 실패
- **진단 방법**: 모든 op을 컴파일만 해 보는 도구(`eager_op_probe.py`). 배치 8 대조군 44/44 통과, 에러 문구가 실제 NPU 실행과 일치
- **수정**: 실패 op을 포함하는 가장 작은 모듈만 배치 축으로 잘라 실행 후 concat. 분할 크기 = 그 PE fusion에서 전체 모델이 통과한 배치(PE 4: 8, PE 1·2: 1). 가중치·state SHA256 그대로, 출력 Top-1 100% 일치
- 출력 채널 분할(Phase 1 방식)로는 해결되지 않았다

## 6. 결과 C — Eager 지연이 배치 1–8에서 약 13 ms로 일정한 원인
- forward당 aten op 호출 **130회**, 그중 42회가 SDK 로드 캐시 miss(캐시 32칸 < 고유 op 44개, `lru_cache(maxsize=32)` 고정값)
- 로드 캐시를 2048칸으로 늘리면 miss 0, 배치 1 13.0 ms. 새 캐시로 다시 잰 원래 설정은 14.07 ms → **재로드 몫은 많아야 약 1 ms**
- 나머지 약 11 ms(NPU 계산 약 2 ms 제외)는 **op별 호스트 디스패치 비용**. 수정 변형에서 op 호출 수(130 → 1,375회)에 따라 지연이 커진 것과도 일치
- 같은 조건 배치 1 Eager가 날짜에 따라 12.98 / 14.07 ms(+8%) → Eager 절대값은 날짜 간 비교 주의

## 7. 결과 D — PE fusion 매트릭스 (18조건 완료, Forward = 전송 제외, 평균 ms, 1회차)
| PE fusion | 모드 | 배치 1 | 배치 16 | 배치 64 |
| ---: | --- | ---: | ---: | ---: |
| 1 (PE 1개) | Compile | 6.55 | 11.65 | 36.65 |
| 2 (PE 2개) | Compile | 3.50 | 6.32 | 16.91 |
| 4 (PE 4개) | Compile | 1.98 | 3.77 | 9.32 |
| 1 | Eager | 16.27 | 224.03‡ | 831.05‡ |
| 2 | Eager | 14.01 | 105.40‡ | 611.67‡ |
| 4 | Eager | 13.08 | 30.25‡ | 120.96‡ |

**처리량 (Forward 기준, img/s)**
| PE fusion | Compile b1 | Compile b16 | Compile b64 | Eager b1 | Eager b16 | Eager b64 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 153 | 1,374 | 1,746 | 61 | 71‡ | 77‡ |
| 2 | 286 | 2,532 | 3,784 | 71 | 152‡ | 105‡ |
| 4 | 505 | 4,249 | 6,870 | 76 | 529‡ | 529‡ |

**구간별 breakdown (평균 ms, 괄호는 E2E 대비 비중)**
| PE fusion | 모드 | 배치 | H2D | Forward | D2H | E2E |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Compile | 1 | 0.233 (3%) | 6.55 (94%) | 0.210 (3%) | 6.99 |
| 1 | Compile | 16 | 1.175 (9%) | 11.65 (89%) | 0.292 (2%) | 13.11 |
| 1 | Compile | 64 | 2.642 (7%) | 36.65 (92%) | 0.370 (1%) | 39.66 |
| 2 | Compile | 1 | 0.223 (6%) | 3.50 (90%) | 0.182 (5%) | 3.91 |
| 2 | Compile | 16 | 1.042 (14%) | 6.32 (83%) | 0.223 (3%) | 7.58 |
| 2 | Compile | 64 | 3.841 (18%) | 16.91 (80%) | 0.306 (1%) | 21.06 |
| 4 | Compile | 1 | 0.211 (9%) | 1.98 (84%) | 0.163 (7%) | 2.35 |
| 4 | Compile | 16 | 1.141 (22%) | 3.77 (73%) | 0.227 (4%) | 5.13 |
| 4 | Compile | 64 | 4.301 (31%) | 9.32 (67%) | 0.244 (2%) | 13.86 |
| 1 | Eager | 1 | 0.225 (1%) | 16.27 (98%) | 0.177 (1%) | 16.67 |
| 1 | Eager | 16‡ | 1.152 (1%) | 224.03 (99%) | 0.252 (0%) | 225.43 |
| 1 | Eager | 64‡ | 2.634 (0%) | 831.05 (100%) | 0.280 (0%) | 833.97 |
| 2 | Eager | 1 | 0.206 (1%) | 14.01 (97%) | 0.160 (1%) | 14.38 |
| 2 | Eager | 16‡ | 0.800 (1%) | 105.40 (99%) | 0.254 (0%) | 106.45 |
| 2 | Eager | 64‡ | 4.185 (1%) | 611.67 (99%) | 0.280 (0%) | 616.14 |
| 4 | Eager | 1 | 0.191 (1%) | 13.08 (97%) | 0.164 (1%) | 13.43 |
| 4 | Eager | 16‡ | 1.123 (4%) | 30.25 (96%) | 0.208 (1%) | 31.58 |
| 4 | Eager | 64‡ | 4.768 (4%) | 120.96 (96%) | 0.256 (0%) | 125.99 |

**PE fusion 효과와 Eager/Compile 격차 (Forward 평균 비율)**
| 비교 | 배치 1 | 배치 16 | 배치 64 |
| --- | ---: | ---: | ---: |
| Compile PE 1 → 2 | 1.87× | 1.84× | 2.17× |
| Compile PE 2 → 4 | 1.77× | 1.68× | 1.81× |
| Compile PE 1 → 4 | 3.31× | 3.09× | 3.93× |
| Eager PE 1 → 4 | 1.24× | 7.41×‡ | 6.87×‡ |
| Eager/Compile PE 1 | 2.48× | 19.2×‡ | 22.7×‡ |
| Eager/Compile PE 2 | 4.00× | 16.7×‡ | 36.2×‡ |
| Eager/Compile PE 4 | 6.61× | 8.0×‡ | 13.0×‡ |

**Eager 배치 분할 변형 구성** (컴파일 실패 op만 배치로 잘라 실행)
| PE fusion | 분할 크기 | op 호출/forward (b16 / b64) |
| ---: | ---: | --- |
| 1 | 1 | 730 / 5,170 |
| 2 | 1 | 835 / 4,855 |
| 4 | 8 | 145 / 662 |

- **Compile은 PE를 늘리는 만큼 빨라진다**: PE 1→4에서 3.1–3.9배(이상적 4배 대비 77–98%). 배치 64에서 가장 잘 늘어나고(3.93배), 처리량은 1,746 → 3,784 → 6,870 img/s
- **Eager(원 모델, 배치 1)는 PE를 늘려도 조금만 빨라진다**: 16.27 → 14.01 → 13.08 ms(1.24배). 지연 대부분이 호스트 디스패치라 NPU 자원이 늘어도 크게 줄지 않는다 → PE가 많을수록 Eager/Compile 격차가 커짐(2.5 → 6.6배)
- **Eager 큰 배치는 분할 크기가 성능을 결정**: PE 1·2는 1씩 분할해야 해 op 호출이 forward당 약 5천 회 → 배치 64에서 612–831 ms. PE 4는 8씩 분할(662회) → 121 ms. 처리량은 어느 경우도 Compile 배치 1(153–505 img/s) 수준 이하
- **Compile은 PE가 많을수록·배치가 클수록 전송 비중이 커진다**: 배치 64 H2D 비중 7%(PE 1) → 18%(PE 2) → 31%(PE 4)
- **메모리**: Compile 배치 64의 그래프 컴파일이 프로세스 하나에 71.3 GiB(PE 1)·91.9 GiB(PE 2)를 쓴다. Eager job은 최대 11 GiB. 컨테이너 한도 160 GiB
- 표시: ‡ Eager 배치 분할 변형(분할 크기가 PE fusion마다 다름 — 같은 모델 구성의 비교가 아님)
- 첫 호출(컴파일) 시간 중 단독 실행 값은 일부 조건뿐이라 조건 간 비교용으로 쓰지 않는다(`MATRIX_REPORT.md` 비고)

## 8. 실험 운영 중 이슈 (기록용)
| 이슈 | 영향 | 조치 |
| --- | --- | --- |
| 컨테이너 재생성 시 aarch64 libc 헤더 패키지 소실 | 모든 신규 컴파일이 `assert.h: No such file`로 실패 | `libc6-dev-arm64-cross` 재설치. runner가 시작 전에 점검 |
| 병렬 실행 시 측정 구간 경합 | Eager(호스트 의존) 지연이 흔들림 | 측정 구간은 호스트 잠금으로 한 번에 1개, 나머지 프로세스는 일시정지 |
| 단일 PE를 `npu0:2`로 표시 → runner가 다른 프로세스로 오인 | job 1개 건너뜀, 약 45분 대기 | 파서 수정, 해당 job 재실행 |
| 컨테이너 재시작 2회(14:34, 16:13) | 진행 중 job 4개 + 진단 손실, 완료 결과는 보존 | 두 번 모두 PE 1·2 Compile 배치 64 2개 등 큰 컴파일 5개가 동시 실행 중. 단독 측정에서 이 컴파일 하나가 71–92 GiB → 둘만으로 한도 160 GiB 초과 → **메모리 부족이 가장 유력한 원인**(호스트 OOM 기록은 미확인). 동시 실행 1개로 마무리 |

## 9. 한계와 주의
- 매트릭스·수정 변형은 **조건당 1회차** → 회차 간 편차 미확인(본 실험 Compile CV < 1.1%, Eager 날짜 간 약 8% 차이)
- synthetic 입력이라 정확도 주장 없음. 출력 일치(Eager↔Compile, CPU BF16)만 확인
- Eager 배치 16 이상은 원 모델이 아니라 **배치 분할 변형** 수치
- Compile 배치 2는 그래프 컴파일 panic(`On populating O4594`)으로 실패, 미해결(SDK 이슈로 보고 대상)
- 지연 지표 차이: 본 실험 = forward_to_cpu(D2H 포함), 매트릭스 = Forward(전송 제외)

## 10. 남은 작업
1. 회차 반복으로 편차 확인(매트릭스·수정 변형은 1회차)
2. Compile 배치 2 panic, Eager op별 컴파일 실패 SDK 보고
3. 큰 배치 그래프 컴파일 메모리(71–92 GiB) — 병렬 실행 시 메모리 한도 고려 필요

## 11. PPT 구성 제안
| 슬라이드 | 내용 | 자료 |
| --- | --- | --- |
| 1 | 핵심 메시지 5줄 | 0절 |
| 2 | 환경·측정 방법 | 1–2절 |
| 3 | Eager vs Compile 지연·처리량 vs 배치 | 4절 표, `figures/latency_vs_batch.png`, `throughput_vs_batch.png` |
| 4 | 배속(Eager/Compile) vs 배치 | `figures/speedup_vs_batch.png` |
| 5 | Eager 컴파일 실패 원인과 수정 | 5절 표 |
| 6 | Eager가 느린 이유(op 디스패치) | 6절 |
| 7 | PE fusion 효과 (Compile 선형, Eager 정체) | 7절 표 — 막대그래프 권장(PE 1·2·4 × 모드) |
| 8 | 구간별 breakdown (H2D 비중 증가) | 7절 breakdown, `figures/transfer_vs_batch.png` |
| 9 | 한계·다음 단계 | 9–10절 |

그림 경로(본 실험): `Resnet/Phase_2/results/batch-scaling-fusion4-pe4-7-20261119-124115/figures/` — `latency_vs_batch.png`, `throughput_vs_batch.png`, `speedup_vs_batch.png`, `latency_per_image.png`, `scaling_efficiency.png`, `first_call_vs_batch.png`, `transfer_vs_batch.png`
