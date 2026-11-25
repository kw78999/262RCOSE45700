# ResNet50 Eager vs Compile 배치 스케일링 (Phase 2) — 결과 보고

> 2026-11-20 10:50 KST 기준. 본 실험(fusion 4) 완료. Eager 배치 2·16·32·64·128 실패 원인 확인·수정 완료, 수정 변형 측정 완료(배치당 1회차).

## 1. 목적
- Furiosa RNGD에서 같은 ResNet50을 Eager / `torch.compile`로 실행하고, 배치 크기에 따른 정상상태 지연·처리량 비교
- Phase 1 배치 1 결과를 독립 세트로 재검증

## 2. 실험 환경
| 항목 | 값 |
| --- | --- |
| NPU | RNGD 1장 (npu0), firmware 2026.4.0 |
| SDK | furiosa-torch / kernels 2026.4.0 (rev 5800e2ba24) |
| SW | Python 3.12.3, torch 2.10.0+cpu, torchvision 0.25.0 |
| 장치 설정 | **PE 4개 fusion**, `furiosa:1` = 물리 PE 4–7 (BERT 실험이 PE 0–3 사용) |
| 모델 | ResNet50 IMAGENET1K_V2, BN folding + layer4 conv2 출력 128채널 분할 (`bn_folded_output_split128`), BF16 |
| 옵션 | CPU fallback=False, SDK 기본 compiler 설정 |
| 입력 | synthetic normal `[B,3,224,224]` (정확도 평가 아님) |

## 3. 측정 방법
- 조건(모드, 배치, 회차)마다 **새 프로세스 + 새 compiler 캐시**
- 첫 호출 1회(컴파일 포함, 별도 기록) → 워밍업 10회 → 반복 50회
- 기준 지연 `forward_to_cpu`: 입력을 NPU에 미리 올린 뒤 forward ~ 출력 `.to("cpu")` 반환까지
- 배치 1·2·4·8·16·32 × 3회 + 확장 64·128·256 × 3회, 배치 1은 5세트 재검증

## 4. 결과 (본 실험, fusion 4, 평균 지연 ms, 회차 평균)
| 배치 | Eager | Compile | 배속 (E/C) | Compile 처리량 (img/s) | Compile 첫 호출 (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12.98 | 2.05 | 6.3× | 489 | 499 |
| 2 | 16.89* | 실패 | – | – | – |
| 4 | 13.69 | 2.45 | 5.6× | 1,631 | 893 |
| 8 | 14.27 | 2.82 | 5.1× | 2,837 | 941 |
| 16 | 30.01* (중앙값) | 3.83 | 7.8× | 4,174 | 1,023 |
| 32 | 62.71* | 5.59 | 11.2× | 5,725 | 1,422 |
| 64 | 120.42* | 9.48 | 12.7× | 6,755 | 1,781 |
| 128 | 300.06* | 17.33 | 17.3× | 7,387 | 2,204 |
| 256 | 건너뜀 | 34.99 | – | 7,317 | 2,455 |

- \* Eager 수정 변형(실패 op만 배치 분할, 6절), 배치당 1회차. 원 모델 Eager는 이 배치들에서 컴파일 실패
- 실행 결과: 성공 35 / 실패 4 / 건너뜀 19 (worker 41,290 s). Eager b32 실패로 확장 단계 Eager는 자동 건너뜀
- **Compile**: 회차 간 편차 매우 작음 (CV < 1.1%). 처리량은 배치 128 부근에서 포화(약 7,400 img/s)
- **Eager**: 배치 1→8에서 지연 13.0→14.3 ms로 거의 일정. forward당 Eager 로드 캐시 miss 42회(고유 op 44개 > SDK LRU 32) — 원인은 6절 끝 진단 참조(대부분 op별 호스트 디스패치)
- **PE fusion 효과 (배치 1)**: Compile 6.78 → 2.05 ms, Eager 16.35 → 12.98 ms (1 PE 대비). 1 PE 값은 Phase 1(6.68 / 16.10 ms)과 근접
- **출력 일치**: Eager↔Compile max abs error 0.016–0.031 (BF16), cosine ≥ 0.99999, top-1 일치

## 5. Eager 실패 원인
- Eager는 aten op마다 **단독 그래프로 따로 컴파일**한다. op 하나라도 SDK 컴파일러가 거부하면 forward 전체가 중단된다. 같은 모델이 Compile(그래프 전체 컴파일)에서는 배치 256까지 통과한다.
- 모든 op을 컴파일만 해 보는 진단 도구(`eager_op_probe.py`)로 실패 op을 전부 찾았다. 실제 NPU에서 통과한 배치 8은 진단에서도 44/44 통과했고, 실패 메시지도 실제 Eager 실행과 일치했다.

| 배치 | 실패 op (고유 44개 중) | 대표 에러 문구 | 실패 위치 |
| ---: | :-: | --- | --- |
| 2 | 3 | `On populating O25: called Option::unwrap() on a None value` (panic) | `maxpool` |
| | | `insufficient sequencer entries: required 9` | layer3.1–5.conv2, layer4.0.conv2 |
| 16 | 3 | `insufficient sequencer entries: required 9` | layer3.*.conv3, layer3.1–5.conv2, layer4.0.conv2 |
| 32 | 5 | 위 sequencer + `O18/O20 lowering failed: TransposedLower[bf16]…` | + layer3 블록의 `add_`·`relu_` |
| 64 | 9 | `TransposedLower…`, `attempt to read incomplete tensor buffer` | stem `relu`, layer3·4 블록, `avgpool` |
| 128 | 9 | `TransposedLower…`, `BridgeTacticFailure`, DMA `completion_ids should not be empty` | + layer1·2 conv3 |

- 실패가 배치에 비례하지 않는다(배치 2 실패, 4·8 통과, 16부터 다시 실패). 컴파일러가 shape별로 고르는 타일링(tactic)이 PE 자원 한도를 넘거나 lowering 단계에서 지원하지 않는 경우로 보인다(SDK 내부 정의는 미확인).
- 실패는 14×14·7×7 해상도(layer3·4)에 몰리고, 배치 128에서는 layer1·2까지 번진다. 1 PE에서는 배치 2·4부터 같은 에러가 났다.
- Compile b2는 그래프 전체에서 같은 계열의 panic(`On populating O4594`)이 나 실패했다(이번 수정 범위 밖).

## 6. 수정 내용과 검증
- **선택적 배치 분할**: 실패 op을 포함하는 가장 작은 모듈(conv, pool, 또는 잔차 블록)만 배치 축으로 나눠 실행 후 concat
  - 배치 16 이상은 8씩, 배치 2는 1씩 분할 → NPU에서 이미 통과한 shape으로만 컴파일
  - 샘플별 독립 연산이라 수학적으로 동일. 가중치·state SHA256 그대로. Eager에만 적용, 변형명 `…+eager_batch_chunkN`
  - 분할 모듈 수: b2 10개, b16 15개, b32 10개, b64 11개, b128 18개 (`eager_batch_partition.json`)
- 출력 채널 분할(Phase 1 방식)은 이 에러들을 해결하지 못했다.
- **검증**
  - 진단 도구 재실행: 수정 모델의 op이 다섯 배치 모두 100% 컴파일 (50/49/49/52/54개)
  - NPU 실행: 다섯 배치 모두 끝까지 실행. CPU BF16 대비 max abs error ≤ 0.031, cosine ≥ 0.99999, top-1 100%. b16–128은 Compile 출력과도 top-1 100%
  - 단위 테스트 42개 통과 (분할 전후 출력 동일, state 불변)

**수정 변형 측정** (배치당 1회차, 새 프로세스·새 캐시, `furiosa:1`, 5개 모두 성공, worker 5,160 s)

| 배치 | Eager 평균 (ms) | 처리량 (img/s) | 배속 E/C | 첫 호출 (s) | op 호출/forward |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 (원 모델) | 14.27 | 560 | 5.1× | 573 | 130 |
| 2 | 16.89 | 118 | – | 416 | 140 |
| 16 | 32.69 (중앙값 30.01) | 489 | 7.8× | 851 | 145 |
| 32 | 62.71 | 510 | 11.2× | 1,301 | 271 |
| 64 | 120.42 | 531 | 12.7× | 1,273 | 662 |
| 128 | 300.06 | 427 | 17.3× | 1,233 | 1,375 |

- 분할 변형의 Eager 처리량은 427–531 img/s로 원 모델 b8(560)을 넘지 못한다. 잘게 나눌수록 op 호출이 늘어 지연이 커진다 → 배치를 키워도 Eager는 이득 없음
- b16 평균은 이상치 1회(155 ms) 포함. 1회차라 회차 간 편차는 미확인

**Eager 지연이 배치 1–8에서 약 13 ms로 일정한 원인 (진단)**
- forward당 op 호출 130회 중 42회가 SDK 로드 캐시 miss다(`lru_cache(maxsize=32)` < 고유 op 44개).
- 캐시를 2048칸으로 늘리면 miss가 0이 되고 b1 13.0 ms다. 새 캐시로 다시 잰 원래 설정(miss 42)은 14.07 ms였다 → 재로드 몫은 많아야 약 1 ms
- 나머지 약 11 ms(NPU 계산 약 2 ms 제외)는 **op별 호스트 디스패치**다. 분할 변형에서 op 호출 수에 비례해 지연이 커진 것과도 일치
- b1 Eager가 하루 사이 12.98 → 14.07 ms(+8%)로 달라졌다. Eager 절대값은 날짜 간 비교에 주의

## 7. 남은 작업 / 한계
1. 수정 변형은 배치당 1회차만 측정(2시간 제한). 회차 간 편차 확인에는 2회차 이상 필요(배치당 약 7–22분)
2. Compile 배치 2 실패(`On populating O4594` panic)는 미해결 — SDK 이슈로 보고 대상
3. Eager 로드 캐시 크기 확대(`ft.config.set_eager_cache_size(num_load=...)`)가 측정 조건에서 주는 이득은 직접 측정하지 않음
