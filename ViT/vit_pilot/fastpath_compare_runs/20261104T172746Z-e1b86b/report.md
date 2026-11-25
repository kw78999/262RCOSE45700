# ViT: 양쪽 MHA fastpath 해제·CPU fallback 차단 비교

run_id: `20261104T172746Z-e1b86b`. 현재 상태 `finished`. 주 성능 완료 12/12 worker. NPU worker 누적 5211.1/7200.0초, 개별 상한 1800초.

## 일곱 질문의 현재 결론

1. **[실행 확인] 제어·동일성:** worker는 새 원래 ViT를 생성하고 기존 AttentionBoundary/하위 compile이 없음을 확인한다. 양쪽 fastpath=False·cpu_fallback=False를 준비 후/측정 전/종료 후 검사하고 상태를 복원한다. 체크포인트 파일, 기존 BF16 입력 파일/배치 바이트, CPU BF16 state_dict 해시를 확인했다. NPU 이동 후 전체 state_dict를 다시 CPU로 복사해 해시했다고 주장하지 않는다.
2. **[실행 확인/미확인] 경로·출력:** 최신 진단 4/4개 완료. native MHA host event, NPU task, backend 진입 및 graph break 로그는 아래 원본과 연결했다. strict와 NPU 증거는 SDK 계약 범위이며 전체 연산 배치, 실제 codegen·disk hit/miss는 미확인이다. 출력은 이번 Eager/Compile끼리 직접 비교하며 허용 오차 미합의이면 pending이다.
3. **[실행 확인] 준비·호출 시간:** 완료한 12개 worker의 prepare/S/F/W/반복 원본을 저장했다. 같은 배치·trial끼리 묶고 50회 반복을 독립 trial 50개로 세지 않는다. 느린 정상 호출도 모두 포함한다.
4. **[조건부 추정] 손익분기:** 현재 계산 가능한 paired trial은 6/6개다. 새 데이터의 A=S+F+sum(W), T(N)=A+N*t만 사용한다. N은 준비 뒤 유효 배치 호출 수, K=N+6은 first+warmup 6회를 포함한 forward 수다. 부품별 좋은 값을 서로 다른 trial에서 골라 합치지 않는다.
5. **[조건부 추정/보류] 계산 상태:** finite/already/equal-rate/no-break-even/initial-advantage-only/insufficient 상태를 trial별로 구분한다. 상태가 다르면 하나의 N으로 합치지 않는다. 정밀한 경계는 정수 ns와 유리수로 확인하며 동률은 승리가 아니다.
   이번 결과: finite_break_even 6쌍. 누락값을 0으로 대체하지 않았다.
6. **[별도 실행 확인/보류] 누적 검증:** 새로운 Eager/Compile process 쌍의 S/F/W와 연속 work prefix를 사용한다. 아래 실제 완료 범위를 확인하며, 실행 전 또는 미도달 구간은 predicted_only/미도달로 남긴다. 검증 시작 비용으로 주 예측을 사후 보정하지 않는다.
   B1: 관측 첫 우위 N=2514, 예측 중앙값 2468 대비 +46회. 종료 N=2962까지 우위를 유지했다.
   B4: 관측 첫 우위 N=2412, 예측 중앙값 2413 대비 -1회. 종료 N=2896까지 우위를 유지했다.
7. **[소스·범위 확인] 이전 설명과의 차이:** 이전 기본 Eager의 CPU fallback 및 약 200/400회 가정 계산을 이번 수식에 넣지 않았다. 새 strict fastpath-off Eager의 실제 초기 비용도 0이 아니며, 모델 준비 비용은 별도 기록하고 주 수식에서는 제외한다.

   이전 값은 경로·캐시·초기 비용 가정이 다른 설명용 계산이었다. 이번에는 두 모드의 새 전용 캐시에서 측정한 S/F/W와 50회 평균을 같은 trial끼리 결합했다. A의 차이를 순수 컴파일 시간으로 해석하거나, 이전 대비 차이의 원인을 실제 codegen으로 확정할 근거는 없다.

## 측정 규약과 자산

`torchvision.models.vit_b_16`, `ViT_B_16_Weights.IMAGENET1K_V1`, BF16, eval/inference_mode, B1/B4×3 trial. 실제 이미지 폴더는 `/root/ViT/data/vit_real_images_4/images/`다. 순서는 cat/coffee/astronaut/rocket이며 `/root/ViT/vit_pilot/followup_runs/20261104T132538Z-c031a6/inputs/preprocessed_bf16.pt`를 재사용했다. B1의 cat은 B4에도 포함돼 고유 사진은 4개다. 이미지 출처·순서·transforms 정보와 파일/텐서 해시는 image_manifest.json과 assets.json에 보존했다. 정답 라벨이 없어 accuracy/loss는 계산하지 않는다.

기존 common.timed_call을 그대로 사용한다: current stream 사전 synchronize → perf_counter_ns → runner(inputs) → synchronize → 종료 시각. 시간은 CPU dispatch·SDK 내부 전송·실행·완료 대기를 포함한 ms/batch-call이다. 외부 모델/입력 준비와 전송, 출력 복사/검증, 파일 쓰기, 로그는 모델 타이머 밖이다. S는 runner callable/torch.compile 래퍼 생성, F는 첫 호출, W는 후속 warmup 5회다. model_input_prepare_ms는 CPU 모델·파일 로드·state hash 검사·NPU 이동·준비 완료 대기를 포함한 참고값으로 A에서 제외한다. 순수 compiler 시간은 null이다. SDK Eager도 내부적으로 연산을 컴파일할 수 있어 Eager의 torch.compile backend 진입 0회를 실제 codegen 0회로 해석하지 않는다.

각 worker는 처음 없던 고유 SDK cache_context 경로를 사용하며 진단/성능/누적 실행은 분리된다. 시스템 전체 cold나 실제 disk miss를 보장하지 않는다. Compile에는 backend만 지정했고 dynamic/fullgraph/mode/options/recompile_limit·CPU 스레드는 조정하지 않았다. 전체 모델 compile 요청과 단일 완전 캡처는 구분한다. 안정성은 기존 CV≤0.10 및 전·후반 평균 차이≤10% 휴리스틱을 재사용했다. preliminary_stable도 steady state 증명은 아니다.

최초 두 B1 진단은 forward 전에 참조 state hash의 키 표현 차이로 실패했다. 공식 체크포인트의 구형 MLP linear_1/linear_2 이름과 torchvision 로더의 0/3 이름 차이를 확인했고, 공식 변환 뒤 152개 tensor 값이 정확히 같음을 검증했다. 최초 오류/키 기준 파일과 수정 코드 버전은 보존했으며 새 attempt로 재실행했다. 모델·타이머·SDK는 수정하지 않았다. logs/state_key_conversion.json과 manifest.code_revision_history를 참고한다.

## 새 실행 경로와 출력

| 케이스 | 상태 | native MHA host | NPU task | backend 진입 | SDK segment 목록 | recompile 로그 |
| --- | --- | --- | --- | --- | --- | --- |
| diagnostic_eager_b1_t1 | error | — | — | — | [] | 0 |
| diagnostic_compile_b1_t1 | error | — | — | — | [] | 0 |
| diagnostic_eager_b4_t1 | ok | 0 | 185 | 0 | [] | 0 |
| diagnostic_compile_b4_t1 | ok | 0 | 3 | 1 | [{'submodules': 1, 'to_compile': 1}] | 0 |
| diagnostic_eager_b1_t1_attempt1 | ok | 0 | 185 | 0 | [] | 0 |
| diagnostic_compile_b1_t1_attempt1 | ok | 0 | 2 | 1 | [{'submodules': 1, 'to_compile': 1}] | 0 |

상세 로그와 profiler는 진단 worker에만 켰다. 성능에서 컴파일 이벤트 부재를 단정하지 않는다. CPU host scope 시간은 NPU kernel 시간이 아니며 Task 이름을 codegen 비용으로 사용하지 않는다. 원본 trace/guard/graph break 위치는 diagnostic_observations.json, 설치 API 근거는 logs/installed_sources.txt 및 control_installed_sources.txt다.

| 배치 | 최대 절대오차 | 평균 절대오차 | cosine 평균/최소 | Top-1 일치 | 허용 오차 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.031250 | 0.004590 | 0.99994014 / 0.99994014 | 1/1 | pending |
| 4 | 0.062500 | 0.005142 | 0.99994004 / 0.99991642 | 4/4 | pending |

이 표는 이번 진단의 fastpath-off Eager를 이번 fastpath-off Compile과 직접 비교한 값이다. first/마지막 warmup/마지막 repeat 비교와 이미지별 클래스·logit·cosine은 output_comparison.json에 있다. Top-1 일치는 정답 accuracy가 아니다. NaN/Inf/구조 오류와 합의된 오차 위반은 비교를 보류하고, 합의되지 않은 허용 오차는 사후에 만들지 않는다.

## 독립 성능 trial

| mode | B | trial | 상태 | prepare ms | S ms | F ms | W 합 ms | A ms | repeat 평균 ± SD ms | 중앙값 ms | CV | images/s | warm state |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eager | 1 | 1 | ok | 1141.011 | 0.001 | 97545.902 | 220.648 | 97766.551 | 44.432 ± 0.077 | 44.424 | 0.002 | 22.506 | preliminary_stable |
| compile | 1 | 1 | ok | 1523.183 | 268.074 | 198102.073 | 19.080 | 198389.227 | 3.656 ± 0.026 | 3.668 | 0.007 | 273.503 | preliminary_stable |
| compile | 4 | 1 | ok | 1171.443 | 243.030 | 433887.522 | 53.131 | 434183.682 | 10.478 ± 0.015 | 10.477 | 0.001 | 381.751 | preliminary_stable |
| eager | 4 | 1 | ok | 1459.204 | 0.003 | 165932.850 | 613.158 | 166546.011 | 122.715 ± 0.105 | 122.716 | 0.001 | 32.596 | preliminary_stable |
| compile | 1 | 2 | ok | 1780.721 | 257.965 | 200781.670 | 18.616 | 201058.251 | 3.629 ± 0.025 | 3.643 | 0.007 | 275.523 | preliminary_stable |
| eager | 1 | 2 | ok | 1142.817 | 0.001 | 97948.202 | 224.925 | 98173.127 | 44.969 ± 0.111 | 44.958 | 0.002 | 22.238 | preliminary_stable |
| eager | 4 | 2 | ok | 1634.602 | 0.003 | 166611.694 | 608.356 | 167220.053 | 121.779 ± 0.136 | 121.783 | 0.001 | 32.846 | preliminary_stable |
| compile | 4 | 2 | ok | 1195.913 | 243.641 | 435404.188 | 53.241 | 435701.069 | 10.479 ± 0.009 | 10.479 | 0.001 | 381.713 | preliminary_stable |
| eager | 4 | 3 | ok | 1737.545 | 0.003 | 166418.641 | 610.779 | 167029.422 | 122.482 ± 0.144 | 122.502 | 0.001 | 32.658 | preliminary_stable |
| compile | 4 | 3 | ok | 1760.220 | 242.101 | 438706.423 | 53.151 | 439001.674 | 10.467 ± 0.029 | 10.462 | 0.003 | 382.145 | preliminary_stable |
| compile | 1 | 3 | ok | 1610.113 | 242.512 | 199201.532 | 18.921 | 199462.964 | 3.637 ± 0.022 | 3.646 | 0.006 | 274.975 | preliminary_stable |
| eager | 1 | 3 | ok | 1227.148 | 0.001 | 97989.372 | 223.098 | 98212.471 | 44.770 ± 0.084 | 44.773 | 0.002 | 22.336 | preliminary_stable |

SD는 표본 SD(ddof=1)이며 신뢰구간이 아니다. 처리량은 B×유효 반복 수/유효 반복 시간 합계로 계산한 순차 호출 처리량이다. 50회 반복 전체 평균을 비용 예측에 사용한다. 평균 대신 빠른 중앙값을 골라 총시간을 예측하지 않는다. 호출 순서 원본은 timings.csv에 보존했다.

B1 독립 trial 평균(ms): eager [44.4317685, 44.96857498, 44.7699428] (중앙값 44.770, 관측 범위 44.432–44.969); compile [3.65626752, 3.62946042, 3.63669236] (중앙값 3.637, 관측 범위 3.629–3.656).

B4 독립 trial 평균(ms): eager [122.7146714, 121.77898472, 122.48178712] (중앙값 122.482, 관측 범위 121.779–122.715); compile [10.4780352, 10.4790671, 10.4672308] (중앙값 10.478, 관측 범위 10.467–10.479).

## 주 손익분기와 첫 호출 포함 표현

```text
A_m = S_m + F_m + sum(W_m)
T_m(N) = A_m + N*t_m, N >= 1
delta_A = A_C-A_E; delta_t = t_E-t_C
T_C(N)-T_E(N) = delta_A-N*delta_t
```

delta_t>0이면 N_win=max(1,floor(delta_A/delta_t)+1)이다. 정수 ns(ROUND_HALF_EVEN)로 정규화한 뒤 유리수로 계산하고 N과 N-1에 대입한다. 반복 50회 합계는 A에 들어가지 않는다. F에 포함된 실행/컴파일과 W를 다시 더하지 않는다. N은 유효 배치 호출 수, 유효 이미지는 B×N, 전체 forward 수는 N+6, 준비까지 센 이미지는 B×(N+6)이다. 전처리·프로세스 기동·model_input_prepare·외부 전송·서비스 큐/배치 조립은 이 시나리오에서 제외한다. 운영 서비스 전체 지연이 아니다.

| B | trial | 상태 | A_E ms | A_C ms | t_E ms | t_C ms | E/C 반복 배속 | 실수 교차 N | N_win | K=N+6 | 유효 이미지 | 변동 보조 상태 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | finite_break_even | 97766.551 | 198389.227 | 44.432 | 3.656 | 12.152 | 2467.724 | 2468 | 2474 | 2468 | mean_gap_exceeds_descriptive_combined_SD |
| 1 | 2 | finite_break_even | 98173.127 | 201058.251 | 44.969 | 3.629 | 12.390 | 2488.808 | 2489 | 2495 | 2489 | mean_gap_exceeds_descriptive_combined_SD |
| 1 | 3 | finite_break_even | 98212.471 | 199462.964 | 44.770 | 3.637 | 12.311 | 2461.524 | 2462 | 2468 | 2462 | mean_gap_exceeds_descriptive_combined_SD |
| 4 | 1 | finite_break_even | 166546.011 | 434183.682 | 122.715 | 10.478 | 11.712 | 2384.584 | 2385 | 2391 | 9540 | mean_gap_exceeds_descriptive_combined_SD |
| 4 | 2 | finite_break_even | 167220.053 | 435701.069 | 121.779 | 10.479 | 11.621 | 2412.230 | 2413 | 2419 | 9652 | mean_gap_exceeds_descriptive_combined_SD |
| 4 | 3 | finite_break_even | 167029.422 | 439001.674 | 122.482 | 10.467 | 11.701 | 2428.008 | 2429 | 2435 | 9716 | mean_gap_exceeds_descriptive_combined_SD |

각 trial의 strict 경계 차이와 N-1 값은 break_even_trials.csv/summary.json에 저장했다. 작은 반복 평균 차이는 두 SD의 제곱합 제곱근과 비교한 descriptive uncertain 표기로 보조하고 exact sign은 바꾸지 않는다. 수치가 나와도 출력 허용 오차 pending·미안정·관측 한계를 해소한 것은 아니다.

B1: 유효 finite paired trial 3/3, N_win 목록 [2468, 2489, 2462], 중앙값 2468, 관측 범위 2462–2489. 상태 [{'trial': 1, 'status': 'finite_break_even', 'N_win': 2468}, {'trial': 2, 'status': 'finite_break_even', 'N_win': 2489}, {'trial': 3, 'status': 'finite_break_even', 'N_win': 2462}]. 표본 3개의 범위는 CI가 아니다.

B4: 유효 finite paired trial 3/3, N_win 목록 [2385, 2413, 2429], 중앙값 2413, 관측 범위 2385–2429. 상태 [{'trial': 1, 'status': 'finite_break_even', 'N_win': 2385}, {'trial': 2, 'status': 'finite_break_even', 'N_win': 2413}, {'trial': 3, 'status': 'finite_break_even', 'N_win': 2429}]. 표본 3개의 범위는 CI가 아니다.

K≤6에서는 G(K)=S+실제 first/warmup prefix 합, K>6에서는 G(K)=A+(K-6)t를 사용한다. G(N_win+6)=T(N_win)를 exact arithmetic으로 확인했다. N_win+6은 준비 뒤 시나리오의 교차 표현이며 전체 K 구간 최초 승리라고 자동 해석하지 않는다. 초기 실제 prefix에서 Compile이 유리했던 K 목록: [{'batch': 1, 'trial': 1, 'K': []}, {'batch': 1, 'trial': 2, 'K': []}, {'batch': 1, 'trial': 3, 'K': []}, {'batch': 4, 'trial': 1, 'K': []}, {'batch': 4, 'trial': 2, 'K': []}, {'batch': 4, 'trial': 3, 'K': []}]. 세부는 initial_prefix.json이다.

## 별도 누적 실행 검증

주 성능의 paired N_win 중앙값×1.2를 올림해 목표를 잡고 N_win=1이면 최소 20회다. mode당 10000회, worker 1800초, 총 7200초 상한 안에서 양쪽 완료 비용을 보수적으로 평가했다. NPU worker는 직렬이며 새 모델·새 runner·처음 없던 전용 SDK cache를 사용한다. first+warmup 뒤 추가 숨은 warmup 없이 N회를 모두 유효 작업으로 센다. 모든 work 출력의 CPU 복사/유한값 검사는 타이머 밖에서 수행했다. 따라서 검증의 호출 사이 간격/호스트 활동이 주 성능과 다를 수 있다.

| B | 상태 | 관측 공통 N | 주 예측 median N | 관측 첫 우위 N | 관측 K | 유효 이미지 | 관측 끝까지 유지된 마지막 전환 N |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | observed_crossing_reached | 2962 | 2468 | 2514 | 2520 | 2514 | 2514 |
| 4 | observed_crossing_reached | 2896 | 2413 | 2412 | 2418 | 9648 | 2412 |

B1는 주 예측 중앙값보다 +46회 (+1.86%) 차이로 실제 첫 우위를 관측했다. 주 성능 trial의 관측 N 범위는 [2462, 2489]이며 실제 교차는 이 범위 밖이다. 이 범위는 신뢰구간이 아니다. 관측 우위는 N=2962까지 유지됐고, 목표 2962회까지 완료했다. 그 뒤 구간은 검증하지 않았다.

B1 체크포인트 (고정한 실제 paired trial 1 예측, ms 원본은 CSV):

| 위치 | N | 예측 E 초 | 예측 C 초 | 관측 E 초 | 관측 C 초 | 관측 C-E 초 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.8x | 1974 | 185.475 | 205.607 | 187.016 | 209.334 | 22.318 |
| 1.0x | 2468 | 207.424 | 207.413 | 209.267 | 211.137 | 1.870 |
| 1.2x | 2962 | 229.373 | 209.219 | 231.511 | 212.937 | -18.574 |

새 검증 시작 비용과 work 평균(사후 설명용):

| mode | S ms | F ms | W 합 ms | A ms | work 평균 ms |
| --- | --- | --- | --- | --- | --- |
| eager | 0.002 | 97809.274 | 223.013 | 98032.289 | 45.064 |
| compile | 240.535 | 201902.189 | 18.782 | 202161.506 | 3.638 |

새 검증 A와 기존 대표 trial의 t만 조합한 **사후 설명용** N_win=2554. 주 예측 곡선은 수정하지 않았다. 전환 목록: [{'N': 1, 'state': 'eager_faster', 'difference_ms': 104088.067755}, {'N': 2514, 'state': 'compile_faster', 'difference_ms': -29.85874}].

B4는 주 예측 중앙값보다 -1회 (-0.04%) 차이로 실제 첫 우위를 관측했다. 주 성능 trial의 관측 N 범위는 [2385, 2429]이며 실제 교차는 이 범위 안이다. 이 범위는 신뢰구간이 아니다. 관측 우위는 N=2896까지 유지됐고, 목표 2896회까지 완료했다. 그 뒤 구간은 검증하지 않았다.

B4 체크포인트 (고정한 실제 paired trial 2 예측, ms 원본은 CSV):

| 위치 | N | 예측 E 초 | 예측 C 초 | 관측 E 초 | 관측 C 초 | 관측 C-E 초 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.8x | 1930 | 402.253 | 455.926 | 403.305 | 457.238 | 53.933 |
| 1.0x | 2413 | 461.073 | 460.987 | 462.490 | 462.293 | -0.197 |
| 1.2x | 2896 | 519.892 | 466.048 | 521.676 | 467.348 | -54.327 |

새 검증 시작 비용과 work 평균(사후 설명용):

| mode | S ms | F ms | W 합 ms | A ms | work 평균 ms |
| --- | --- | --- | --- | --- | --- |
| eager | 0.003 | 166067.708 | 612.074 | 166679.785 | 122.581 |
| compile | 240.871 | 436749.153 | 53.100 | 437043.124 | 10.465 |

새 검증 A와 기존 대표 trial의 t만 조합한 **사후 설명용** N_win=2430. 주 예측 곡선은 수정하지 않았다. 전환 목록: [{'N': 1, 'state': 'eager_faster', 'difference_ms': 270251.056095}, {'N': 2412, 'state': 'compile_faster', 'difference_ms': -84.370203}].

observed_T는 새 S+F+sum(W)+work prefix 시간의 합이다. 사전 대기·출력 검증·로그·파일 쓰기·process 기동 등 타이머 밖 시간까지 포함한 wall-clock 전체가 아니다. 관측 마지막까지의 우위 유지가 관측 이후 영구 우위를 보장하지 않는다. 교차에 미도달하면 관측 범위와 상한을 보고하며 손익분기 없음으로 결론 내리지 않는다.

## 판단과 남은 범위

**조건부 추정:** 관측한 동일 조건에서만 반복 성능과 초기 비용을 바탕으로 손익분기를 해석한다. 별도 누적 실행의 검증 상태는 위 표대로이며, 초기 비용·호출 변동과 실행 시점 차이로 예측/관측 교차가 다를 수 있다. 출력 허용 기준은 미합의라면 계속 pending이다. 독립 3 trial과 누적 확인 1쌍을 합쳐 표본 수를 늘리지 않는다. B8/B16, 다른 모델, 수식/커널 변경, 동적 compile 튜닝은 실행하지 않았다.

원본 성능 행 672/672. all_workers_complete=True, all_unheld_workers_complete=True, all_attempts_successful=False. 앞의 두 값은 최신 logical case 기준, 마지막은 보존한 최초 실패 attempt까지 포함한다. 오류·보류는 미측정 0ms로 채우지 않았다. SDK/torch/드라이버·기존 결과·공용 캐시는 보존했다.

재현 명령과 실제 버전은 [README.md](README.md), 입력/가중치 식별자는 assets.json/asset_verification.json, 예산·명령·PID·실제 순서는 manifest.json에 있다. [PyTorch 2.10 MHA API](https://docs.pytorch.org/docs/2.10/backends.html#torch.backends.mha). Furiosa 전용 API는 저장한 설치 소스가 직접 근거다.
