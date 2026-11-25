# ViT RNGD PE·batch 성능 확장 보고서

run `20261118T081725Z-a6c274`, 상태 `finished_with_coverage_recorded`. 분석 2026-11-19T08:54:35.641213+09:00 (KST). elapsed 11.078/12시간. 수치 허용 기준 pending으로 모든 성능은 **provisional**이다.

## 검토된 완료 상태

**부분 완료**: 현재 규약의 정식 mode trial 36개, Eager/Compile paired trial 18쌍이다. 측정 제어 종료 시각은 2026-11-19 04:22:05 KST, 당시 예산은 약 55.3분 남아 있었다. <code>finished_with_coverage_recorded</code>는 제어 루프가 종료됐다는 상태이며 전체 실험 목표를 달성했다는 뜻은 아니다.

대표 5 trial 미완료 (PE, batch, mode, 부족 수): [(1, 1, 'eager', 2), (1, 1, 'compile', 2), (4, 4, 'eager', 2), (4, 4, 'compile', 2)]. 별도 stability 성공/시도: 0/2. B1/B4 결과에서 큰 배치 처리량 포화점이나 B16 이상 다중 PE 이점을 추론하지 않는다. 원본 12시간 기한과 보고서 재집계 시각은 구분했다.

**재개 후 환경 오류:** 10개 worker가 SDK 내부 C 컴파일 단계에서 <code>fatal error: assert.h: No such file or directory</code>로 실패했다. 대표 조건 trial 4/5와 stability는 해당 첫 호출을 완료하지 못했으며, 이 실패에서 나온 반복 성능 자료는 없다. 실행 프로세스와 NPU 할당이 관측됐다는 사실은 성공한 forward를 뜻하지 않는다. 제어기가 이 공통 환경 오류를 전체 중단 사유로 분류하지 못해 후속 시도도 실패했다. SDK/ViT/PE 미지원으로 판정하지 않는다. 정확히 어떤 환경 패키지가 달라졌는지는 전후 식별 자료가 부족해 미확인이다. [failure_summary.csv](failure_summary.csv)와 각 case의 원본 error를 근거로 한다.

## 증거 구분

① 실행: 각 worker의 새 프로세스/PID/모델/runner, BF16 state·입력·weights 해시, strict control, 요청 Fusion/SDK 구성, timed_call 원본을 저장했다. 완료한 범위만 아래에 제공한다.

② 설치 소스: set_fusion(P)는 장치 생성 전에 한 번 지정한다. 1/2/4PE는 하드웨어 fusion, 8PE는 4PE×2 cluster 소프트웨어 fusion. 동일 fused logical device의 current stream은 그 stream에 제출된 작업 완료를 기다린다는 SDK 계약을 적용했다. evidence/ 및 pe_configuration.md 참조.

③ 조건부 추정: 손익분기/확장 효율은 이 실험의 성능 비율이다. 상대 효율은 실제 하드웨어 사용률이 아니다. 시간 변화만으로 codegen/cache/thermal throttling을 확정하지 않는다.

④ 미확인: 실제 compiler 시간/codegen/disk hit·miss, 모든 연산 placement, binary runtime 내부의 PE별 상세 동기화, 수치 허용 기준. 프로세스 새 시작은 시스템 전체 cold가 아니다.

## 완료 범위와 연구 질문

정식 mode worker 36개, 탐색 mode worker 1개. 반복 호출 수를 독립 trial 수로 세지 않았다. coverage.csv에 미지원/OOM/timeout/미측정/보류를 0 성능 대신 빈칸·사유로 보존한다.

|단계|PE|Batch|Mode|독립 trial|warmup|반복 수/trial|반복 측정 s/trial|trial 평균 중앙값 ms|trial 평균 최소–최대 ms|중앙 images/s|안정화|
|---|---:|---:|---|---:|---|---|---|---:|---|---:|---|
|exploration|8|4|eager|1|[20]|[50]|[6.105]|122.1|122.1–122.1|32.76|None|
|formal|1|1|compile|3|[100]|[2724, 2723, 2729]|[10.0, 10.003, 10.003]|3.671|3.666–3.674|272.4|heuristic_satisfied|
|formal|1|1|eager|3|[100]|[500, 500, 500]|[22.607, 22.296, 22.547]|45.09|44.59–45.21|22.18|heuristic_satisfied|
|formal|1|4|compile|3|[140]|[955, 958, 955]|[10.007, 10.01, 10.006]|10.48|10.45–10.48|381.8|heuristic_satisfied|
|formal|1|4|eager|3|[140]|[500, 500, 500]|[61.251, 61.514, 61.477]|123|122.5–123|32.53|heuristic_satisfied|
|formal|2|1|compile|3|[100]|[3995, 3989, 4051]|[10.001, 10.002, 10.0]|2.503|2.469–2.507|399.5|heuristic_satisfied|
|formal|2|1|eager|3|[100]|[500, 500, 500]|[21.96, 22.065, 22.151]|44.13|43.92–44.3|22.66|heuristic_satisfied|
|formal|4|1|compile|3|[180]|[5464, 5431, 5474]|[10.001, 10.001, 10.001]|1.83|1.827–1.842|546.3|heuristic_satisfied|
|formal|4|1|eager|3|[180]|[500, 500, 500]|[22.477, 22.252, 22.444]|44.89|44.5–44.95|22.28|heuristic_satisfied|
|formal|4|4|compile|3|[100]|[1805, 1798, 1803]|[10.001, 10.003, 10.002]|5.547|5.541–5.563|721.1|heuristic_satisfied|
|formal|4|4|eager|3|[100]|[500, 500, 500]|[62.143, 62.286, 62.15]|124.3|124.3–124.6|32.18|heuristic_satisfied|
|formal|8|1|compile|3|[100]|[5291, 5236, 5328]|[10.001, 10.001, 10.002]|1.89|1.877–1.91|529.1|heuristic_satisfied|
|formal|8|1|eager|3|[100]|[500, 500, 500]|[23.19, 22.602, 23.384]|46.38|45.2–46.77|21.56|heuristic_satisfied|

최소 핵심 범위의 미완료 (PE, batch, mode, 부족 trial 수): [(1, 16, 'eager', 3), (1, 16, 'compile', 3), (2, 4, 'eager', 3), (2, 4, 'compile', 3), (2, 16, 'eager', 3), (2, 16, 'compile', 3), (4, 16, 'eager', 3), (4, 16, 'compile', 3), (8, 4, 'eager', 3), (8, 4, 'compile', 3), (8, 16, 'eager', 3), (8, 16, 'compile', 3)]. 공통 최대 탐색 배치는 1이다. 이 값은 관측 범위의 최댓값이며 B1/B4라면 큰 배치의 검증을 대신하지 않는다. 미완료 핵심 범위를 완료한 확장 연구로 표현하지 않는다.

trial별 평균·중앙값·표본 SD·CV·p95/p99, 실제 호출 수·측정 합계·PID·cache 경로·안정화 상태는 [trial_summary.csv](trial_summary.csv)에 있다. [timings.csv](timings.csv)는 느린 호출을 제거하지 않은 원본 호출 시간이다.

### RQ1·RQ3: PE 및 Eager/Compile 비율

아래 각 범위는 paired trial 비율의 최소–최대이며 신뢰구간이 아니다. exploration과 formal을 합치지 않았다.

|단계|PE|Batch|paired n|Eager/Compile 중앙값|최소–최대|
|---|---:|---:|---:|---:|---|
|formal|1|1|3|12.3|12.14–12.32|
|formal|1|4|3|11.73|11.69–11.77|
|formal|2|1|3|17.6|17.54–17.95|
|formal|4|1|3|24.56|24.17–24.57|
|formal|4|4|3|22.41|22.39–22.43|
|formal|8|1|3|24.54|23.67–24.91|

|단계|PE|Batch|Mode|짝 n|PE speedup 중앙값|상대 효율 중앙값|
|---|---:|---:|---|---:|---:|---:|
|formal|1|1|compile|3|1|1|
|formal|1|1|eager|3|1|1|
|formal|1|4|compile|3|1|1|
|formal|1|4|eager|3|1|1|
|formal|2|1|compile|3|1.466|0.7332|
|formal|2|1|eager|3|1.018|0.5089|
|formal|4|1|compile|3|2.006|0.5014|
|formal|4|1|eager|3|1.005|0.2511|
|formal|4|4|compile|3|1.889|0.4722|
|formal|4|4|eager|3|0.9876|0.2469|
|formal|8|1|compile|3|1.942|0.2428|
|formal|8|1|eager|3|0.9748|0.1219|

PE scaling은 같은 mode/batch/trial의 유효 1PE를 기준으로만 계산했다. 비교 범위·relative efficiency·PE당 처리량은 scaling_summary.csv에 있다. SDK allocation 확인은 PE 활용률 증명이 아니며 진단 task 관측과 구분한다. 요청 PE/SDK 구성/SMI core/trace 이름/미확인은 pe_observations.csv에 따로 보존했다. PE 간 trial index를 맞췄지만 측정 시각이 인접한 것은 아니므로 시간 경과의 영향이 남을 수 있다.

### RQ2: Batch 증가

batch-call latency는 한 번의 배치 호출 전체 시간이다. images/s=B×호출 수/호출 시간 합계, amortized ms/image=평균 batch 시간/B. 후자를 개별 요청 응답시간으로 부르지 않는다. 고정 네 이미지 반복이며 다양한 이미지 정확도 평가가 아니다. 각 배치의 입력 hash는 조건 간 확인한다. 처리량 증감은 condition_summary.csv와 figures/01·02에서 그대로 표시한다.

|PE|Mode|B1→B4 batch 지연시간 ms|B1→B4 images/s|처리량 증가 비율|
|---:|---|---|---|---:|
|1|eager|45.09→123|22.18→32.53|1.467|
|1|compile|3.671→10.48|272.4→381.8|1.402|
|4|eager|44.89→124.3|22.28→32.18|1.445|
|4|compile|1.83→5.547|546.3→721.1|1.32|

여기서 B4는 B1 대비 배치 확대의 관측점이며 큰 배치 포화점을 뜻하지 않는다.

### RQ4: 워밍업·독립 trial·장시간 변화

100회 초과를 필요로 한 안정화 탐색: ['stabilization_p1_b4_compile_t1_a2:140', 'stabilization_p4_b1_compile_t1_a1:140', 'stabilization_p4_b1_eager_t1_a1:180']. 최대500회에도 불명확: 완료 probe에서 없음. 최근20회×3 구간 평균 변화≤2%, 각 CV≤0.05는 보조 휴리스틱이며 steady state 증명이 아니다. Eager/Compile은 두 probe가 요구한 횟수의 최댓값을 같은 PE/batch의 모든 정식 trial에 공통 적용했다.

안정화 상태는 별도 probe의 최근 구간 판정이며 정식 trial 전체의 steady state를 증명하지 않는다. 정식 trial에서는 공통 고정 warmup 수를 지키고, 반복 구간의 CV·느린 호출을 그대로 보고한다. 기존 warmup5 충분성 검토: warmup_curves.csv의 1–5/6–20/마지막60회 구간을 warmup_assessment.csv에 함께 제공한다. 첫 다섯 호출 이후 변화가 작더라도 과거 측정 시점·장치·자원 상태가 달라 워밍업 효과 하나로 이전 차이를 설명하지 않는다. 느린 호출은 제거하지 않았다.

PE1 B1 compile: trial 평균 CV 0.113%, 최소–최대 3.666–3.674ms (n=3).

PE1 B1 eager: trial 평균 CV 0.733%, 최소–최대 44.59–45.21ms (n=3).

PE1 B4 compile: trial 평균 CV 0.161%, 최소–최대 10.45–10.48ms (n=3).

PE1 B4 eager: trial 평균 CV 0.232%, 최소–최대 122.5–123ms (n=3).

PE2 B1 compile: trial 평균 CV 0.858%, 최소–최대 2.469–2.507ms (n=3).

PE2 B1 eager: trial 평균 CV 0.433%, 최소–최대 43.92–44.3ms (n=3).

PE4 B1 compile: trial 평균 CV 0.414%, 최소–최대 1.827–1.842ms (n=3).

PE4 B1 eager: trial 평균 CV 0.543%, 최소–최대 44.5–44.95ms (n=3).

PE4 B4 compile: trial 평균 CV 0.208%, 최소–최대 5.541–5.563ms (n=3).

PE4 B4 eager: trial 평균 CV 0.130%, 최소–최대 124.3–124.6ms (n=3).

PE8 B1 compile: trial 평균 CV 0.874%, 최소–최대 1.877–1.91ms (n=3).

PE8 B1 eager: trial 평균 CV 1.766%, 최소–최대 45.2–46.77ms (n=3).

장시간 stability: 완료 자료 없음. 시도한 worker 2개, 성공 0개다. 첫 호출 실패로 300초 연속 측정은 시작하지 못했다. 정식 trial 통계와 합치지 않으며 지속 성능을 검증했다고 표현하지 않는다.

### RQ5: 초기 비용

initial_costs.csv와 break_even.csv는 반복 성능과 별도다. A=S+F+sum(W)는 이번 공통 warmup 정책을 포함한 준비 후 시나리오이며 모델·입력 준비 시간은 별도다. warmup도 실제 forward다. N은 준비 후 유효 batch 호출 수, 유효 이미지는 B×N이다. 같은 trial의 초기 비용과 반복 평균을 결합한 선형 추정이며 이번에는 수천 회 누적 검증을 하지 않았다. compiler_time_ms는 null. 기존 5회 warmup의 경계와 동일시하지 않는다.

|PE|Batch|Mode|trial n|준비 중앙값 s|S 중앙값 s|F 중앙값 s|W 중앙값 s|A 중앙값 s|A 최소–최대 s|
|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
|1|1|compile|3|1.865|0.1428|199.5|0.3675|200|196.8–203|
|1|1|eager|3|1.718|4.44e-06|98.41|4.474|102.8|102.4–102.9|
|1|4|compile|3|1.862|0.1442|442.3|1.468|443.9|442.5–453.7|
|1|4|eager|3|1.853|3.81e-06|166.5|17.18|183.6|183.3–184.4|
|2|1|compile|3|1.863|0.1438|186.3|0.2508|186.7|180–189.5|
|2|1|eager|3|1.879|4.48e-06|110.3|4.399|114.6|114.6–114.8|
|4|1|compile|3|1.882|0.1441|205.9|0.3276|206.3|205.2–206.5|
|4|1|eager|3|1.867|4.74e-06|121.4|8.008|129.4|129.3–129.9|
|4|4|compile|3|1.868|0.1426|471.3|0.5544|472|468.7–479.1|
|4|4|eager|3|1.914|4.45e-06|256.5|12.4|268.9|268.5–269.2|
|8|1|compile|3|1.909|0.1442|460.7|0.1899|461.1|459.4–462.6|
|8|1|eager|3|1.928|4.08e-06|369.4|4.52|373.9|372.6–375.5|

|PE|Batch|paired n|각 trial 추정 N의 중앙값|최소–최대 N|유효 이미지 중앙값|상태|
|---:|---:|---:|---:|---|---:|---|
|1|1|3|2349|2296–2417|2349|{'finite': 3}|
|1|4|3|2312|2306–2404|9248|{'finite': 3}|
|2|1|3|1740|1563–1795|1740|{'finite': 3}|
|4|1|3|1776|1759–1810|1776|{'finite': 3}|
|4|4|3|1710|1680–1770|6840|{'finite': 3}|
|8|1|3|1989|1869–2050|1989|{'finite': 3}|

N 요약은 같은 trial에서 계산한 추정값들의 기술 통계이며, 서로 다른 trial의 초기 비용·반복 평균을 새로 결합한 추정이 아니다. 범위는 신뢰구간이 아니다.

## 실행 경로·출력·장치 관측

별도 출력 구조/finite 체크 14건, 비교 관계 22건. max/mean abs error, cosine, Top1은 output_validation.json. 반복 이미지는 독립 검증 표본으로 세지 않는다. Batch1 vs 큰 batch는 첫 이미지 한 장에 대한 비교다. 허용 오차 pending이므로 출력 검증 완료나 배포 추천이 아니다. 성능 반복의 모든 출력에 finite 검사를 했다는 뜻은 아니며 별도 검증에서 관측한 출력만 확인했다. BF16은 모델·입출력 dtype이며 SDK 기본 implicit casting을 유지해 모든 내부 연산의 dtype을 보장하지 않는다. 진단의 backend 진입은 codegen 횟수가 아니며 Eager backend 진입0도 실제 컴파일0을 뜻하지 않는다.

furiosa-smi info/status/ps를 controller에서30초 간격 및 worker 경계에 수집했다. 온도/전력/메모리/core process는 지원되며 이 환경에서 utilization은 `-`로 미지원/미확인이다. 장치 reset·governor·전력 설정 변경·타인 프로세스 종료는 하지 않았다. 공용 장치 예약은 원자적이지 않으며 관측 시 타인 PID가 있으면 추가 실행을 보류한다.

PE 관측을 mode별로 구분한 아래 표는 별도 진단 배치의 근거다. 다른 배치에서 동일 task 구성이 재관측됐다는 뜻은 아니다. task 미관측은 NPU 미실행 증명이 아니며, SDK 할당과 stream launch가 확인돼도 실제 PE 활용 전체를 확인한 것으로 표시하지 않는다.

|요청 PE|Mode|진단 Batch|NPU task 수|관측 cluster|관측 상태|
|---:|---|---:|---:|---|---|
|1|compile|1|2|['0']|observed_in_separate_diagnostic; single_cluster_configuration|
|1|eager|1|185|['0']|observed_in_separate_diagnostic; single_cluster_configuration|
|2|compile|1|0|[]|not_observed_in_accepted_diagnostic; single_cluster_configuration|
|2|eager|1|185|['0']|observed_in_separate_diagnostic; single_cluster_configuration|
|4|compile|1|0|[]|not_observed_in_accepted_diagnostic; single_cluster_configuration|
|4|eager|1|184|['0']|observed_in_separate_diagnostic; single_cluster_configuration|
|8|compile|1|0|[]|not_observed_in_accepted_diagnostic; unverified|
|8|eager|1|368|['0', '1']|observed_in_separate_diagnostic; both_clusters_observed|

PE별 utilization 및 작업 균형은 모두 미확인이다. 설치된 Profiler 문서는 NPU task 기록을 설명하지만 이번 task 누락 원인이나 다중 cluster 제한을 명시하지 않는다. NPU task 수를 커널 수 또는 fusion 횟수로 해석하지 않는다. 최초 2PE Compile 진단 a1은 장치 감시가 worker의 gcc/cc1 자식 프로세스를 외부 점유로 오인해 contaminated로 보존했다. process group/session/부모 계보 확인을 추가한 뒤 새 a2로 재실행했다. a1의 task 관측은 원본 보존 자료이며 승인된 a2의 task 누락을 대체하는 근거로 쓰지 않는다. compiler 자식 프로세스 관측으로 SDK 전체 codegen 횟수·순수 compiler 시간·disk miss를 확정하지 않는다.

## 이전 기준점과 차이

SDK/torch 버전은 environment.json을 기준으로 비교한다. 이전 npu1 PCI04:00.0과 이번 npu0 PCI03:00.0은 장치 식별이 다르다. CPU threads22/inter-op64 고정, 양쪽 strict path, 기존 입력·가중치 해시 재사용. warmup5/repeat50에서 공통>=100/repeat>=500 및>=10초로 측정 규약이 달라졌다. 이전과 같게 만들기 위한 설정 조정은 하지 않았다. baseline_comparison.csv는 현재 유효1PE formal trial 중앙값과 이전 trial평균 중앙값의 기술적 비교다.

## 실패·보류와 한계

실패/timeout/중단 상태 수: {'contaminated': 2, 'timeout': 1, 'interrupted': 1, 'error': 10}. 보류 이벤트 9건(전체 이벤트 14건). 개별 실패의 원인·단계·측정 완료 여부는 [failure_summary.csv](failure_summary.csv), 전체 명령·시각·PID·예산은 [manifest.json](manifest.json)에 있다. timeout은 하드웨어 미지원 판정이 아니다. 탐색/정식/안정화/stability는 합산하지 않는다.

8PE B4 Compile 검증은 3600초 실행 기한으로 종료됐다. manifest wall_s에는 감지·종료 정리 시간이 추가되며 완료된 first_call F는 측정되지 않았다(null). initial_costs.csv 및 budget_accounting.json의 incomplete_first_attempt_wall_s는 first_start 이벤트부터 worker 종료까지의 wall 관측이며 정확한 모델 호출 시간이나 compiler 시간이 아니다.

추가 보류: exploration_p4_b4_compile_t1_a1에서 PID267156가 SMI 조회 뒤 사라져 신원을 확인하지 못했다. 이 PID를 타인 또는 자체 compiler라고 확정하지 않고 contaminated 원본을 유지했다. 재개 전 장치가 비어 있음을 확인했다. 이후 장치 조회는30초 간격을 유지하면서 SMI ps 실행 중에만 /proc identity를5ms 간격으로 보강했다. 여전히 미확인 PID는 보류하며 supervisor는 두 번 연속 빈 장치를 확인한 뒤 새 attempt로 재개한다. 관측 방법 변경의 영향은 별도 통제 실험으로 분해하지 않았다.

시간 배분 변경: 핵심3 trial을 먼저 완료하고, 가능하면 대표5/stability 예산을 남긴 상태에서 B16 이상 frontier 탐색을 추가한다. 대표5는 완료 핵심 중 baseline B1 및 Compile 처리량 중앙값이 가장 높은 조합 등 최대2개다. 첫 호출 비용과 보류·규약 수정 재측정·중단 시간 모두 최초 생성 시각부터의12시간 wall 예산에 포함된다. 이 때문에 누락된 조건을 SDK 미지원으로 해석하지 않는다.

로그 규약 수정: SDK import가 Python furiosa_torch logger를 DEBUG로 지정하므로 root WARNING만으로는 첫 호출 상세 로그가 차단되지 않았다. worker v1 원본·결과를 모두 보존하고, v2에서 해당 logger를 명시적으로 WARNING(진단만 DEBUG)으로 설정했다. 모델·정밀도·backend·동기화·측정 경계는 바꾸지 않았다. v1 성능은 trial_summary.csv에 남기되 현재 집계·비율·최고 처리량에서 제외한다. 정식 trial과 안정화는 새 attempt로 재측정한다. 이전 탐색은 실행 가능성·출력·대략 비용 및 조건부 진단 조건 선정의 참고로 재사용하며 현재 규약의 성능 비교 근거로 합치지 않는다. 별도 진단 출력·trace는 계속 근거로 사용한다. initial_costs.csv도 버전을 표시하며 초기 비용 그림은 현재 버전만 사용한다. logs/logging_protocol_correction.json 참조.

## 마지막 질문에 대한 답변

1. 실제 비교 범위: 정식 [(1, 1), (1, 4), (2, 1), (4, 1), (4, 4), (8, 1)]; 탐색 미완료. 탐색과 정식 통계는 별도다.

2. 작은/큰 batch의 다중PE 효과: P2 B1 compile: 1.466배 (1PE 대비, n=3); P2 B1 eager: 1.018배 (1PE 대비, n=3); P4 B1 compile: 2.006배 (1PE 대비, n=3); P4 B1 eager: 1.005배 (1PE 대비, n=3); P4 B4 compile: 1.889배 (1PE 대비, n=3); P4 B4 eager: 0.988배 (1PE 대비, n=3); P8 B1 compile: 1.942배 (1PE 대비, n=3); P8 B1 eager: 0.975배 (1PE 대비, n=3). 1보다 작으면 성능 저하다. 측정되지 않은 batch/PE로 일반화하지 않는다.

3. 최고 관측 정식 처리량: {'pe': 4, 'batch': 4, 'mode': 'compile', 'images_s': 721.0502678550577, 'trials': 3}. 완료 trial 전체의 중앙값 기준이며 가장 빠른 trial을 고르지 않았다.

탐색 최고 처리량(별도 first1/warmup20/repeat50 규약): {'pe': 8, 'batch': 4, 'mode': 'eager', 'images_s': 32.76070129835378}. 정식 최고와 합산·혼합 순위화하지 않았다.

4. PE당 최고 정식 처리량: {'pe': 1, 'batch': 4, 'mode': 'compile', 'images_s_pe': 381.7655164511419, 'trials': 3}. 하드웨어 utilization 측정이 아니다.

탐색 PE당 최고 처리량(별도 규약): {'pe': 8, 'batch': 4, 'mode': 'eager', 'images_s_pe': 4.0950876622942225}.

5. Eager/Compile 차이: paired trial 표와 paired_speedups.csv의 같은PE/batch 비율. 1은 동률,1보다 크면 Compile이 빠르다. 모든PE가 유리하다고 가정하지 않는다.

6. warmup100 초과: ['stabilization_p1_b4_compile_t1_a2:140', 'stabilization_p4_b1_compile_t1_a1:140', 'stabilization_p4_b1_eager_t1_a1:180']; 미확인 안정화 [].

7. 독립 trial/장시간 유지 정도: 위 trial CV·범위와 10초 stability bin을 참조한다. 미완료 stability를 안정적이라고 판정하지 않는다.

8. 출력/실행 경로 미확인: 수치 허용 기준·전체 placement·codegen/cache·PE별 활용률·binary runtime 내부 구현. 관측 trace와 SDK 계약의 범위를 구분했다.

9. 다음 가치: 우선 공통 C 컴파일 환경 오류를 복구하고 전체 오류 발생 시 중단 판단을 보완해야 한다. 이후 별도 예산으로 미완료 대표5/stability, B16 핵심 paired3, 출력 허용 기준 합의, 다양한 실제 이미지 검증 및 반복일·장치 비교를 진행할 가치가 있다. 이번 검토에서는 환경·SDK·드라이버를 변경하거나 NPU 측정을 추가 실행하지 않았다.

## 산출물

원본timings.csv / trial_summary.csv / condition_summary.csv / paired_speedups.csv / scaling_summary.csv / initial_costs.csv / break_even.csv / output_validation.json / warmup_curves.csv / stability_bins.csv / device_states.jsonl / diagnostic_summary.json / coverage.csv / figures/*.png 및 대응CSV. p95/p99는 각trial 내 n개 호출과 repeat_sum_ms 범위의 표본 분위수이며 독립 trial 불확실성 추정이 아니다.

## 실행 환경 재개 기록

추가 재개 2026-11-18T18:54:58.049561+00:00: 이전 worker validation_p1_b16_compile_t1_a1는 interrupted; incomplete first call, no output or latency conclusion. PID1 시작 추정 시각 2026-11-19T03:30:42.889561+09:00가 이전 manifest 중단 시각과 근접하고 이전 supervisor/controller/worker PID가 모두 없어 실행 컨테이너 재시작 정황이 있다. 재시작을 유발한 원인은 미확인이다. 새 supervisor/controller/단일 NPU worker 실행 및 SIGHUP 무시를 확인했다. SIGHUP 무시는 실행 환경 자체 종료를 막지 못한다. 완료 결과·중단 attempt·기존 12시간 wall 예산을 보존했다. 근거: logs/runtime_recovery_20261118T1854Z.json.

## 그림과 대응 CSV

누락 조건은 0이 아닌 빈칸과 사유로 표시하며, 누락 지점을 연결해 보간하지 않는다. 08번은 성공한 stability 측정이 없어 빈 패널이다.

|번호|그림|CSV|축·해석|
|---:|---|---|---|
|01|[PE별 batch 처리량](figures/01_throughput.png)|[CSV](figures/01_throughput.csv)|batch x축 log; formal과 exploration 별도|
|02|[PE별 batch-call 지연시간](figures/02_batch_latency.png)|[CSV](figures/02_batch_latency.csv)|batch x축 log|
|03|[Eager/Compile 배속](figures/03_eager_compile_heatmap.png)|[CSV](figures/03_eager_compile_heatmap.csv)|1=동률; paired n 및 누락 사유|
|04|[PE scaling speedup](figures/04_pe_scaling.png)|[CSV](figures/04_pe_scaling.csv)|1 기준선; 같은 batch/mode/trial의 1PE 기준|
|05|[상대 scaling 효율](figures/05_relative_efficiency.png)|[CSV](figures/05_relative_efficiency.csv)|1 기준선; 하드웨어 utilization 아님|
|06|[warmup 원본 지연시간](figures/06_warmup_curves.png)|[CSV](figures/06_warmup_curves.csv)|y축 log; 첫 호출 별도; 5/100회 기준|
|07|[독립 trial 평균 분포](figures/07_trial_means.png)|[CSV](figures/07_trial_means.csv)|x/y log; 모든 유효 trial 표시|
|08|[장시간 추세](figures/08_stability_trend.png)|[CSV](figures/08_stability_trend.csv)|측정 실패로 유효 데이터 없음|
|09|[초기 비용](figures/09_initial_costs.png)|[CSV](figures/09_initial_costs.csv)|x/y log; A=S+F+W, 순수 compiler 시간 아님|

![Eager/Compile 배속과 paired trial 수](figures/03_eager_compile_heatmap.png)

![워밍업 지연시간; y축 log](figures/06_warmup_curves.png)

## 보고서 검토 기록

[원본 대조 검토](report_review.md)와 [기계 판독 결과](report_review.json)에 통계·출력·파일 불변 검증을 남겼다. 검토 통과는 전체 실험 완료나 수치 허용 기준의 합격을 뜻하지 않는다.
