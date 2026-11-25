# 실행 계획과 예상 소요시간

12시간 wall elapsed 예산, worker당 3600초. 기존 첫 호출 B1 Eager 약98초/Compile 약199초, B4 약166초/435초였으므로 새 경로·새 장치에서 같은 비용이라고 가정하지 않는다. 기준점 검증/안정화/첫 formal pair 약 30–60분 예상, 넓은 탐색·핵심 3 trial·대표 5 trial은 수시간 이상이며 전체 행렬은 12시간 초과 가능. 다음 worker 비용은 이미 관측한 준비·첫 호출·warmup·반복 시간으로 보수적으로 추정한다. 남은 예산 안에 완료가 어려우면 보류하고 누락 사유를 남긴다.

A: 1PE B1/B4 별도 출력/경로 검증, 안정화 탐색, 공통 warmup을 적용한 첫 formal trial. B: B1/4/16/64/128을 우선하되 작은 batch부터 PE 순환, 이후 B8/32, 조건 성공시 B256/512 검토. PE별 작은 batch 검증에서 구조/finite 오류면 그 PE 확대 중단, 자원 부족이면 그 batch 이상 보류. timeout은 unsupported가 아니다. C: 공통 큰 batch를 결정한 뒤 모든 유효 PE의 B1/B4/B16/공통 큰 batch를 3 trial씩, 남은 조건 후순위. D: 각 PE의 B1/공통 큰 batch/관측 최대 처리량 batch를 5 trial까지. E: 공통 큰 batch에서 최대 관측 Compile 처리량을 보인 PE를 골라 양쪽 300초 stability. 모두 직렬.

정식 trial과 별도 validation/diagnostic/stabilization/exploration/stability worker를 분리한다. 각 worker는 새 프로세스/모델/runner/전용 cache_context. 새로운 cache 경로는 실제 codegen/disk miss의 증거가 아니다. 반복 원본/실패/느린 호출 보존. 출력 허용 기준 pending이므로 성능은 provisional.

예산 예약 보강(최초 정식 trial 전): broad exploration이 전체 wall 예산45%에 도달하면 남은 넓은 탐색을 보류하고 유효 core의 paired3 trial을 우선한다. 새로운 PE1만 소모하지 않도록 각 batch에서PE 순환은 유지한다. formal mode 선행 순서는 seed로 정한 첫 순서 후 trial마다 교대한다. 스케줄링 변경만 적용했으며 worker 측정 코드/모델/정밀도/compile 옵션은 변경하지 않았다. 완료 worker 경계에서 controller를 체크포인트했고 해당 worker의 완료 result로 재개한다.

장시간 안정성 두 모드 비용을 미리 예약해 핵심3 trial 이후 대표조건을5 trial로 늘리는 단계가 stability 예산을 소모하지 않게 한다. D(가능한5 trial) 후 E(별도300초/모드), 남는 예산에서 나머지 조건을 진행한다. core batch마다PE 순서도 seed/rotation을 사용한다. 측정 worker 소스는 최초 해시를 유지한다.

장치 감시 수정: SMI가 NPU descriptor를 상속한 gcc/cc1 자식도 core 사용 PID로 표시할 수 있다. PID 명칭이나UID만 신뢰하지 않고 worker의OS session/process group 또는PPID 계보를 SMI 전후에 확인한다. 자신의 검증된 컴파일 자식은 독립 NPU worker로 세지 않는다. 타인 또는 확인되지 않은 PID는 계속 보류한다. 과거 contaminated attempt는 재분류하지 않고 새 attempt로 재시도한다. actual codegen 전체 시간/횟수와 disk hit/miss는 이 관측으로 확정하지 않는다.

핵심3 trial을 시작하기 전에 해당 PE/batch의 남은 mode-trial 전체와 필요한 안정화 probe 비용을 합쳐 완료 가능성을 확인한다. 비교 가능한3 trial을 완료하는 우선순위를 유지하며, 대표 stability 양쪽의 첫 호출/공통warmup/300초 비용은 C단계 전에 예약한다. Stability는 worker cap에 맞는 가장 큰 공통 batch, 그 batch의 탐색 Compile 처리량이 가장 높은PE를 선정한다. 예상 비용 기준이며 실제성능이 좋은 trial만 선택하지 않는다.

시간 예측은 clipping하지 않은 관측 기반 예상값을 사용한다. 예상 worker 비용이3600초를 초과하면 held_predicted_worker_timeout으로 보류하며 하드웨어 unsupported로 판정하지 않는다. C/D의 trial 묶음 예측은 관측 첫 호출×1.35, 반복 평균×호출 수×1.3,30초 여유와 알려진 공통warmup(미확인시max500)을 사용한다. 남은 budget에서 완료할 수 있는 묶음을 먼저 끝낸다. Stability 후 예약한 예산의 잔여분으로 미완료 core3 trial을 먼저 재검토한다.


## 로그 규약 수정 (2026-11-18 10:32 UTC)

SDK Python logger 기본 DEBUG를 발견해 명시적 WARNING 설정을 적용한 worker_protocol=2를 추가했다. code/worker.py(v1)는 보존하고 code/worker_v2.py로 새 attempt를 실행한다. 모델·정밀도·backend·timed_call·반복 규약은 동일하다. v1 성능 원본은 보존하되 현재 집계에서 제외한다. 해당 단계의 재측정 시간도 최초 생성 시각부터의 12시간 예산에 포함된다. 상세 근거와 SHA256은 logs/logging_protocol_correction.json. 진단은 계속 별도 DEBUG worker다.


## 2026-11-18 14:06 UTC 재개 시 우선순위

짧은 수명 PID의 소유권을 확인하지 못한 exploration_p4_b4_compile_t1_a1은 contaminated로 유지한다. /proc identity를 SMI ps 실행 중에만 5ms 간격으로 추가 관측하며 장치 조회 자체는 기존 30초 간격을 유지한다. 사라진 PID는 여전히 소유권 미확인으로 처리한다. supervisor는 외부 점유/미확인 상태에서 두 번 연속 장치가 비어 있는 것을 확인한 뒤 새 attempt로 재개하며 타인 PID를 종료하지 않는다. 8PE B4 Compile은 실제 3600초 cap timeout이다.

핵심 paired3 trial을 우선 완료한다. 이후 대표5 trial 대상을 최대2 조합(완료된 핵심 중 baseline B1 및 Compile 처리량 중앙값이 가장 높은 조합)으로 제한한다. 대표5/stability 예산을 남길 수 있는 경우에만 B16/64/128 우선의 post-core 탐색을 추가한다. 이 단계도 PE 순환·worker cap·고정 경로·별도 validation/exploration을 유지한다. 미측정 큰 배치를 unsupported로 선언하지 않는다. 상세 코드는 controller_revision_11, 변경 사유는 logs/controller_revision_11.json.
