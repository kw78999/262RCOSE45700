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

실행 경로: eager B1: 일부 CPU tensor 연산 확인, compile B1: NPU 사용 확인 / 전체 연산 배치 미확인, eager B4: 일부 CPU tensor 연산 확인, compile B4: NPU 사용 확인 / 전체 연산 배치 미확인.

캐시: 진단/실제 출력 검증/성능을 분리했다. 성능 worker별 새 SDK cache_context 경로를 사용한다. 해당 경로 초기 부재가 시스템 전체 cold 또는 실제 disk miss를 뜻하지 않는다. CPU dispatch·내부 전송·실행·완료 대기를 포함한 ms/batch call이며 kernel-only가 아니다. 상세 로그/프로파일은 성능 worker에 없다.

장애물·질문(최대 3개): SDK 2026.4.0의 원본 native attention에 대한 공식 NPU 지원·해결 경로; BF16 logits atol/rtol 합의; Compile 전체 연산 배치·실제 codegen·disk cache의 공식 관측 방법. 실제 이미지 4장과 trial 3개는 전체 입력 분포나 일반적 우열을 대표하지 않는다.
