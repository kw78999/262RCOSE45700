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

최소 조치: 실제 이미지 4장으로 일관성 점검, 출력 기준·연산 실행 경로·컴파일 관측 방법 확인 후 독립 trial 확대를 검토한다. B8·B16 및 대표 모듈 분석은 후속 단계이며 이번에 수행하지 않았다.

첫 피드백:

- 피드백 질문 1: 이 SDK에서 실제 codegen과 disk cache hit/miss를 구분하는 공식 관측 방법은 무엇인가?
- 피드백 질문 2: BF16 Eager/Compile logits 비교의 atol/rtol 기준을 어떻게 합의할 것인가?
- 피드백 질문 3: 기본 Eager CPU fallback=True에서 전체 연산의 NPU 실행 범위를 확인할 최소 공식 방법은 무엇인가?

