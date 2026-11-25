# 중간 실험 피드백

2026-11-19 00:19:13 KST까지의 고정 결과다. 원본 run은 기준 시점에 계속 실행 중이다.

- 현재 규약 정식 12 mode trial, Eager/Compile 6쌍. 8PE B1만 각3 trial; 1PE B1/B4와 4PE B1은 각1 trial.
- 반복 Compile 이점은 확인됐다. B1 PE scaling은 4PE 약2.006배, 8PE 약1.942배(n=1 기준). Eager는 개선이 작거나 느렸다.
- 최고 관측 처리량 4PE B1 Compile 546.32 images/s(n=1). PE당 처리량 최고는 1PE B4 Compile 381.72. 최적 조건 확정이나 배포 추천이 아니다.
- 4PE B1 Eager probe는180회, Compile140회가 필요해 정식 공통180회를 사용했다. 1PE B4는 공통140회다.
- 큰 배치 포화·전PE 핵심3·대표5·5분 안정성은 아직 미완료다. 8PE B4 Compile timeout은 미지원 판정이 아니다.
- SDK logger DEBUG 문제를 고친 v2를 현재 성능 규약으로 삼았다. v1 결과는 원본 보존하되 현재 집계에서 제외했다.
- 출력 허용 기준 pending, Compile 다중PE 활용과 task 누락 원인·전체 placement·내부 완료 barrier·codegen/cache는 미확인이다.

다음 우선순위는 비교 가능한 핵심 paired3, 가능한 B4/B16 frontier, 별도 장시간 안정성, 수치 허용 기준 및 Compile 다중PE 관측 보강이다. 상세 수치와 근거는 [report.md](report.md), 누락은 [coverage.csv](coverage.csv).
