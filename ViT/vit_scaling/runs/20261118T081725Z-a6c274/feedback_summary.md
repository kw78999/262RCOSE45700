# 실험 피드백

run 20261118T081725Z-a6c274: 제어 루프는 종료됐으나 목표 일부 미완료. 모든 성능은 수치 허용 기준 pending인 provisional 자료다.

- 현재 규약 정식 mode trial 36개, paired trial 18쌍. B1은 PE1/2/4/8, B4는 PE1/4에서 각 mode 3 trial이다.
- 최고 관측 처리량: PE4 B4 compile: 721.05 images/s (n=3). 가장 빠른 trial을 고르지 않은 중앙값이다.
- PE당 최고 관측 처리량: PE1 B4 compile: 381.77 images/s/할당 PE. utilization 측정이 아니다.
- 대표5 미완료: [(1, 1, 'eager', 2), (1, 1, 'compile', 2), (4, 4, 'eager', 2), (4, 4, 'compile', 2)]. Stability 성공/시도 0/2.
- assert.h 미발견 C 컴파일 환경 오류 10건. 실패 조건에서 유효 반복 시간은 얻지 못했다. SDK/ViT/PE 미지원 판정이 아니다.
- 측정 종료 2026-11-19 04:22:05 KST, 예산 약 55.3분 잔여. 예산 소진만으로 모든 미완료를 설명하지 않는다.
- 큰 배치 포화점·장시간 안정성은 미확인. 수치 허용 기준·전체 op placement·PE별 활용·codegen/cache도 미확인이다.

세부 통계와 그림은 [report.md](report.md), 원인별 실패는 [failure_summary.csv](failure_summary.csv), 누락·부족 trial은 [coverage.csv](coverage.csv). 다음 우선순위는 공통 컴파일 환경 복구 후 별도 예산으로 대표5·stability와 B16 비교를 보완하는 것이다.
