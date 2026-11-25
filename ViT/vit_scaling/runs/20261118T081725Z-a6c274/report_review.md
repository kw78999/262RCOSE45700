# 보고서 검토 결과

검토 2026-11-18T23:54:39.599167+00:00. 판정: **passed — 원본과 통계·문서의 일관성 검토**. 전체 실험 목표 완료나 출력 수치 허용 기준의 합격 판정은 아니다.

- 현재 규약 formal 36개, PE×batch×mode 12개, paired 18쌍을 원본 repeat에서 독립 재계산했다.
- 평균·중앙값·표본 SD·CV·p95/p99·측정 합계·images/s, trial 그룹 중앙값·범위·SD·CV, paired 배속·PE scaling·상대 효율을 대조했다.
- raw timings 81,789행과 case 원본을 대조했다. 원본 case 파일 522개 및 worker manifest 기록의 불변을 확인했다.
- trial_summary·condition_summary·paired_speedups·scaling_summary·output_validation은 보완 전과 바이트 단위로 동일하다.
- 출력 14개를 CPU에서 다시 읽어 shape·finite·입력 식별을 확인하고 저장된 오차·Top1 수치를 대조했다. 허용 오차 pending은 유지했다.
- PNG 9개·대응 CSV·보고서 링크를 확인했다. 성공 stability 0개로 추세 그림에 측정값을 만들지 않았다.

## 발견해 보완한 내용

1. 재개 후 10개 error의 원인은 공통으로 SDK 내부 C 컴파일의 assert.h 미발견이다. 모델/PE 미지원으로 분류하지 않았으며 [failure_summary.csv](failure_summary.csv)에 기록했다.
2. 대표5와 stability를 완료하지 못한 이유를 보고서 앞에 명시했다. 제어 루프 종료를 전체 연구 성공으로 표현하지 않았다.
3. 실제 측정 종료는 2026-11-19 04:22:05 KST이며 약55.3분 예산이 남아 있었다. 나중에 보고서를 재집계한 시간을 측정 elapsed에 더하지 않도록 고쳤다.
4. coverage에서는 실제 환경 오류·timeout·중단을 과거 budget hold 문구보다 먼저 표시하며, 하위 배치 timeout에 따른 확대 보류를 명시했다.
5. 전체 이벤트 14개 중 실제 보류는9개다. 기존의 모든 이벤트를 보류로 세던 서술을 수정했다.
6. trial 평균 CV 단위를 %로 명시하고 B1→B4 처리량·지연시간 표, 그림9개와 CSV 링크, 읽을 수 있는 feedback 요약을 추가했다.

## 남은 한계

대표5·300초 stability·B16 이상 정식 비교, 수치 허용 기준, Compile 다중PE 실제 활용·모든 op placement·binary 내부 동기화·codegen/cache는 미확인이다. 공통 환경 오류를 제어기가 전체 중단 사유로 판단하지 못한 실행 제어의 한계도 남아 있다. 이번 작업은 보고서 검토이며 환경 복구·SDK/드라이버 변경·새 NPU 실험은 수행하지 않았다.

원본 자동 보고서와 CSV는 reviews/20261118T234659Z/에 보존했다. 현재 [report.md](report.md), [기계 판독 검토 결과](report_review.json), [feedback_summary.md](feedback_summary.md)를 함께 제공한다.
