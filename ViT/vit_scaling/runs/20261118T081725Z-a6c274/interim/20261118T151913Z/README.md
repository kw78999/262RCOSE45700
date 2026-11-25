# ViT RNGD 확장 실험 중간 보고서 스냅샷

**2026-11-19 00:19:13 KST / 2026-11-18 15:19:13 UTC** 기준. 원본 run: 20261118T081725Z-a6c274.

먼저 [report.md](report.md)를 읽는다. 수치 허용 기준 pending으로 성능은 provisional이다. 기준 시점 진행 중 worker는 통계에서 제외했다. 이 디렉터리는 재개 대상 run이 아니라 고정 보고 자료다.

- [실험 설정](experiment_config.json), [환경](environment.json), [manifest](manifest.json), [스냅샷 정보](snapshot_metadata.json)
- [PE 근거](pe_configuration.md), [입력·weights·state hash](assets.json), [evidence](evidence/)
- [timings.csv](timings.csv), [trial_summary.csv](trial_summary.csv), [condition_summary.csv](condition_summary.csv), [paired_speedups.csv](paired_speedups.csv), [scaling_summary.csv](scaling_summary.csv)
- [워밍업 판정 원본](stabilization_checks.json), [warmup_assessment.csv](warmup_assessment.csv), [초기 비용](initial_costs.csv), [break_even.csv](break_even.csv)
- [출력 검증](output_validation.json), [diagnostic summary](diagnostic_summary.json), [PE 관측](pe_observations.csv), [장치 상태](device_states.jsonl), [coverage](coverage.csv)
- [figures PNG/CSV](figures/), [feedback_summary](feedback_summary.md), [자동 기술 부록](technical_appendix.md)

원본 run은 기준 시점에 실행 중이었다. 상태 확인은 원본 manifest와 logs를 읽는다. 종료 뒤 미완료 조건만 재개하는 명령:

    cd /root/ViT
    /root/venv/bin/python -u vit_scaling/supervise.py --detach --run /root/ViT/vit_scaling/runs/20261118T081725Z-a6c274

기존 supervisor가 있으면 잠금으로 중복 실행을 막는다. 12시간 wall 예산은 최초 생성부터 계산한다. 이미 기한이 지나면 설정의 예산을 명시적으로 변경하기 전까지 추가 worker가 시작되지 않는다. 완료 attempt는 덮어쓰지 않는다.

스냅샷의 CSV는 원본 manifest와 case 결과를 CPU에서 재집계했다. 성능 worker는 변경하지 않았다. 입력 tensor·weights·cache는 원본 경로로 연결하고, 완료 case의 output tensor·trace는 이 디렉터리에 복사했다. raw result JSON의 기존 절대 경로는 원형을 보존했다.
