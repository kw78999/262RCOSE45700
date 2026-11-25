# ViT RNGD PE·batch 확장 실험

기존 vit_pilot과 결과·캐시·SDK는 변경하지 않는 별도 실행 진입점이다. 기존 최종 run은 20261104T172746Z-e1b86b, 새 run은 [20261118T081725Z-a6c274](runs/20261118T081725Z-a6c274/README.md)이다. 상세 결과와 미측정 사유는 해당 run의 report.md / coverage.csv / manifest.json 참조.

[중간 보고서: 2026-11-19 00:19:13 KST 기준](runs/20261118T081725Z-a6c274/interim/20261118T151913Z/report.md)은 완료 결과를 고정한 별도 스냅샷이다. 9개 그림과 대응 CSV, trial별 통계, 출력·PE 관측 근거를 포함한다.

```bash
cd /root/ViT
# 새 run 생성, 환경·입력·설정·SDK 근거 준비
/root/venv/bin/python -u vit_scaling/experiment.py init
# 생성된 경로로 실행·재개 (새 worker를 중복 실행하지 않음)
/root/venv/bin/python -u vit_scaling/supervise.py --detach --run /absolute/path/to/run
# 원본에서 재집계 (NPU forward 없음)
/root/venv/bin/python vit_scaling/experiment.py analyze --run /absolute/path/to/run
```

기본 예산12시간, worker3600초, stability 모드당300초. experiment_config.json에서 변경한다. wall elapsed 예산은 최초 run 생성 시각부터이며 정지 시간도 포함한다. 단일 controller/단일 NPU worker 락을 사용한다. 다른 사용자의 NPU process가 있으면 보류하며 장치 reset·타인 프로세스 종료를 하지 않는다. 독립 trial은 새 프로세스/모델/runner/cache 경로다. controller 재개는 살아 있는 기존 자기 worker를 먼저 회수하고 같은 trial로 보존한다.

정식 측정은 별도 안정화 탐색에서 선택한 공통 warmup≥100, repeat≥500 및 measured sum≥10초, repeat cap10000. exploration은 first1+warm20+repeat50, 정식과 합치지 않는다. 수치 허용 기준 pending이므로 모든 성능은 provisional이다. 단일 fused logical device에 원래 ViT 하나를 올리며 replica·LLM tensor_parallel_size·하위 compile·모델 수식 변경은 사용하지 않는다. trace의 Task 수는 kernel 수/fusion 수가 아니다.

CPU fixture 검증: `/root/venv/bin/python -m unittest discover -s vit_scaling/tests -v`. fixture는 /tmp의 임시 디렉터리에만 생성되며 NPU 실측과 섞지 않는다.
