# RNGD ViT PE·batch 확장 실험

## 검토된 현재 상태

측정 제어는 **2026-11-19 04:22:05 KST**에 종료됐다. 실험은 부분 완료이며, 현재 규약 정식 trial36개·paired18쌍이 유효하다. 재개 후 assert.h 미발견 C 컴파일 오류10건으로 대표5 및 stability는 완료하지 못했다. 당시 예산은 약55.3분 남아 있었다.

[검토·보완된 보고서](report.md), [검토 기록](report_review.md), [오류 원인 CSV](failure_summary.csv), [피드백](feedback_summary.md)을 먼저 읽는다. 기존 실행/재개 명령은 아래 보존했다. 추가 측정에는 공통 컴파일 환경 복구와 별도 예산이 필요하다. 이번 보고서 검토에서는 NPU 재실행·환경 설치·SDK/드라이버 변경을 하지 않았다.

run `20261118T081725Z-a6c274`. 모델/가중치/BF16/224×224/eval/inference_mode 고정. 양쪽 MHA fastpath=False, SDK CPU fallback=False. Compile backend만 지정. 별도 단일 worker의 단일 모델을 fused logical device 하나에서 호출한다. 결과는 허용 오차 pending인 provisional 자료다.

```bash
cd /root/ViT
/root/venv/bin/python -u vit_scaling/supervise.py --detach --run /root/ViT/vit_scaling/runs/20261118T081725Z-a6c274
/root/venv/bin/python vit_scaling/experiment.py analyze --run /root/ViT/vit_scaling/runs/20261118T081725Z-a6c274
```

새 실험: `/root/venv/bin/python -u vit_scaling/experiment.py run [--config /path/config.json]`. 위 resume 한 명령이 완료하지 않은 logical case만 새 attempt/cache로 이어간다. 완료 결과를 덮어쓰지 않는다. `--retry-errors`를 명시하면 오류 logical case도 새 attempt로 재시도한다. 실행 전 config에서 total_elapsed_budget_s, worker_timeout_s, 안정성 시간 등을 변경할 수 있으나 버전/정밀도/모델/경로를 바꾸지 않는다. wall elapsed 예산은 최초 생성 시각부터다. 중단 시간이 예산에 포함된다. `manifest.json`이 실제 시각·순서·PID·attempt·보류의 기준이다. 다른 사용자의 NPU 프로세스가 있으면 새 worker를 시작하지 않는다. 장치 원자 예약은 제공하지 않는다.

원본 `cases/*/result.json`, 통합 `timings.csv`, `trial_summary.csv`, `scaling_summary.csv`, `output_validation.json`, `device_states.jsonl`, `figures/*.png`와 대응 CSV, `report.md` 참조. 모든 호출은 CPU dispatch 포함 batch-call 완료 지연시간이며 순수 kernel 시간이 아니다. batch latency/B는 amortized ms/image이며 개별 요청 응답시간이 아니다.

장시간 실행은 supervisor가 독립 OS session에서 controller를 실행한다. 기존 controller/worker가 살아 있으면 중복 실행하지 않고 관측한다. tool session에서 controller가 종료되어도 기존 자기 worker를 회수한 뒤 미완료 case만 재개한다. 독립 trial 수는 증가하지 않는다. supervisor는3회 연속 controller 종료 후 추가 자동 재시도를 보류한다. 원인은 logs/ 참조.


## 로그 규약 수정 (2026-11-18 10:32 UTC)

SDK Python logger 기본 DEBUG를 발견해 명시적 WARNING 설정을 적용한 worker_protocol=2를 추가했다. code/worker.py(v1)는 보존하고 code/worker_v2.py로 새 attempt를 실행한다. 모델·정밀도·backend·timed_call·반복 규약은 동일하다. v1 성능 원본은 보존하되 현재 집계에서 제외한다. 해당 단계의 재측정 시간도 최초 생성 시각부터의 12시간 예산에 포함된다. 상세 근거와 SHA256은 logs/logging_protocol_correction.json. 진단은 계속 별도 DEBUG worker다.


## 2026-11-18 14:06 UTC 재개 시 우선순위

짧은 수명 PID의 소유권을 확인하지 못한 exploration_p4_b4_compile_t1_a1은 contaminated로 유지한다. /proc identity를 SMI ps 실행 중에만 5ms 간격으로 추가 관측하며 장치 조회 자체는 기존 30초 간격을 유지한다. 사라진 PID는 여전히 소유권 미확인으로 처리한다. supervisor는 외부 점유/미확인 상태에서 두 번 연속 장치가 비어 있는 것을 확인한 뒤 새 attempt로 재개하며 타인 PID를 종료하지 않는다. 8PE B4 Compile은 실제 3600초 cap timeout이다.

핵심 paired3 trial을 우선 완료한다. 이후 대표5 trial 대상을 최대2 조합(완료된 핵심 중 baseline B1 및 Compile 처리량 중앙값이 가장 높은 조합)으로 제한한다. 대표5/stability 예산을 남길 수 있는 경우에만 B16/64/128 우선의 post-core 탐색을 추가한다. 이 단계도 PE 순환·worker cap·고정 경로·별도 validation/exploration을 유지한다. 미측정 큰 배치를 unsupported로 선언하지 않는다. 상세 코드는 controller_revision_11, 변경 사유는 logs/controller_revision_11.json.
