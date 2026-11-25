# ViT fastpath-off 동일 조건 비교 재현

실제 실행 Python `/root/venv/bin/python`, `3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0]`. torch `2.10.0+cpu`, torchvision `0.25.0+cpu`, Furiosa `furiosa-torch 2026.4.0 (rev: 5800e2ba24, built_at: 2026-10-12T09:11:33Z)`. 드라이버/펌웨어/장치 상세는 environment.json의 furiosa-smi 기록을 참고한다.

```bash
cd /root/ViT
/root/venv/bin/python -u vit_pilot/fastpath_compare.py run --stop-after diagnostic
/root/venv/bin/python -u vit_pilot/fastpath_compare.py resume --run-dir /root/ViT/vit_pilot/fastpath_compare_runs/20261104T172746Z-e1b86b --stop-after performance
/root/venv/bin/python -u vit_pilot/fastpath_compare.py resume --run-dir /root/ViT/vit_pilot/fastpath_compare_runs/20261104T172746Z-e1b86b --stop-after cumulative
/root/venv/bin/python vit_pilot/fastpath_compare.py analyze --run-dir /root/ViT/vit_pilot/fastpath_compare_runs/20261104T172746Z-e1b86b
vit_pilot/.plot-venv/bin/python vit_pilot/fastpath_compare_plots.py --run /root/ViT/vit_pilot/fastpath_compare_runs/20261104T172746Z-e1b86b
/root/venv/bin/python vit_pilot/fastpath_compare_verify.py --run /root/ViT/vit_pilot/fastpath_compare_runs/20261104T172746Z-e1b86b
python -m unittest discover -s vit_pilot/tests/fastpath_compare -p test_math.py -v
```

기본 run은 세 단계 모두 실행한다. 단계별 stop/resume는 완료된 최신 case를 재실행하지 않고 미실시만 진행한다. 누적 NPU worker wall budget은 7200초이며 resume에서도 이미 사용한 시간을 유지한다. worker 1800초, repeat 50, warmup 5, trials 3, cumulative cap 10000이 기본이다. CLI `--worker-timeout`, `--total-budget`, `--warmup`, `--repeats`, `--trials`, `--max-cumulative-calls`를 지원하며 예산을 자동 연장하지 않는다. resume에서 명시적인 budget/timeout 변경만 적용한다. 오류 재시도는 `--retry-case CASE_ID --retry-reason TEXT`로 새 attempt/cache 경로를 만들고 최초 결과와 코드 revision을 보존한다. 성공 case를 덮어쓰지 않는다. 자신의 process group만 정리하고 다른 사용자의 NPU 사용이 있으면 보류한다.

입력은 `/root/ViT/vit_pilot/followup_runs/20261104T132538Z-c031a6/inputs/preprocessed_bf16.pt`의 byte-identical 텐서다. 이미지 재생성/재전처리 없이 cat/coffee/astronaut/rocket을 사용한다. 원본 파일/텐서/가중치 파일/state_dict 해시는 서로 다른 의미로 저장한다. checkpoint의 구형 MLP 키는 torchvision 공식 loader가 변환하므로 state_dict 모델 해시는 loader 뒤 CPU FP32 및 CPU BF16을 기준으로 한다. NPU 이동 뒤 전체 모델을 CPU로 복사해 검증한 것은 아니다.

양쪽 모두 `torch.backends.mha.set_fastpath_enabled(False)`와 `ft.config.cpu_fallback(False)`를 forward/추적 전에 적용한다. Eager는 원래 모델 직접 호출, Compile은 전체 모델의 기본 `torch.compile(model,backend=ft.backend)`다. 하위 compile/AttentionBoundary/수식 변경은 없다. current_stream synchronize는 설치 SDK 계약을 사용한다. 준비/출력 검사와 계산/저장은 모델 타이머 밖이다. first/마지막 warmup/마지막 repeat의 출력을 저장하며 별도 cumulative 실행에서는 매 work 출력의 유한값을 타이머 밖에서 확인한다.

프로파일·상세 로그·backend observer는 진단 worker만 사용한다. 모든 worker는 고유 SDK cache_context 경로를 사용하며 system cold/실제 disk miss는 미확인이다. 순수 컴파일 시간은 null이다. 표준편차는 ddof=1, images/s=B×유효 반복 수/시간 합계다. 호출 시간은 CPU dispatch와 SDK 내부 비용을 포함한다. atol/rtol 기본은 미정이고, 합의된 값이 있을 때만 `--atol --rtol --tolerance-source`로 명시한다. 라벨이 없으므로 accuracy/loss를 계산하지 않는다.

손익분기 fixture는 tests/fastpath_compare에 분리됐고 NPU 시간/보고서에 섞지 않는다. main 성능 12개 완료 시 672행, 별도 진단/누적 기록은 분리한다. manifest의 all_workers_complete는 최신 logical case 전체, all_unheld_workers_complete는 그중 보류 제외, all_attempts_successful은 보존한 과거 실패까지 포함한 값이다.
