# Attention CPU fallback 연구 — 예비 실험

## 확인한 원인

[실행 확인/소스 확인] PyTorch native MHA 연산은 PrivateUse1 전용 kernel이 없고 SDK backend fallback이 등록되어 있다. eval/inference_mode의 ViT self-attention이 이 fastpath를 선택한다. strict 오류와 CPU profiler를 함께 근거로 사용한다. torch의 +cpu 버전 표기나 output.device만으로 판정하지 않는다.

[소스 확인] torch/nn/modules/activation.py:1338–1445 및 _check_arg_device는 PrivateUse1 장치를 허용하지만 native kernel 존재를 검사하지 않는다. torch/_dynamo/variables/torch.py:213은 tracing 중 _is_make_fx_tracing을 True로 취급하므로 Compile은 native 경로를 건너뛴다. 기존 전체 Compile FX graph에서도 functional.multi_head_attention_forward를 관측했다. furiosa/torch/_db/local_decomp.py:1392,1452,2478–2479에는 SDPA math 분해가 있다. 설치 소스 사본·해시는 logs/attention_installed_sources.txt와 manifest에 보존한다.

## 이번 실행

실제 입력은 `/root/ViT/vit_pilot/followup_runs/20261104T132538Z-c031a6/inputs/preprocessed_bf16.pt`를 다시 사용했다. 고유 사진 4개, B1은 첫 cat, B4는 전체 4개이며 이미지당 토큰은 197개다. 입력 파일/전처리 텐서/가중치 해시는 assets.json, 원본 출처·순서·전처리는 image_manifest.json에 보존했다. 정답 라벨 없이 logits와 예측 일치성만 비교한다.

Attention만 Compile은 기존 12개 self_attention 모듈에 Module.compile(backend=ft.backend)을 한 번씩 적용한다. 나머지 ViT는 Eager다. 이번 run에서는 각 기존 MHA를 그대로 호출하는 사용자 코드 AttentionBoundary를 준비 단계에 두고 이 경계를 compile한다. 원래 Parameter 객체 집합을 그대로 유지하는지 확인한다. state_dict 경로에 original_attention이 추가되지만 파라미터 값과 호출 인자는 변경하지 않는다. Attention 수식·가중치·need_weights=False·dtype·해상도를 바꾸지 않는다. fastpath-off Eager는 공개 API로 worker 내부 실행 경로만 변경하고 원래 값으로 복원한다. 이는 기본 Eager와 구분한 ablation이다. Compile dynamic/fullgraph/mode/recompile_limit은 조정하지 않았다.

CPU fallback 금지 진단과 profiler는 별도 worker에서 수행하고, 성능 worker에는 상세 로그·프로파일·backend observer를 넣지 않았다. warmup 5회, repeat 20회, 독립 trial 1회인 초기 진단이다. 시간은 전체 ViT 배치 호출부터 공식 current-stream 완료까지의 CPU 관측 ms다. 내부 dispatch·복사·대기는 포함하고 외부 전처리·준비·출력 복사는 제외한다. 순수 Attention/NPU kernel 시간은 아니다.

| 조건 | B | 상태 | setup ms | first ms | repeat mean ± sample SD ms | median ms | images/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| attention_compile | 1 | ok | 149.1720 | 113301.6091 | 40.8857 ± 0.1725 | 40.9425 | 24.4584 |
| attention_compile | 4 | ok | 149.0097 | 208352.8310 | 118.1715 ± 0.0719 | 118.1593 | 33.8491 |

기본 Eager는 CPU fallback이 포함된 혼합 실행 기준선이다. comparisons.csv의 비율은 이 혼합 기준선 대비 전체 호출 시간의 예비 비율이며 NPU-only 가속률이 아니다. 서로 다른 batch의 비율은 평균내지 않는다. 느린 정상 호출을 삭제하지 않았으며 throughput은 순차 반복 처리량이다. first-call은 순수 compile 시간이 아니다.

| 진단 | strict | 상태 | CPU native MHA 횟수 | NPU task events | backend 진입 | limit 경고 |
| --- | --- | --- | --- | --- | --- | --- |
| diagnostic_attention_compile_b1_strict | True | ok | 0 | 137 | 1 | False |
| diagnostic_attention_compile_b4_strict | True | ok | 0 | 137 | 1 | False |

strict 성공은 해당 SDK fallback 제어 계약 범위의 결과이며 전체 연산 배치의 완전한 증명은 아니다. CPU host span은 NPU kernel 시간이 아니고 중첩 구간을 더해 원인별 비중을 만들지 않는다. NPU task 수, backend 진입, 실제 codegen 수, disk cache hit/miss는 별개다. 마지막 두 항목은 미확인(null)이다. 각 worker의 SDK cache는 처음 없던 전용 경로이며, 시스템 전체 cold 또는 실제 miss를 보장하지 않는다.

| 조건 | 참조 | max abs | mean abs | cos mean | cos min | Top-1 분자/분모 |
| --- | --- | --- | --- | --- | --- | --- |
| performance_attention_compile_b1 | eager | 0.0312 | 0.0057 | 0.9999 | 0.9999 | 1/1 |
| performance_attention_compile_b1 | compile | 0.0312 | 0.0043 | 0.9999 | 0.9999 | 1/1 |
| performance_attention_compile_b4 | eager | 0.0469 | 0.0051 | 0.9999 | 0.9999 | 4/4 |
| performance_attention_compile_b4 | compile | 0.0625 | 0.0048 | 0.9999 | 0.9999 | 4/4 |

출력 허용 오차는 pending이다. Top-1 일치율은 정답 accuracy가 아니며 B1과 B4를 고유 이미지 5개로 합산하지 않는다.

## 해결 방향과 해석 범위

1. 전체 Compile: 기존 strict B1/B4 성공 근거가 있고 새 control 시간을 위 표에 남겼다.
2. Attention만 Compile: 모든 12개 모듈을 감싸되 strict·FX graph·CPU profiler·출력·전체 호출 시간으로 실제 효과를 판단한다. 나머지 Eager dispatch와 fusion 경계 비용이 남을 수 있다.
3. MHA fastpath 해제: native 연산 대신 기존 functional MHA/SDPA 경로를 선택하는 별도 조건이다. 다른 미지원 연산이 나타나면 설정을 숨기거나 수정하지 않는다.
4. 근본 지원 개선: SDK 공급자에게 native MHA PrivateUse1 kernel 또는 kernel 지원을 고려한 dispatch 처리를 문의할 근거가 된다. 현재 환경 업그레이드·커널 개발은 수행하지 않는다.

worker 기본 상한 1800초, 전체 NPU worker 예산 5550.374782020692초, 실제 누적 670.7초. 타임아웃은 미지원 확정이 아니다. 원본 합성 및 실제 이미지 후속 run은 변경하지 않았다. 새 실험은 이전 범위에 추가된 사용자 요청의 Attention 부분 compile 연구로 분리했다.

## 재현

```bash
cd /root/ViT
/root/venv/bin/python -u vit_pilot/attention.py run
/root/venv/bin/python vit_pilot/attention.py analyze --run /root/ViT/vit_pilot/attention_runs/20261104T160225Z-35e28e
```

공식 참고: [PyTorch 2.10 MHA](https://docs.pytorch.org/docs/2.10/generated/torch.nn.MultiheadAttention.html), [mha backend 제어](https://docs.pytorch.org/docs/2.10/backends.html#torch-backends-mha). 실제 설치 소스와 실행 로그가 이 환경의 직접 근거다.
