# ViT: 양쪽 MHA fastpath 해제·CPU fallback 차단 비교

run_id: `20261104T172746Z-e1b86b`. 현재 상태 `finished`. 주 성능 완료 12/12 worker. NPU worker 누적 5211.1/7200.0초, 개별 상한 1800초.

## 측정 규약과 자산

`torchvision.models.vit_b_16`, `ViT_B_16_Weights.IMAGENET1K_V1`, BF16, eval/inference_mode, B1/B4×3 trial. 실제 이미지 폴더는 `/root/ViT/data/vit_real_images_4/images/`다. 순서는 cat/coffee/astronaut/rocket이며 `/root/ViT/vit_pilot/followup_runs/20261104T132538Z-c031a6/inputs/preprocessed_bf16.pt`를 재사용했다. B1의 cat은 B4에도 포함돼 고유 사진은 4개다. 이미지 출처·순서·transforms 정보와 파일/텐서 해시는 image_manifest.json과 assets.json에 보존했다. 정답 라벨이 없어 accuracy/loss는 계산하지 않는다.

기존 common.timed_call을 그대로 사용한다: current stream 사전 synchronize → perf_counter_ns → runner(inputs) → synchronize → 종료 시각. 시간은 CPU dispatch·SDK 내부 전송·실행·완료 대기를 포함한 ms/batch-call이다. 외부 모델/입력 준비와 전송, 출력 복사/검증, 파일 쓰기, 로그는 모델 타이머 밖이다. S는 runner callable/torch.compile 래퍼 생성, F는 첫 호출, W는 후속 warmup 5회다. model_input_prepare_ms는 CPU 모델·파일 로드·state hash 검사·NPU 이동·준비 완료 대기를 포함한 참고값으로 A에서 제외한다. 순수 compiler 시간은 null이다. SDK Eager도 내부적으로 연산을 컴파일할 수 있어 Eager의 torch.compile backend 진입 0회를 실제 codegen 0회로 해석하지 않는다.

각 worker는 처음 없던 고유 SDK cache_context 경로를 사용하며 진단/성능/누적 실행은 분리된다. 시스템 전체 cold나 실제 disk miss를 보장하지 않는다. Compile에는 backend만 지정했고 dynamic/fullgraph/mode/options/recompile_limit·CPU 스레드는 조정하지 않았다. 전체 모델 compile 요청과 단일 완전 캡처는 구분한다. 안정성은 기존 CV≤0.10 및 전·후반 평균 차이≤10% 휴리스틱을 재사용했다. preliminary_stable도 steady state 증명은 아니다.

최초 두 B1 진단은 forward 전에 참조 state hash의 키 표현 차이로 실패했다. 공식 체크포인트의 구형 MLP linear_1/linear_2 이름과 torchvision 로더의 0/3 이름 차이를 확인했고, 공식 변환 뒤 152개 tensor 값이 정확히 같음을 검증했다. 최초 오류/키 기준 파일과 수정 코드 버전은 보존했으며 새 attempt로 재실행했다. 모델·타이머·SDK는 수정하지 않았다. logs/state_key_conversion.json과 manifest.code_revision_history를 참고한다.

## 새 실행 경로와 출력

| 케이스 | 상태 | native MHA host | NPU task | backend 진입 | SDK segment 목록 | recompile 로그 |
| --- | --- | --- | --- | --- | --- | --- |
| diagnostic_eager_b1_t1 | error | — | — | — | [] | 0 |
| diagnostic_compile_b1_t1 | error | — | — | — | [] | 0 |
| diagnostic_eager_b4_t1 | ok | 0 | 185 | 0 | [] | 0 |
| diagnostic_compile_b4_t1 | ok | 0 | 3 | 1 | [{'submodules': 1, 'to_compile': 1}] | 0 |
| diagnostic_eager_b1_t1_attempt1 | ok | 0 | 185 | 0 | [] | 0 |
| diagnostic_compile_b1_t1_attempt1 | ok | 0 | 2 | 1 | [{'submodules': 1, 'to_compile': 1}] | 0 |

상세 로그와 profiler는 진단 worker에만 켰다. 성능에서 컴파일 이벤트 부재를 단정하지 않는다. CPU host scope 시간은 NPU kernel 시간이 아니며 Task 이름을 codegen 비용으로 사용하지 않는다. 원본 trace/guard/graph break 위치는 diagnostic_observations.json, 설치 API 근거는 logs/installed_sources.txt 및 control_installed_sources.txt다.

| 배치 | 최대 절대오차 | 평균 절대오차 | cosine 평균/최소 | Top-1 일치 | 허용 오차 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.031250 | 0.004590 | 0.99994014 / 0.99994014 | 1/1 | pending |
| 4 | 0.062500 | 0.005142 | 0.99994004 / 0.99991642 | 4/4 | pending |

이 표는 이번 진단의 fastpath-off Eager를 이번 fastpath-off Compile과 직접 비교한 값이다. first/마지막 warmup/마지막 repeat 비교와 이미지별 클래스·logit·cosine은 output_comparison.json에 있다. Top-1 일치는 정답 accuracy가 아니다. NaN/Inf/구조 오류와 합의된 오차 위반은 비교를 보류하고, 합의되지 않은 허용 오차는 사후에 만들지 않는다.

