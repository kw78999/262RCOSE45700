# ViT 후속 실험 재현

원래 합성 실험 `runs/20261104T113745Z-8fccff/`는 변경하지 않는다. 후속 결과는 `followup_runs/20261104T132538Z-c031a6/`에 따로 보존한다. 기존 `common.py` 타이머, `vit.py` 모델/입력 준비 및 `analyze.py` 출력 오차 계산을 재사용한다. 새 진입점은 followup.py와 followup_worker.py이며 성능 worker에는 진단 wrapper/profiler/hook을 넣지 않는다.

이미지 묶음 최종 위치는 `/root/ViT/data/vit_real_images_4/`, 실제 입력 폴더는 `/root/ViT/data/vit_real_images_4/images/`이다. 사용자 제공 ZIP의 README·image_manifest·SOURCES를 읽고 다음 검증을 실행해 네 파일의 SHA256을 확인했다. 이미지 검색·생성·재다운로드는 하지 않았다. 제공된 출처·순서·null 정답 라벨을 보존한다.

```bash
/root/venv/bin/python /root/ViT/data/vit_real_images_4/prepare_images.py --verify-only
```

기존 `pilot.py --images`와 새 `followup.py --images` 모두 **파일 4개 목록**을 받는다. 폴더를 인자로 전달하지 않는다. 새 실행은 항상 새 run_id를 만든다.

```bash
cd /root/ViT
/root/venv/bin/python -u vit_pilot/followup.py run \
  --images \
  /root/ViT/data/vit_real_images_4/images/01_cat.png \
  /root/ViT/data/vit_real_images_4/images/02_coffee.png \
  /root/ViT/data/vit_real_images_4/images/03_astronaut.png \
  /root/ViT/data/vit_real_images_4/images/04_rocket.jpg
```

CPU에서 RGB 변환 및 `ViT_B_16_Weights.IMAGENET1K_V1.transforms()`를 한 번 적용해 FP32와 BF16 텐서를 저장한다. 모든 실제 검증/성능 worker는 같은 저장된 BF16 텐서를 로드한다. B1은 cat, B4는 전체 네 장. 원본 이미지 파일 해시, 직렬화 파일 해시 및 tensor 바이트 해시는 서로 구분한다. 출처 manifest 사본 및 확장 manifest에 절대 경로·순서·버전·전처리 정보를 기록한다. synthetic_input=false이며 정답 라벨이 없어 accuracy/loss를 계산하지 않는다. category 이름은 모델 예측값이다.

실행 순서는 원래 합성 입력으로 짧은 경로 진단 → 별도 strict 진단 → 실제 이미지 기본 출력 검증 → 유효한 mode/batch별 독립 trial이다. CPU tensor fallback이 확인된 조건은 성능 확대를 보류한다. 출력 쌍에 NaN/Inf·구조 오류·Top-1 불일치·명시한 허용 오차 위반이 있으면 해당 batch 확대를 보류한다. 허용 오차 미정이나 전체 장치 배치 미확인만으로 독립적인 유효 조건까지 중단하지 않는다.

기본 성능은 trial 3회·warmup 5회·repeat 50회, mode×batch×trial=12개 worker(모두 성공하면 672개 시간)다. 보류한 worker는 원본 상태와 이유를 남기고 0ms 또는 가상 결과로 채우지 않는다. Eager가 보류되면 Compile 자체 trial을 수집할 수 있으나 paired speedup은 계산하지 않는다. 요청된 trial 순서의 상대 순서를 유지한다.

진단/실제 검증/성능 cache는 분리하고 성능은 worker별 처음 존재하지 않는 새 SDK cache_context 경로를 쓴다. 이는 시스템 전체 cold나 실제 disk miss를 보장하지 않는다. 기존 캐시는 삭제하지 않는다. 각 worker 기본 1800초, 전체 실행 NPU 예산 7200초이며 CLI의 --worker-timeout/--total-budget으로 조정 가능하다. 재개할 때 이미 사용한 예산을 차감한다. NPU worker는 항상 직렬이고 자기 process group만 정리한다.

```bash
# 새 run을 단계 경계에서 중단
python -u vit_pilot/followup.py run --stop-after diagnostic --images FILE1 FILE2 FILE3 FILE4
# 위 run의 미실시 단계만 재개; 실행한 worker는 덮어쓰지 않음
python -u vit_pilot/followup.py resume --run-dir /ABS/RUN
# 실패/보류한 diagnostic 또는 strict를 새 ID로 재시도할 때만 사용
python -u vit_pilot/followup.py resume --run-dir /ABS/RUN --retry-case CASE_ID
# NPU forward 없이 저장된 원본으로 재집계
python vit_pilot/followup.py analyze --run-dir /ABS/RUN
python -m unittest discover -s vit_pilot/tests -p test_followup.py -v
```

이번 실행 중 Compile 진단 wrapper의 utc 중복 인자 오류를 수정했고, 최초 실패와 자체 중단된 strict worker를 덮어쓰지 않고 `_retry1` ID로 보존했다. 이 오류는 SDK 지원 실패가 아니다. 최초/수정 코드 해시, 재개 명령 및 누적 사용 예산은 manifest.json과 resume_history에 기록한다. 수정은 진단 표식 및 재개 제어에 한정하며 모델/입력/측정 타이머는 바꾸지 않았다.

그림은 기존 프로젝트 전용 `.plot-venv`를 재사용한다. torch/SDK/드라이버/전역 설정을 설치하거나 변경하지 않는다. 실제 입력·로그/trace·출력·timings.csv·trial_summary.csv·comparisons.csv·across_trial_summary.csv·artifact_audit.json과 보고서가 각 followup run에 보존된다. B8·B16, Attention 구현 교체, 모듈 독립 벤치마크는 수행하지 않는다.
