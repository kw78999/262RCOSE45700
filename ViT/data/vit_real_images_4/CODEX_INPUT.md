# 기존 ViT 후속 실험 프롬프트에 추가할 입력 지정

실제 이미지 4장을 준비했어. 첨부한 vit_real_images_4.zip 또는 압축 해제 폴더를 사용해줘.

- 서버 프로젝트의 데이터 폴더에 기존 파일을 덮어쓰지 않고 배치하고, 최종 절대 경로를 알려줘.
- 준비된 폴더의 README.md와 image_manifest.json을 읽고 prepare_images.py --verify-only로 해시를 확인해줘.
- 실제 입력 폴더는 이 묶음의 images/야. 다른 이미지를 검색하거나 새 이미지를 생성할 필요 없어.
- 파일 순서는 01_cat.png, 02_coffee.png, 03_astronaut.png, 04_rocket.jpg로 고정해줘.
- B1은 첫 사진, B4는 전체 네 사진이야. 동일한 전처리 완료 텐서를 두 모드와 세 독립 trial에 공유해줘.
- 가중치 ViT_B_16_Weights.IMAGENET1K_V1의 transforms()로 전처리하고, 입력 파일 해시와 전처리 텐서 해시는 구분해서 기록해줘.
- 제공된 image_manifest.json의 출처와 순서를 보존하면서 실험 결과 폴더에 실제 입력 경로·전처리 정보를 기록해줘.
- 이 사진들은 실제 사진이며 synthetic_input=false야. ImageNet 정답 라벨은 없으므로 accuracy/loss는 계산하지 말고 logits 오차와 두 모드의 예측 일치성을 비교해줘. 설명을 정답으로 쓰지 마.
- 기존 후속 실험 프롬프트의 실행 경로 진단 → 실제 이미지 검증 → 조건이 유효하면 독립 3회 재측정 순서를 이어서 수행해줘.
- 기존 CLI의 --images 인자 형식을 확인한 뒤 정확한 실행 명령을 구성해줘. 원본 합성 입력 결과와 새 실제 이미지 결과는 분리해서 보존해줘.

