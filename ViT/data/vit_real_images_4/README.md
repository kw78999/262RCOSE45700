# ViT 실제 사진 4장 — 초기 출력 일치성·성능 점검용

사진 파일이 이미 포함되어 있다. 서버에 이 폴더를 복사하면 scikit-image 설치나 이미지 다운로드 없이 사용할 수 있다.

## 포함 이미지

| 순서 | 파일 | 내용 | 원본 크기(W×H) | 출처 표기 |
| --- | --- | --- | --- | --- |
| 1 | `images/01_cat.png` | 고양이 Chelsea | 451×300 | Stefan van der Walt, CC0 |
| 2 | `images/02_coffee.png` | 커피잔 | 600×400 | Rachel Michetti, CC0 |
| 3 | `images/03_astronaut.png` | 우주비행사 Eileen Collins | 512×512 | NASA, public domain |
| 4 | `images/04_rocket.jpg` | Falcon 9 발사 | 640×427 | SpaceX, public domain |

모두 실제 촬영 사진이며 생성형 AI 이미지나 랜덤 텐서가 아니다. 로컬에 설치된 scikit-image 0.19.3의 예제 파일을 바이트 그대로 복사했고, 공식 0.26.0 데이터 레지스트리에 등록된 SHA-256과 4개 모두 일치함을 확인했다. 파일명에 순서 접두사만 추가했다. 패키지 배포용 사진이므로 원본 촬영 파일과 동일한 해상도라는 의미는 아니다.

## 서버에서 준비하기

1. ZIP을 서버의 프로젝트 데이터 폴더에 복사하고 새 폴더에 압축 해제한다. 기존 실험 결과를 덮어쓰지 않는다.
2. 예를 들어 `/root/ViT/data/vit_real_images_4/` 아래에 README와 image_manifest.json이 있고, 그 아래 `images/`에 4장이 있는 구조로 둔다.
3. 아래 명령으로 파일 무결성을 확인한다. 경로는 실제 압축 해제 위치로 바꾼다.

```bash
/root/venv/bin/python /root/ViT/data/vit_real_images_4/prepare_images.py --verify-only
```

명령은 Python 표준 라이브러리만 사용한다. torch·SDK·torchvision·scikit-image를 설치하거나 변경하지 않는다. 파일이 없다면 `--verify-only`를 빼고 실행하면 고정된 공식 URL에서 누락 파일만 받고 해시를 검사한다. 기존 파일의 해시가 다르면 덮어쓰지 않고 중단한다.

## 후속 실험에 연결하기

- 모델 입력 폴더는 **`images/`만** 지정한다. 설명 파일을 이미지로 읽지 않는다.
- `image_manifest.json` 순서대로 입력을 고정한다. B1은 `01_cat`, B4는 네 이미지 전체다.
- 서버에서 `Image.open(...).convert('RGB')` 후 `ViT_B_16_Weights.IMAGENET1K_V1.transforms()`를 적용한다.
- 원본 사진을 미리 224×224로 강제 변형하지 않는다. 모델 가중치의 전처리에 리사이즈·크롭·정규화를 맡긴다.
- 전처리는 한 번 수행해 저장하고, Eager·Compile 및 모든 trial에서 같은 전처리 완료 텐서를 사용한다.
- 사진 읽기·전처리·명시적인 NPU 입력 전송은 모델 호출 시간에서 제외한다.
- 기존 후속 CLI에 이미지 경로를 전달할 때는 실제 `--help`와 인자 정의를 확인한다. `--images`가 폴더인지 파일 목록인지 추측하지 않는다.
- 구체적인 Codex 전달 문구는 `CODEX_INPUT.md`에 있다.

## 정답 라벨에 관한 주의

이 묶음에는 **ImageNet 정답 라벨이 없다.** Manifest의 description은 사람이 파일을 알아보는 설명이며, ground_truth_label은 모두 null이다. 설명·파일명·모델 예측을 정답으로 바꾸지 않는다.

가능한 평가:

- Eager·Compile logits의 최대·평균 절대오차, 코사인 유사도.
- 두 모드의 Top-1 예측 일치율.
- 같은 입력 조건에서의 지연시간·처리량과 독립 실행 재현성.

이 자료만으로 분류 정확도나 전체 ImageNet 성능을 계산하지 않는다. 이미지 4장은 입력 경로와 출력 일치성을 점검할 소규모 예제이며 전체 입력 분포를 대표하지 않는다. 모델 정답 정확도 평가는 후속 단계에서 해당 작업에 맞는 라벨된 데이터셋으로 수행한다.

## 출처와 재현

- 개별 저작자·이용 조건·출처·고정 다운로드 URL: `image_manifest.json`, `SOURCES.md`.
- 파일별 SHA-256: `SHA256SUMS.txt`.
- 무결성 검사 및 누락 파일 재다운로드: `prepare_images.py`.
- 사진 원본에는 주석·워터마크·합성·보정·재인코딩을 적용하지 않았다.

