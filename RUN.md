# 실행 방법

## 설치

개발 환경: Windows 11, **Python 3.10** (3.10.16에서 개발)

```bash
conda create -n deepsign python=3.10
conda activate deepsign
pip install -r requirements.txt
```

- `requirements.txt`의 버전은 고정해 두었습니다. 최신 버전을 설치하면 실행되지 않을 수 있습니다.
  - 모델 파일(`.h5`)이 Keras 2.10.0으로 저장되어 있어 **TensorFlow 2.10.1**이 필요합니다. TensorFlow 2.10은 Python 3.7~3.10만 지원합니다.
  - `scaler_*.pkl`, `label_encoder_*.pkl`은 **scikit-learn 1.3.0**으로 저장되어 있습니다.
  - `mp.solutions.hands` API를 쓰기 때문에 **MediaPipe 0.10.5**를 사용합니다.
- GPU를 쓰려면 **CUDA 11.2**와 **cuDNN 8.1**을 설치하세요. TensorFlow 2.10은 Windows에서 GPU를 직접 지원하는 마지막 버전입니다. GPU가 없으면 CPU로 실행됩니다.

## 실시간 인식

```bash
python IPWebcam_test.py
```

- 실행하기 전에 스크립트 안의 IP Webcam 주소(`http://<폰 IP>:8080/video`)를 자신의 환경에 맞게 바꿔 주세요.
- 키 조작: `1` 한글 모델, `2` 숫자 모델, `ESC` 종료

학습된 모델(`.h5`, `.pkl`)은 저장소에 있으므로 실시간 인식만 할 때는 데이터셋이 필요 없습니다.

## 데이터셋과 모델 직접 만들기

사진부터 새로 모으거나, [README의 데이터셋](README.md#데이터셋)에서 받은 사진으로 CSV와 모델을 다시 만들려면 다음 순서로 실행합니다.

```text
collect_datasets_IPWebcam.py   →  dataset_raw/<라벨>/*.jpg
get_hand_coords_csv.py         →  digits.csv, hangul.csv
get_featured_data_csv.py       →  *_feature.csv
sign_language_distinction_MLP_model_train.py  →  model_*.h5, scaler_*.pkl, label_encoder_*.pkl
```

> 각 스크립트의 입력·출력 파일 이름은 코드에 직접 적혀 있으니, 필요하면 수정한 뒤 실행하세요.
