# 실시간 한글 지문자(수어) 인식

MediaPipe로 손 관절 좌표를 추출하고, MLP 모델로 한글 자음·모음과 숫자 지문자를 실시간으로 인식해 글자로 조합하는 프로젝트입니다.
모양이 비슷한 `ㅅ`/`ㅠ`는 앞뒤 문맥을 보는 LSTM 모델로 한 번 더 구분합니다.

## 동작 방식

1. 스마트폰 **IP Webcam** 앱으로 받은 영상에서 MediaPipe Hands가 손 관절 21개의 좌표를 찾습니다.
2. 좌표로 특징 21개를 계산합니다. 손가락 마디 각도, 손가락 끝 사이 거리(손 크기로 정규화), 거리 비율, 손 회전 각도입니다.
3. MLP 모델이 자모나 숫자를 분류합니다.
4. 최근 1초 동안 같은 결과가 80% 이상이고 신뢰도도 0.8 이상이면 입력으로 받아들입니다.
5. 입력된 자모를 `hgtk`로 조합해 완성형 한글로 만듭니다. 쌍자음, 겹모음, 겹받침도 처리합니다.
6. `space`, `back_space`, `conversion_model_1`(한글↔숫자 전환) 동작은 특수 입력으로 처리합니다.

## 설치

```bash
pip install -r requirements.txt
```

> MediaPipe의 `mp.solutions.hands` API를 사용합니다. 이 API를 지원하는 MediaPipe 버전을 설치해야 합니다.

## 실행

### 실시간 인식

```bash
python IPWebcam_test.py
```

- 실행하기 전에 스크립트 안의 IP Webcam 주소(`http://<폰 IP>:8080/video`)를 자신의 환경에 맞게 바꿔 주세요.
- 키 조작: `1` 한글 모델, `2` 숫자 모델, `ESC` 종료

## 파일 구성

### 스크립트

| 파일 | 설명 |
| --- | --- |
| `collect_datasets_IPWebcam.py` | 라벨을 입력하고 스페이스바를 누르면 잘라낸 손 이미지를 `dataset_raw/<라벨>/`에 저장 |
| `get_hand_coords_csv.py` | `dataset_raw`의 이미지에서 손 좌표를 뽑아 `digits.csv`와 `hangul.csv` 생성 |
| `get_hand_coords_onlyNum.py` | 숫자와 특수 동작 라벨만 뽑아 `digits.csv` 생성 |
| `get_featured_data_csv.py` | 좌표 CSV를 특징 CSV로 변환 (예: `digits.csv` → `digit_feature.csv`) |
| `separate_csv.py` | 특징 CSV를 한글용과 숫자용으로 나눔 |
| `sign_language_distinction_MLP_model_train.py` | MLP 분류 모델 학습 후 `model_*.h5`, `scaler_*.pkl`, `label_encoder_*.pkl` 저장 |
| `SiotYu_distinction_LSTM_model_train.py` | `ㅅ`/`ㅠ` 구분용 자모 시퀀스 LSTM 모델 학습 |
| `SiotYu_distinction_LSTM_model_test.py` | LSTM 모델의 자모 시퀀스 예측 테스트 |
| `IPWebcam_test.py` | 실시간 인식과 한글 조합 (메인) |
| `original_separate_model_test.py` | 이전 버전: 자음, 모음, 숫자 모델을 따로 쓰는 로컬 웹캠 테스트 |

### 학습 결과물 / 데이터

| 파일 | 설명 |
| --- | --- |
| `model_hangul.h5`, `model_digit.h5` | 한글과 숫자 MLP 모델 |
| `scaler_*.pkl`, `label_encoder_*.pkl` | 각 모델의 MinMaxScaler와 LabelEncoder |
| `SiotYu_distinction*.h5` | `ㅅ`/`ㅠ` 구분용 LSTM 모델 |
| `non_conversion/`, `with_no_space/`, `미리 받은 파일들/` | 이전 실험 버전의 모델 |
| `NanumGothic.ttf` | 화면에 한글을 표시하기 위한 나눔고딕 폰트 (SIL Open Font License) |

## 데이터셋

손 사진과 CSV는 용량이 커서 저장소에 넣지 않았습니다. 학습된 모델(`.h5`, `.pkl`)은 저장소에 있으므로 **실시간 인식만 할 때는 데이터가 필요 없습니다.**

### 내려받기

[Releases의 `dataset-v1`](https://github.com/HsH1121/DeepSign/releases/tag/dataset-v1)에서 받아 프로젝트 루트에 압축을 풉니다.

| 파일 | 내용 | 압축을 풀면 |
| --- | --- | --- |
| [`dataset_raw.zip`](https://github.com/HsH1121/DeepSign/releases/download/dataset-v1/dataset_raw.zip) | 라벨별 손 이미지 (약 1.4GB) | `dataset_raw/<라벨>/*.jpg` |
| [`csv_data.zip`](https://github.com/HsH1121/DeepSign/releases/download/dataset-v1/csv_data.zip) | 손 좌표와 특징 CSV | 루트의 `*.csv`, `non_conversion/*.csv` |

### 이전 버전

커밋 기록은 개발 과정의 폴더(`hangul_test` ~ `hangul_test_8`)를 버전 순서대로 담고 있습니다. 각 버전 커밋에는 같은 이름의 태그가 있고, 그 태그의 Release에 해당 버전에서 쓴 CSV(`csv_data.zip`)가 첨부되어 있습니다.

| 버전 | 코드 | CSV |
| --- | --- | --- |
| hangul_test | [태그](https://github.com/HsH1121/DeepSign/tree/hangul_test) | [Release](https://github.com/HsH1121/DeepSign/releases/tag/hangul_test) |
| hangul_test_2 | [태그](https://github.com/HsH1121/DeepSign/tree/hangul_test_2) | [Release](https://github.com/HsH1121/DeepSign/releases/tag/hangul_test_2) |
| hangul_test_3 | [태그](https://github.com/HsH1121/DeepSign/tree/hangul_test_3) | [Release](https://github.com/HsH1121/DeepSign/releases/tag/hangul_test_3) |
| hangul_test_4 | [태그](https://github.com/HsH1121/DeepSign/tree/hangul_test_4) | [Release](https://github.com/HsH1121/DeepSign/releases/tag/hangul_test_4) |
| hangul_test_5 | [태그](https://github.com/HsH1121/DeepSign/tree/hangul_test_5) | [Release](https://github.com/HsH1121/DeepSign/releases/tag/hangul_test_5) |
| hangul_test_6 | [태그](https://github.com/HsH1121/DeepSign/tree/hangul_test_6) | [Release](https://github.com/HsH1121/DeepSign/releases/tag/hangul_test_6) |
| hangul_test_7 | [태그](https://github.com/HsH1121/DeepSign/tree/hangul_test_7) | [Release](https://github.com/HsH1121/DeepSign/releases/tag/hangul_test_7) |
| hangul_test_8 | [태그](https://github.com/HsH1121/DeepSign/tree/hangul_test_8) | [Release](https://github.com/HsH1121/DeepSign/releases/tag/hangul_test_8) |

> 손 이미지는 버전 간 차이가 작아 최신본(`dataset-v1`의 `dataset_raw.zip`)만 제공합니다.

### 직접 만들기

사진부터 새로 모으거나, 받은 사진으로 CSV와 모델을 다시 만들려면 다음 순서로 실행합니다.

```text
collect_datasets_IPWebcam.py   →  dataset_raw/<라벨>/*.jpg
get_hand_coords_csv.py         →  digits.csv, hangul.csv
get_featured_data_csv.py       →  *_feature.csv
sign_language_distinction_MLP_model_train.py  →  model_*.h5, scaler_*.pkl, label_encoder_*.pkl
```

> 각 스크립트의 입력·출력 파일 이름은 코드에 직접 적혀 있으니, 필요하면 수정한 뒤 실행하세요.
