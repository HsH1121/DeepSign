import cv2
import numpy as np
import mediapipe as mp
import os
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image

model = load_model('hand_pose_classifier_20_64_nadam_with_han.h5')

# 레이블 데이터 리스트 생성
label_path = "D:/University/4-1/DeepLearning_Programming/DLP_Project/DLP_Project/dataset_raw"
labels = [
    name for name in os.listdir(label_path)
    if os.path.isdir(os.path.join(label_path, name))
]

# 한글 폰트 파일 경로 지정
font_path = "NanumGothic.ttf"
font = ImageFont.truetype(font_path, 40)

# 한글 쓰기용 함수
def draw_text_with_pil(img, text, position, color=(0,255,0)):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)

# 라벨 (예측 숫자 복원용)
# labels = [str(i) for i in range(1, 42)]  # 1~41

# 웹캠 열기
ip_webcam_url = "http://192.168.219.196:8080/video"
cap = cv2.VideoCapture(ip_webcam_url)

print("[i] 웹캠 시작. 손을 카메라 앞에 가져다 대세요.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # 1. 좌표 추출
            coords = []
            for lm in hand_landmarks.landmark:
                coords.append(lm.x)
                coords.append(lm.y)

            coords = np.array(coords).reshape(1, -1)

            # 2. 예측
            pred = model.predict(coords)
            pred_idx = np.argmax(pred)
            pred_label = labels[pred_idx]
            confidence = np.max(pred)

            # 3. 시각화
            # bounding box 표시
            x_list = [lm.x for lm in hand_landmarks.landmark]
            y_list = [lm.y for lm in hand_landmarks.landmark]
            xmin = int(min(x_list) * w) - 20
            ymin = int(min(y_list) * h) - 20
            xmax = int(max(x_list) * w) + 20
            ymax = int(max(y_list) * h) + 20
            xmin, ymin = max(xmin, 0), max(ymin, 0)
            xmax, ymax = min(xmax, w), min(ymax, h)

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            text = f"{pred_label} ({confidence:.2f})"
            frame = draw_text_with_pil(frame, text, (10, 30), color=(0, 255, 0))

    # 결과 출력
    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) == 27:  # ESC 종료
        break

cap.release()
cv2.destroyAllWindows()
