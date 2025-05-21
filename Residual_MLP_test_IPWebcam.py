import cv2
import numpy as np
import mediapipe as mp
import joblib
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image # 한글 인식을 위한 라이브러리

# 한글 폰트 파일 경로 지정
font_path = "NanumGothic.ttf"
font = ImageFont.truetype(font_path, 40)

# 1. 모델 및 유틸 로드
model = load_model("residual_mlp_model_with_han.h5")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# 2. MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 3. 특징 추출 함수
def angle_between(p1, p2, p3):
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def hand_rotation_angle(coords):
    dx = coords[5][0] - coords[17][0]
    dy = coords[5][1] - coords[17][1]
    return np.arctan2(dy, dx)

def extract_feature_from_coords(coords):
    features = []

    finger_joints = [(5,6,7), (9,10,11), (13,14,15), (17,18,19), (2,3,4)]
    for a, b, c in finger_joints:
        features.append(angle_between(coords[a], coords[b], coords[c]))

    angle_pairs = [(4,8), (8,12), (12,16), (16,20)]
    for i, j in angle_pairs:
        features.append(angle_between(coords[0], coords[i], coords[j]))

    tip_indices = [4, 8, 12, 16, 20]
    distances = []
    for i in range(len(tip_indices)-1):
        d = euclidean_distance(coords[tip_indices[i]], coords[tip_indices[i+1]])
        distances.append(d)
        features.append(d)

    if distances[1] != 0:
        features.append(distances[0] / distances[1])
    else:
        features.append(0.0)

    features.append(hand_rotation_angle(coords))

    return np.array(features)

# 한글 쓰기용 함수
def draw_text_with_pil(img, text, position, color=(0,255,0)):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

# 4. IP Webcam 연결 (IP Webcam 주소 입력)
ip_webcam_url = "http://192.168.219.137:8080/video"

cap = cv2.VideoCapture(ip_webcam_url)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[!] 프레임을 불러올 수 없습니다.")
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            coords = [[lm.x, lm.y] for lm in hand_landmarks.landmark]

            if len(coords) == 21:
                features = extract_feature_from_coords(coords)

                if len(features) == 15:
                    features_scaled = scaler.transform([features])
                    pred = model.predict(features_scaled)
                    label = le.inverse_transform([np.argmax(pred)])[0]
                    confidence = np.max(pred)

                    text = f"{label} ({confidence:.2f})"
                    frame = draw_text_with_pil(frame, text, (10, 30), color=(0, 255, 0))

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("IP Webcam - Residual MLP Sign Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
