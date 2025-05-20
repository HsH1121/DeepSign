import cv2
import numpy as np
import mediapipe as mp
import joblib
from tensorflow.keras.models import load_model

# 1. 모델 및 유틸 로드
model = load_model("residual_mlp_model.h5")
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
    """중지 기준 손 회전 각도 계산 (라디안)"""
    dx = coords[5][0] - coords[17][0]
    dy = coords[5][1] - coords[17][1]
    return np.arctan2(dy, dx)

def extract_feature_from_coords(coords):
    features = []

    # 1. 손가락 굽힘 각도
    finger_joints = [(5,6,7), (9,10,11), (13,14,15), (17,18,19), (2,3,4)]
    for a, b, c in finger_joints:
        features.append(angle_between(coords[a], coords[b], coords[c]))

    # 2. 손가락 간 각도
    angle_pairs = [(4,8), (8,12), (12,16), (16,20)]
    for i, j in angle_pairs:
        features.append(angle_between(coords[0], coords[i], coords[j]))

    # 3. 손가락 끝 간 거리
    tip_indices = [4, 8, 12, 16, 20]
    distances = []
    for i in range(len(tip_indices)-1):
        d = euclidean_distance(coords[tip_indices[i]], coords[tip_indices[i+1]])
        distances.append(d)
        features.append(d)

    # 4. 거리 비율
    if distances[1] != 0:
        features.append(distances[0] / distances[1])
    else:
        features.append(0.0)

    # 5. 손 회전 각도
    features.append(hand_rotation_angle(coords))

    return np.array(features)

# 4. WebCam 실행
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
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

                    cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Residual MLP - Real-time Sign Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
