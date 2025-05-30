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
model = load_model("featured_data_hand_pose_classifier_20_64_nadam_with_han.h5")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")
print("클래스 순서:", le.classes_)

# 2. MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 3. 특징 추출용 함수

# 오른손/왼손

# 손 사이즈 구하기
def get_hand_size(coords):
    coords = np.array(coords)
    x_min, y_min = np.min(coords[:, 0]), np.min(coords[:, 1])
    x_max, y_max = np.max(coords[:, 0]), np.max(coords[:, 1])

    return np.linalg.norm([x_max - x_min, y_max - y_min])

# 손가락 마디별 각도 계산
def angle_finger_joint(wrist, p1, p2, p3, p4):
    angle1 = angle_between(wrist, p1, p2)
    angle2 = angle_between(p1, p2, p3)
    angle3 = angle_between(p2, p3, p4)

    return [angle1, angle2, angle3]

# 마디간 각도를 0 ~ 360까지의 거리를 라디안으로 계산
def angle_between(p1, p2, p3):
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)

    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)

    dot = np.dot(a_norm, b_norm)
    cross = a_norm[0] * b_norm[1] - a_norm[1] * b_norm[0]  # 2D 외적: z성분

    angle = np.arctan2(cross, dot)
    if angle < 0:
        angle += 2 * np.pi

    return angle

# 손 사이즈 대비 손가락 관절 간 거리 계산
def euclidean_distance(p1, p2, hand_size):
    distance = np.linalg.norm(np.array(p1) - np.array(p2))
    result_distance = distance / hand_size
    if hand_size == 0:
        result_distance = 0.0
    return result_distance

def hand_orientation_angle(coords):
    wrist = np.array(coords[0])
    middle_mcp = np.array(coords[9])
    vec = middle_mcp - wrist
    angle = np.arctan2(vec[1], vec[0])  # 라디안
    return angle  # 그대로 추가해도 되고 np.degrees(angle)로 바꿔도 됨

def extract_feature_from_coords(row):
    features = []

    # 손가락 굽힘 각도 (손목 + 손끝까지 향하는 각 관절 4개)
    finger_joints = [
        (0, 1, 2, 3, 4),  # 엄지
        (0, 5, 6, 7, 8),  # 검지
        (0, 9, 10, 11, 12),  # 중지
        (0, 13, 14, 15, 16),  # 약지
        (0, 17, 18, 19, 20)  # 소지
    ]
    for wrist, p1, p2, p3, p4 in finger_joints:
        angles = angle_finger_joint(coords[wrist], coords[p1], coords[p2], coords[p3], coords[p4])
        features.extend(angles)

    '''
    # 손가락 간 각도
    angle_pairs = [(4,8), (8,12), (12,16), (16,20)]
    for i, j in angle_pairs:
        features.append(angle_between(coords[0], coords[i], coords[j]))
    '''

    # 손가락 간 거리
    distances = []
    hand_size = get_hand_size(coords)  # 손 사이즈 구하기
    tip_indices = [4, 8, 12, 16, 20]
    for i in range(len(tip_indices)-1):
        d = euclidean_distance(coords[tip_indices[i]], coords[tip_indices[i+1]], hand_size)
        distances.append(d)
        features.append(d)

    # 거리 비율
    if distances[1] != 0:
        features.append(distances[0] / distances[1])
    else:
        features.append(0.0)

    # 손 회전 각도
    features.append(hand_orientation_angle(coords))

    return np.array(features)

# 한글 쓰기용 함수
def draw_text_with_pil(img, text, position, color=(0,255,0)):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

# 4. IP Webcam 연결 (IP Webcam 주소 입력)
ip_webcam_url = "http://192.168.219.134:8080/video"
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

                if len(features) == 21:
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
