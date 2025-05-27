import cv2
import numpy as np
import mediapipe as mp
import joblib
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image

# ====== 폰트 설정 ======
font_path = "NanumGothic.ttf"
font = ImageFont.truetype(font_path, 40)

# ====== 모델 로드 함수 ======
def load_model_set(key):
    model_path = f"model_{key}.h5"
    scaler_path = f"scaler_{key}.pkl"
    encoder_path = f"label_encoder_{key}.pkl"
    return (
        load_model(model_path),
        joblib.load(scaler_path),
        joblib.load(encoder_path)
    )

# ====== 특징 추출 함수 ======
def get_hand_size(coords):
    coords = np.array(coords)
    x_min, y_min = np.min(coords[:, 0]), np.min(coords[:, 1])
    x_max, y_max = np.max(coords[:, 0]), np.max(coords[:, 1])
    return np.linalg.norm([x_max - x_min, y_max - y_min])

def angle_between(p1, p2, p3):
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    dot = np.dot(a_norm, b_norm)
    cross = a_norm[0] * b_norm[1] - a_norm[1] * b_norm[0]
    angle = np.arctan2(cross, dot)
    if angle < 0:
        angle += 2 * np.pi
    return angle

def angle_finger_joint(wrist, p1, p2, p3, p4):
    return [
        angle_between(wrist, p1, p2),
        angle_between(p1, p2, p3),
        angle_between(p2, p3, p4)
    ]

def euclidean_distance(p1, p2, hand_size):
    dist = np.linalg.norm(np.array(p1) - np.array(p2))
    return dist / hand_size if hand_size != 0 else 0.0

def hand_orientation_angle(coords):
    wrist = np.array(coords[0])
    middle_mcp = np.array(coords[9])
    vec = middle_mcp - wrist
    return np.arctan2(vec[1], vec[0])

def extract_feature_from_coords(coords):
    features = []
    finger_joints = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                     (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]
    for wrist, p1, p2, p3, p4 in finger_joints:
        features.extend(angle_finger_joint(coords[wrist], coords[p1], coords[p2], coords[p3], coords[p4]))
    distances = []
    hand_size = get_hand_size(coords)
    tips = [4, 8, 12, 16, 20]
    for i in range(len(tips) - 1):
        d = euclidean_distance(coords[tips[i]], coords[tips[i+1]], hand_size)
        distances.append(d)
        features.append(d)
    features.append(distances[0] / distances[1] if distances[1] != 0 else 0.0)
    features.append(hand_orientation_angle(coords))
    return np.array(features)

# ====== 한글 텍스트 출력 함수 ======
def draw_text_with_pil(img, text, position, color=(0,255,0)):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

# ====== 초기 모델: 자음 ======
current_model_key = "consonant"
model, scaler, le = load_model_set(current_model_key)
print("✅ 초기 모델: 자음 (1번 키로 다시 변경 가능)")

# ====== MediaPipe 초기화 ======
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# ====== 실시간 웹캠 ======
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임을 불러올 수 없습니다.")
        break

    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            coords = [[lm.x, lm.y] for lm in hand_landmarks.landmark]
            if len(coords) == 21:
                features = extract_feature_from_coords(coords)
                if len(features) >= 15:
                    features_scaled = scaler.transform([features])
                    pred = model.predict(features_scaled)
                    label = le.inverse_transform([np.argmax(pred)])[0]
                    confidence = np.max(pred)
                    text = f"{label} ({confidence:.2f})"
                    frame = draw_text_with_pil(frame, text, (10, 30))
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("🖐 실시간 손 모양 인식 (1: 자음 / 2: 모음 / 3: 숫자 / ESC: 종료)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('1') and current_model_key != "consonant":
        model, scaler, le = load_model_set("consonant")
        current_model_key = "consonant"
        print("🔁 자음 모델로 전환됨.")
    elif key == ord('2') and current_model_key != "vowel":
        model, scaler, le = load_model_set("vowel")
        current_model_key = "vowel"
        print("🔁 모음 모델로 전환됨.")
    elif key == ord('3') and current_model_key != "digit":
        model, scaler, le = load_model_set("digit")
        current_model_key = "digit"
        print("🔁 숫자 모델로 전환됨.")

cap.release()
cv2.destroyAllWindows()
