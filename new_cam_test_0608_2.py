import cv2
import numpy as np
import mediapipe as mp
import joblib
import time
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image
from collections import Counter, deque
import hgtk
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

# LSTM 자모 예측 모델 로드
lstm_model = keras_load_model("SiotYu_distinction_2.h5")

# 자모 인덱싱 준비
jaum_list = list("ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ")
moum_list = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ")
all_jamo = sorted(set(jaum_list + moum_list + jaum_list))
char_to_index = {char: idx for idx, char in enumerate(all_jamo)}
index_to_char = {idx: char for char, idx in char_to_index.items()}
max_len = 20

def try_compose_and_add():
    global final_inputed_labels, label_check_compose_list, inputed_moum

    if (
        len(label_check_compose_list) >= 2 and
        is_full_choseong(label_check_compose_list[0]) and
        is_full_jungseong(label_check_compose_list[1]) and
        (len(label_check_compose_list) == 2 or (
            len(label_check_compose_list) == 3 and is_full_jongseong(label_check_compose_list[2]))
        )
    ):
        try:
            composed_char = hgtk.letter.compose(*label_check_compose_list)
            final_inputed_labels += composed_char
            label_check_compose_list.clear()
            inputed_moum = False
        except Exception as e:
            print(f"⚠️ 조합 실패: {label_check_compose_list} -> {e}")
    else:
        print(f"⛔️ 조합 조건 불충족: {label_check_compose_list}")


def predict_with_interval(features, scaler, model, le, interval=0.1, cache={}):
    """
    features: numpy array of input features
    scaler: the fitted scaler object
    model: the loaded Keras model
    le: the label encoder
    interval: time in seconds between predictions
    cache: dictionary to store last prediction results
    """
    current_time = time.time()

    # Initialize cache if empty
    if not cache:
        cache["last_prediction_time"] = 0
        cache["cached_label"] = ""
        cache["cached_confidence"] = 0.0

    if current_time - cache["last_prediction_time"] > interval:
        features_scaled = scaler.transform([features])
        pred = model.predict(features_scaled)
        cache["cached_label"] = le.inverse_transform([np.argmax(pred)])[0]
        cache["cached_confidence"] = float(np.max(pred))
        cache["last_prediction_time"] = current_time

    return cache["cached_label"], cache["cached_confidence"]


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

# ====== 자음 판별 함수 ======
def is_jaum(char):
    jaum = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    moum = ['ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
    if char in jaum:
        return True
    elif char in moum:
        return False
    else:
        return None

def is_full_choseong(char):
    full_choseong = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ',
                     'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    if char in full_choseong:
        return True
    else:
        return False

def is_full_jungseong(char):
    full_jungseong = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ',
                      'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ',
                      'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
    if char in full_jungseong:
        return True
    else:
        return False

def is_full_jongseong(char):
    full_jongseong = ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ',
                      'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ',
                      'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ',
                      'ㅌ', 'ㅍ', 'ㅎ']
    if char in full_jongseong:
        return True
    else:
        return False

# ====== 초성 된소리 획득 함수 ======
def get_double_choseong(char1, char2):
    double_choseong_map = {
        ('ㄱ', 'ㄱ'): 'ㄲ',
        ('ㄷ', 'ㄷ'): 'ㄸ',
        ('ㅂ', 'ㅂ'): 'ㅃ',
        ('ㅅ', 'ㅅ'): 'ㅆ',
        ('ㅈ', 'ㅈ'): 'ㅉ'
    }

    return double_choseong_map.get((char1, char2), None)

# ====== 중성 겹모음 획득 함수 ======
def get_double_jungseong(char1, char2):
    double_jungseong_map = {
        ('ㅗ', 'ㅏ'): 'ㅘ',
        ('ㅗ', 'ㅐ'): 'ㅙ',
        ('ㅗ', 'ㅣ'): 'ㅚ',
        ('ㅜ', 'ㅓ'): 'ㅝ',
        ('ㅜ', 'ㅔ'): 'ㅞ',
        ('ㅜ', 'ㅣ'): 'ㅟ',
        ('ㅡ', 'ㅣ'): 'ㅢ',
    }

    return double_jungseong_map.get((char1, char2), None)

# ====== 종성 겹받침 획득 함수
def get_double_jongseong(char1, char2):
    double_jongseong_map = {
        ('ㄱ', 'ㄱ'): 'ㄲ',
        ('ㅅ', 'ㅅ'): 'ㅆ',
        ('ㄱ', 'ㅅ'): 'ㄳ',
        ('ㄴ', 'ㅈ'): 'ㄵ',
        ('ㄴ', 'ㅎ'): 'ㄶ',
        ('ㄹ', 'ㄱ'): 'ㄺ',
        ('ㄹ', 'ㅁ'): 'ㄻ',
        ('ㄹ', 'ㅂ'): 'ㄼ',
        ('ㄹ', 'ㅅ'): 'ㄽ',
        ('ㄹ', 'ㅌ'): 'ㄾ',
        ('ㄹ', 'ㅍ'): 'ㄿ',
        ('ㄹ', 'ㅎ'): 'ㅀ',
        ('ㅂ', 'ㅅ'): 'ㅄ',
    }

    return double_jongseong_map.get((char1, char2), None)

def compose_hangul_move_jonseong_to_choseong(label_check_compose_list):
    global final_inputed_labels, inputed_moum
    choseong = label_check_compose_list.pop()
    if (
            len(label_check_compose_list) == 2 and
            is_full_choseong(label_check_compose_list[0]) and
            is_full_jungseong(label_check_compose_list[1])
    ):
        final_inputed_labels = final_inputed_labels + hgtk.letter.compose(*label_check_compose_list)
    else:
        final_inputed_labels = final_inputed_labels + "".join(label_check_compose_list)
    label_check_compose_list.clear()
    label_check_compose_list.append(choseong)
    inputed_moum = False

def compose_hangul(label_check_compose_list):
    global final_inputed_labels, inputed_moum

    if (
            len(label_check_compose_list) >= 2 and
            is_full_choseong(label_check_compose_list[0]) and
            is_full_jungseong(label_check_compose_list[1]) and
            (len(label_check_compose_list) == 2 or (len(label_check_compose_list) == 3 and is_full_jongseong(label_check_compose_list[2])))
    ):
        final_inputed_labels = final_inputed_labels + hgtk.letter.compose(*label_check_compose_list)
    else:
        final_inputed_labels = final_inputed_labels + "".join(label_check_compose_list)
    label_check_compose_list.clear()
    inputed_moum = False

#def add_label(label):
 #   global final_inputed_labels, inputed_moum
  #  inputed_moum = False
   # final_inputed_labels = final_inputed_labels + label

def add_label(label):
    global final_inputed_labels, inputed_moum
    inputed_moum = False
    final_inputed_labels = final_inputed_labels + str(label)  # <== 여기서 str()로 변환


# ====== 초기 모델: 한글 ======
current_model_key = "hangul"
model, scaler, le = load_model_set(current_model_key)
print("✅ 초기 모델: 한글 (1번 키로 다시 변경 가능)")

# ====== MediaPipe 초기화 ======
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# ====== 자모 조합, 출력용 ======
final_inputed_labels = "" # 최종 입력 문자열
label_history = deque(maxlen = 20) # 최근 20 Frame 예측
last_added_time = 0
label_check_compose_list = []
inputed_moum = False
last_input = ""

# ====== 실시간 웹캠 ======
cap = cv2.VideoCapture(0)

predict_cache = {}

while cap.isOpened():
    for _ in range(5):
        cap.grab()
    ret, frame = cap.retrieve()
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
                    label, confidence = predict_with_interval(features, scaler, model, le, 0.1, predict_cache)
                    current_time = time.time()
                    label_history.append((label, confidence, current_time))

                    recent = [(l, c) for l, c, t in label_history if current_time - t <= 1.0]
                    if recent:
                        labels = [l for l, _ in recent]
                        pred_label, count = Counter(labels).most_common(1)[0]
                        portion = count / len(recent)

                        if portion >= 0.8:
                            latest_conf = [c for l, c in recent if l == pred_label][-1]
                            if latest_conf >= 0.8 and current_time - last_added_time > 1.0:

                                # 🔍 'ㅅ'/'ㅠ' 보정 로직 적용
                                if pred_label in ['ㅅ', 'ㅠ']:
                                    context_jamo = list(final_inputed_labels.replace(" ", ""))
                                    if label_check_compose_list:
                                        context_jamo += label_check_compose_list
                                    input_idx = [char_to_index[j] for j in context_jamo if j in char_to_index]
                                    if input_idx:
                                        input_pad = pad_sequences([input_idx], maxlen=max_len, padding='pre')
                                        lstm_pred = lstm_model.predict(input_pad, verbose=0)
                                        lstm_pred_idx = np.argmax(lstm_pred)
                                        lstm_pred_jamo = index_to_char[lstm_pred_idx]
                                        print("🧪 LSTM 입력:", input_idx)
                                        print("🧪 LSTM 예측:", lstm_pred)
                                        print("🧪 예측 자모:", lstm_pred_jamo)
                                        if lstm_pred_jamo in ['ㅅ', 'ㅠ']:
                                            pred_label = lstm_pred_jamo
                                        else:
                                            pred_label = 'ㅅ' if is_jaum(lstm_pred_jamo) else 'ㅠ'
                                            print(f"👉 보정된 라벨: {pred_label}")

                                if pred_label == "conversion_model_1":
                                    if current_model_key == "hangul":
                                        model, scaler, le = load_model_set("digit")
                                        current_model_key = "digit"
                                        print("🔁 [자동전환] 숫자 모델로 전환됨.")
                                    elif current_model_key == "digit":
                                        model, scaler, le = load_model_set("hangul")
                                        current_model_key = "hangul"
                                        print("🔁 [자동전환] 한글 모델로 전환됨.")
                                    label_history.clear()
                                    last_added_time = time.time()
                                    continue

                                if is_jaum(pred_label):
                                    if not label_check_compose_list:
                                        # 1. 리스트가 비어있으면 초성 시작
                                        label_check_compose_list.append(pred_label)
                                        label_history.clear()
                                        last_added_time = time.time()

                                    elif not inputed_moum:
                                        # 2. 아직 모음이 입력되지 않은 경우
                                        if get_double_choseong(label_check_compose_list[-1], pred_label) is not None:
                                            # 2-1. 겹초성 형성 가능
                                            label_check_compose_list[-1] = get_double_choseong(
                                                label_check_compose_list[-1], pred_label)
                                        else:
                                            # 2-2. 겹초성 불가 → 조합 마무리 후 새 글자 시작
                                            add_label(label_check_compose_list[-1])
                                            label_check_compose_list.clear()
                                            label_check_compose_list.append(pred_label)
                                        label_history.clear()
                                        last_added_time = time.time()

                                    elif inputed_moum:
                                        # 3. 모음 입력 이후 → 종성 후보
                                        if len(label_check_compose_list) == 2:
                                            # 3-1. 초성+중성 상태일 때
                                            if is_full_jongseong(pred_label):
                                                label_check_compose_list.append(pred_label)  # 종성 추가
                                            else:
                                                # 종성 불가 → 조합 마치고 새 글자 시작
                                                compose_hangul(label_check_compose_list)
                                                label_check_compose_list.clear()
                                                label_check_compose_list.append(pred_label)
                                                inputed_moum = False
                                        elif len(label_check_compose_list) == 3:
                                            # 3-2. 이미 종성이 있을 때 → 겹받침 가능성 체크
                                            last_jong = label_check_compose_list[-1]
                                            double_jong = get_double_jongseong(last_jong, pred_label)
                                            if double_jong:
                                                label_check_compose_list[-1] = double_jong
                                            else:
                                                compose_hangul(label_check_compose_list)
                                                label_check_compose_list.clear()
                                                label_check_compose_list.append(pred_label)
                                                inputed_moum = False
                                        else:
                                            # 예외 처리: 비정상 길이 → 리셋
                                            compose_hangul(label_check_compose_list)
                                            label_check_compose_list.clear()
                                            label_check_compose_list.append(pred_label)
                                            inputed_moum = False

                                        label_history.clear()
                                        last_added_time = time.time()


                                    elif inputed_moum:
                                        # 현재 받은 자음이 종성 후보
                                        last_char = label_check_compose_list[-1]

                                        # ① 현재 종성이 비어있고 자음 추가 → 종성 가능 여부 판단
                                        if is_jaum(last_char):
                                            double_jong = get_double_jongseong(last_char, pred_label)
                                            if double_jong is not None:
                                                # ② 겹종성 형성 가능 → 붙이기
                                                label_check_compose_list[-1] = double_jong
                                            else:
                                                # ③ 겹종성 불가인데 종성 이미 존재 → 조합 완료 후 새 글자 시작
                                                compose_hangul(label_check_compose_list)
                                                label_check_compose_list.clear()
                                                label_check_compose_list.append(pred_label)
                                                inputed_moum = False
                                        else:
                                            # ④ 종성이 자음이 아닌 경우 (예외)
                                            compose_hangul(label_check_compose_list)
                                            label_check_compose_list.clear()
                                            label_check_compose_list.append(pred_label)
                                            inputed_moum = False


                                elif not is_jaum(pred_label):
                                    if not label_check_compose_list:
                                        add_label(pred_label)
                                        label_history.clear()
                                        last_added_time = time.time()
                                    elif label_check_compose_list:
                                        if not inputed_moum:
                                            label_check_compose_list.append(pred_label)
                                            inputed_moum = True
                                            label_history.clear()
                                            last_added_time = time.time()
                                        elif inputed_moum:
                                            if is_jaum(label_check_compose_list[-1]):
                                                compose_hangul_move_jonseong_to_choseong(label_check_compose_list)
                                                label_check_compose_list.append(pred_label)
                                                inputed_moum = True
                                                label_history.clear()
                                                last_added_time = time.time()
                                            elif not is_jaum(label_check_compose_list[-1]):
                                                if get_double_jungseong(label_check_compose_list[-1], pred_label) is not None:
                                                    label_check_compose_list[-1] = get_double_jungseong(label_check_compose_list[-1], pred_label)
                                                    label_history.clear()
                                                    last_added_time = time.time()
                                                else:
                                                    compose_hangul(label_check_compose_list)
                                                    add_label(pred_label)
                                                    label_history.clear()
                                                    last_added_time = time.time()

                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    frame = draw_text_with_pil(frame, f"{label} ({confidence:.2f})", (10, 30))

    if (
        len(label_check_compose_list) >= 2 and
        is_full_choseong(label_check_compose_list[0]) and
        is_full_jungseong(label_check_compose_list[1]) and
        (len(label_check_compose_list) == 2 or (len(label_check_compose_list) == 3 and is_full_jongseong(label_check_compose_list[2])))
    ):
        frame = draw_text_with_pil(frame, final_inputed_labels + hgtk.letter.compose(*label_check_compose_list), (10, 80), color=(255, 255, 0))
    else:
        frame = draw_text_with_pil(frame, final_inputed_labels + "".join(map(str, label_check_compose_list)), (10, 80), color=(255, 255, 0))

    cv2.imshow("🖐 실시간 손 목사 인식 (1: 한글 / 2: 숫자 / ESC: 종료)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('1') and current_model_key != "hangul":
        model, scaler, le = load_model_set("hangul")
        current_model_key = "hangul"
        print("🔁 한글 모델로 전환됨.")
    elif key == ord('2') and current_model_key != "digit":
        model, scaler, le = load_model_set("digit")
        current_model_key = "digit"
        print("🔁 숫자 모델로 전환됨.")

cap.release()
cv2.destroyAllWindows()
