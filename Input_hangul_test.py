import cv2
import numpy as np
import mediapipe as mp
import joblib
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import ImageFont, ImageDraw, Image
from collections import Counter, deque
import hgtk

# ====== 폰트 설정 ======
font_path = "NanumGothic.ttf"
font = ImageFont.truetype(font_path, 40)

# LSTM 모델용 자모 리스트와 인덱스 매핑
jaum_list = list("ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ")
moum_list = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ")
all_jamo = sorted(set(jaum_list + moum_list + jaum_list))
char_to_index = {char: idx for idx, char in enumerate(all_jamo)}
index_to_char = {idx: char for char, idx in char_to_index.items()}

VOCAB_SIZE = len(char_to_index)
MAX_LEN = 20  # 학습 시 사용한 max_len과 동일하게 설정해야 함

SiotYu_distinction_model = load_model("SiotYu_distinction.h5")

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
        d = euclidean_distance(coords[tips[i]], coords[tips[i + 1]], hand_size)
        distances.append(d)
        features.append(d)
    features.append(distances[0] / distances[1] if distances[1] != 0 else 0.0)
    features.append(hand_orientation_angle(coords))
    return np.array(features)

# ====== 한글 텍스트 출력 함수 ======
def draw_text_with_pil(img, text, position, color=(0, 255, 0)):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

# ====== 자/모음 판별 함수 ======
def is_jaum(char):
    jaum = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    moum = ['ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ', 'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅢ']
    if char in jaum:
        return True
    elif char in moum:
        return False
    else:
        return None

# ====== 초/중/종성 판별 함수 ======
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

# ====== 겹자/모음 조합 함수 ======
def get_double_choseong(char1, char2):
    double_choseong_map = {
        ('ㄱ', 'ㄱ'): 'ㄲ',
        ('ㄷ', 'ㄷ'): 'ㄸ',
        ('ㅂ', 'ㅂ'): 'ㅃ',
        ('ㅅ', 'ㅅ'): 'ㅆ',
        ('ㅈ', 'ㅈ'): 'ㅉ'
    }

    return double_choseong_map.get((char1, char2), None)

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

# ====== 겹자/모음 해체 함수 ======
def decompose_double_moum(char):
    double_jungseong_map = {
        'ㅘ': ['ㅗ', 'ㅏ'],
        'ㅙ': ['ㅗ', 'ㅐ'],
        'ㅚ': ['ㅗ', 'ㅣ'],
        'ㅝ': ['ㅜ', 'ㅓ'],
        'ㅞ': ['ㅜ', 'ㅔ'],
        'ㅟ': ['ㅜ', 'ㅣ'],
        'ㅢ': ['ㅡ', 'ㅣ'],
    }

    return double_jungseong_map.get(char, [char])

def decompose_double_jaum(char):
    double_jaum_map = {
        # 초성 전용 쌍자음
        'ㄲ': ['ㄱ', 'ㄱ'],
        'ㄸ': ['ㄷ', 'ㄷ'],
        'ㅃ': ['ㅂ', 'ㅂ'],
        'ㅆ': ['ㅅ', 'ㅅ'],
        'ㅉ': ['ㅈ', 'ㅈ'],
        'ㄳ': ['ㄱ', 'ㅅ'],
        'ㄵ': ['ㄴ', 'ㅈ'],
        'ㄶ': ['ㄴ', 'ㅎ'],
        'ㄺ': ['ㄹ', 'ㄱ'],
        'ㄻ': ['ㄹ', 'ㅁ'],
        'ㄼ': ['ㄹ', 'ㅂ'],
        'ㄽ': ['ㄹ', 'ㅅ'],
        'ㄾ': ['ㄹ', 'ㅌ'],
        'ㄿ': ['ㄹ', 'ㅍ'],
        'ㅀ': ['ㄹ', 'ㅎ'],
        'ㅄ': ['ㅂ', 'ㅅ'],
    }

    return double_jaum_map.get(char, [char])

# ====== 한글 결합, 입력 함수 ======
def compose_hangul_move_jonseong_to_choseong(label_list):
    global final_inputed_labels, inputed_moum
    choseong = label_list.pop()
    if (
            len(label_list) == 2 and
            is_full_choseong(label_list[0]) and
            is_full_jungseong(label_list[1])
    ):
        final_inputed_labels = final_inputed_labels + hgtk.letter.compose(*label_list)
    label_list.clear()
    label_list.append(choseong)
    inputed_moum = False

def compose_hangul(label_list):
    global final_inputed_labels, inputed_moum
    if (
            len(label_list) >= 2 and
            is_full_choseong(label_list[0]) and
            is_full_jungseong(label_list[1]) and
            (len(label_list) == 2 or (len(label_list) == 3 and is_full_jongseong(label_compose_check_list[2])))
    ):
        final_inputed_labels = final_inputed_labels + hgtk.letter.compose(*label_list)
    else:
        final_inputed_labels = final_inputed_labels + "".join(label_list)
    label_list.clear()
    inputed_moum = False

def input_label(label):
    global final_inputed_labels, inputed_moum
    inputed_moum = False
    final_inputed_labels = final_inputed_labels + label

# ====== 한글 입력 후 초기화 진행 함수 ======
def reset_jamo_input(label):
    global label_history, last_added_time, distinguished_SiotYu
    label_history.clear()
    last_added_time = time.time()

    # 겹모음일 경우 분해하여 추가
    if len(decompose_double_jaum(label)) == 2:
        decomposed = decompose_double_moum(label)
        inputed_jamo_list.extend(decomposed)
    else:
        inputed_jamo_list.append(label)

    # 인덱스 길이를 max_len으로 유지
    if len(inputed_jamo_list) > MAX_LEN:
        del inputed_jamo_list[0: len(inputed_jamo_list) - MAX_LEN]

    distinguished_SiotYu = '' # 판별된 ㅅ/ㅠ 초기화

# ====== ㅅ/ㅠ 판별 함수
def distinguish_SiotYu(input_jamo_list):
    # 유효한 자모만 필터링
    input_indices = [char_to_index[j] for j in input_jamo_list if j in char_to_index]
    input_pad = pad_sequences([input_indices], maxlen=MAX_LEN, padding='pre')

    pred = SiotYu_distinction_model.predict(input_pad, verbose=0)
    pred_idx = np.argmax(pred)
    pred_jamo = index_to_char[pred_idx]

    if pred_jamo in ['ㅅ', 'ㅠ']:
        return pred_jamo
    elif is_full_choseong(pred_jamo) or is_full_jongseong(pred_jamo):
        return 'ㅅ'
    elif is_full_jungseong(pred_jamo):
        return 'ㅠ'

# ====== 초기 모델: 한글 ======
current_model_key = "hangul"
model, scaler, le = load_model_set(current_model_key)
print("✅ 초기 모델: 한글 (1번 키로 다시 변경 가능)")

# ====== MediaPipe 초기화 ======
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# ====== 자모 조합, 출력용 ======
final_inputed_labels = ""  # 최종 입력 문자열
label_history = deque(maxlen=20)  # 최근 20 Frame 예측
last_added_time = 0
label_compose_check_list = []
inputed_moum = False
last_input = ""
double_moum_list = ['ㅘ', 'ㅙ', 'ㅚ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅢ']

distinguished_SiotYu = ''  # 매 프레임마다 판별 모델 호출을 방지하기 위한 값(판별 시 ㅅ/ㅠ 입력)
inputed_jamo_list = []  # LSTM에 입력값으로 넣을 이전 입력값

# ====== 실시간 웹캠 ======
ip_webcam_url = ("http://192.168.219.177:8080/video")
cap = cv2.VideoCapture(ip_webcam_url)

predict_cache = {}

while cap.isOpened():

    # ====== iPWebcam 프레임 동기화용 코드 ======
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
                    label, confidence = predict_with_interval(features, scaler, model, le, interval=0.1,cache=predict_cache)

                    # ㅅ/ㅠ를 인식했을 경우
                    if (label == 'ㅅ' or label == 'ㅠ') and distinguished_SiotYu == '':

                        # 첫 입력일 경우
                        if not label_compose_check_list and not final_inputed_labels:
                            distinguished_SiotYu = label = 'ㅅ'

                        else:
                            distinguished_SiotYu = label = distinguish_SiotYu(inputed_jamo_list)

                    # 이전 프레임에서 판별된 ㅅ/ㅠ가 있을 경우
                    elif distinguished_SiotYu in ['ㅅ', 'ㅠ']:
                        label = distinguished_SiotYu

                    # 시간 측정
                    current_time = time.time()
                    label_history.append((label, confidence, current_time))

                    # 입력조건(1초 이상, 정확도 80% 이상) 판별
                    recent = [(l, c) for l, c, t in label_history if current_time - t <= 1.0]
                    if recent:
                        labels = [l for l, _ in recent]
                        pred_label, count = Counter(labels).most_common(1)[0]
                        portion = count / len(recent)

                        if portion >= 0.8:
                            # 가장 최근 confidence로 판단
                            latest_conf = [c for l, c in recent if l == pred_label][-1]
                            if latest_conf >= 0.8 and current_time - last_added_time > 1.0: \
                                    # 조합 시작

                                # 자음 입력
                                if is_jaum(pred_label):
                                    # 초성 : 첫 입력
                                    if not label_compose_check_list:
                                        label_compose_check_list.append(pred_label)
                                        reset_jamo_input(pred_label)

                                    # 초성(자음 입력, 입력된 모음X) : 된소리(쌍자음) 판별 합격 (ㄱ + ㄱ = ㄲ)
                                    elif (
                                            label_compose_check_list
                                            and not inputed_moum
                                            and get_double_choseong(label_compose_check_list[-1],
                                                                    pred_label) is not None  # 초성 된소리 판별 합격
                                    ):
                                        label_compose_check_list[-1] = get_double_choseong(label_compose_check_list[-1],
                                                                                           pred_label)
                                        reset_jamo_input(pred_label)

                                    # 초성(자음 입력, 입력된 모음X) : 된소리(쌍자음) 판별 불합 (ㄱ + ㄴ)
                                    elif (
                                            not inputed_moum
                                            and get_double_choseong(label_compose_check_list[-1], pred_label) is None
                                    ):
                                        input_label(label_compose_check_list[-1])
                                        label_compose_check_list.clear()
                                        label_compose_check_list.append(pred_label)
                                        reset_jamo_input(pred_label)

                                    # 3-1. 종성 입력(입력된 모음 있음)
                                    elif inputed_moum:

                                        # 직전 글자가 모음 : 현재 입력을 받침으로
                                        if is_jaum(label_compose_check_list[-1]) is False:
                                            label_compose_check_list.append(pred_label)
                                            reset_jamo_input(pred_label)

                                        # 직전 글자가 자음 : 겹받침 판별
                                        elif is_jaum(label_compose_check_list[-1]):

                                            # 겹받침 판별 합격
                                            if get_double_jongseong(label_compose_check_list[-1],
                                                                    pred_label) is not None:
                                                label_compose_check_list[-1] = get_double_jongseong(
                                                    label_compose_check_list[-1], pred_label)
                                                reset_jamo_input(pred_label)

                                            # 겹받침 판별 불합
                                            elif get_double_jongseong(label_compose_check_list[-1], pred_label) is None:
                                                compose_hangul(label_compose_check_list)
                                                label_compose_check_list.append(pred_label)
                                                reset_jamo_input(pred_label)

                                        # 직전 글자가 이미 겹받침
                                        elif is_full_jongseong(label_compose_check_list[-1]):
                                            compose_hangul(label_compose_check_list)
                                            label_compose_check_list.append(pred_label)
                                            reset_jamo_input(pred_label)

                                # 모음 입력
                                elif is_jaum(pred_label) is False:

                                    # 마지막 입력 없음 : 단일 입력
                                    if not label_compose_check_list:
                                        input_label(pred_label)
                                        reset_jamo_input(pred_label)

                                    # 마지막 입력 있음
                                    elif label_compose_check_list:

                                        # 입력된 모음 없음
                                        if not inputed_moum:
                                            label_compose_check_list.append(pred_label)
                                            inputed_moum = True
                                            reset_jamo_input(pred_label)

                                        # 모음 입력 있음
                                        elif inputed_moum:
                                            # 이전 입력 : 자음
                                            if is_jaum(label_compose_check_list[-1]):
                                                compose_hangul_move_jonseong_to_choseong(label_compose_check_list)
                                                label_compose_check_list.append(pred_label)
                                                inputed_moum = True
                                                reset_jamo_input(pred_label)

                                            # 마지막 입력 : 모음
                                            elif is_jaum(label_compose_check_list[-1]) is False:

                                                # 겹모음 판별 : 합격
                                                if get_double_jungseong(label_compose_check_list[-1],
                                                                        pred_label) is not None:
                                                    label_compose_check_list[-1] = get_double_jungseong(
                                                        label_compose_check_list[-1], pred_label)
                                                    reset_jamo_input(pred_label)

                                                # 겹모음 판별 : 불합
                                                elif get_double_jungseong(label_compose_check_list[-1],
                                                                          pred_label) is None:
                                                    compose_hangul(label_compose_check_list)
                                                    input_label(pred_label) # 모음 단일 추가
                                                    reset_jamo_input(pred_label)

                                            # 마지막 입력 : 겹받침
                                            elif is_full_jongseong(label_compose_check_list[-1]):
                                                label_compose_check_list[-1], choseong = decompose_double_jaum(label_compose_check_list[-1])
                                                compose_hangul(label_compose_check_list)
                                                label_compose_check_list[0:0] = [choseong, pred_label]
                                                inputed_moum = True
                                                reset_jamo_input(pred_label)

                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    text = f"{label} ({confidence:.2f})"
                    frame = draw_text_with_pil(frame, text, (10, 30))

    if (
            len(label_compose_check_list) >= 2 and
            is_full_choseong(label_compose_check_list[0]) and
            is_full_jungseong(label_compose_check_list[1]) and
            (len(label_compose_check_list) == 2 or (
                    len(label_compose_check_list) == 3 and is_full_jongseong(label_compose_check_list[2])))
    ):
        frame = draw_text_with_pil(frame, final_inputed_labels + hgtk.letter.compose(*label_compose_check_list), (10, 80), color=(255, 255, 0))  # 완성 문자 출력
    else:
        frame = draw_text_with_pil(frame, final_inputed_labels + "".join(label_compose_check_list), (10, 80), color=(255, 255, 0))  # 완성 문자 출력

    cv2.imshow("🖐 실시간 손 모양 인식 (1: 한글 / 2: 숫자 / ESC: 종료)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
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
