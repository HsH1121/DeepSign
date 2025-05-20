import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# 모델 불러오기 (LSTM용)
model = load_model('hand_pose_lstm_model.h5')

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)

# 라벨 설정 (1~10)
labels = [str(i) for i in range(1, 11)]

# 웹캠 열기
cap = cv2.VideoCapture(0)
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
            coords = []
            for lm in hand_landmarks.landmark:
                coords.append([lm.x, lm.y])  # [x, y]로 묶어서 저장

            coords = np.array(coords).reshape(1, 21, 2)  # (1,21,2)로 reshape

            pred = model.predict(coords)
            pred_label = labels[np.argmax(pred)]
            confidence = np.max(pred)

            # 손 bounding box 계산
            x_list = [lm.x for lm in hand_landmarks.landmark]
            y_list = [lm.y for lm in hand_landmarks.landmark]
            xmin = int(min(x_list) * w) - 20
            ymin = int(min(y_list) * h) - 20
            xmax = int(max(x_list) * w) + 20
            ymax = int(max(y_list) * h) + 20

            xmin, ymin = max(xmin, 0), max(ymin, 0)
            xmax, ymax = min(xmax, w), min(ymax, h)

            # 손 bounding box 시각화
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"{pred_label} ({confidence:.2f})", (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 화면 출력
    cv2.imshow("Hand Gesture Recognition (LSTM)", frame)

    if cv2.waitKey(1) == 27:  # ESC 키로 종료
        break

cap.release()
cv2.destroyAllWindows()
