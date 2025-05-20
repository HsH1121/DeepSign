import cv2
import os
import mediapipe as mp

# 사용자 입력: 저장할 라벨 (1~10)
label = input("수어 숫자 (1~10)를 입력하세요: ")
save_dir = os.path.join("dataset_raw", label)
os.makedirs(save_dir, exist_ok=True)

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# 기존 파일 수 파악 → 번호 이어서 저장
existing = len(os.listdir(save_dir))
img_count = 0
max_additional = 500  # 추가 촬영할 최대 수

# 웹캠 열기
cap = cv2.VideoCapture(0)
print("[i] 웹캠 시작됨. 손 detect 후, 스페이스바 누르면 crop된 손 저장. ESC로 종료.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    hand_crop = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # 손 관절 좌표로 bounding box 계산
            x_list = [lm.x for lm in hand_landmarks.landmark]
            y_list = [lm.y for lm in hand_landmarks.landmark]
            xmin = int(min(x_list) * w) - 20
            ymin = int(min(y_list) * h) - 20
            xmax = int(max(x_list) * w) + 20
            ymax = int(max(y_list) * h) + 20

            xmin, ymin = max(xmin, 0), max(ymin, 0)
            xmax, ymax = min(xmax, w), min(ymax, h)

            hand_crop = frame[ymin:ymax, xmin:xmax]

            # 손 위치 디버그용으로 표시
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # 화면 표시
    cv2.putText(frame, f"Label: {label} | New: {img_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Hand Detection Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        if hand_crop is not None and hand_crop.size > 0:
            filename = f"{label}_{existing + img_count:04d}.jpg"
            save_path = os.path.join(save_dir, filename)
            cv2.imwrite(save_path, hand_crop)
            print(f"[✓] Saved: {filename}")
            img_count += 1

            if img_count >= max_additional:
                print(f"[i] {max_additional}장 저장 완료. 자동 종료합니다.")
                break
        else:
            print("[!] 손이 인식되지 않아 저장할 수 없습니다.")

    elif key == 27:  # ESC 종료
        print("[i] 사용자에 의해 종료됨.")
        break

cap.release()
cv2.destroyAllWindows()
