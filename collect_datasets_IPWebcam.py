import cv2
import os
import mediapipe as mp
import time
import sys
import numpy as np
from PIL import ImageFont, ImageDraw, Image # 한글 인식을 위한 라이브러리

# 한글 폰트 파일 경로 지정
font_path = "NanumGothic.ttf"
font = ImageFont.truetype(font_path, 40)

# 사용자 입력 : 저장할 라벨 (숫자, 한글 자·모음)
label = input("수어 데이터(숫자, 한글)를 입력하세요: ")
save_dir = os.path.join("dataset_raw", label)
os.makedirs(save_dir, exist_ok=True)

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# 기존 파일 수 파악 → 번호 이어서 저장
existing = len(os.listdir(save_dir))
img_count = 0
max_additional = 800  # 추가 촬영할 최대 수

# 웹캠 열기
ip_webcam_url = ("http://192.168.219.177:8080/video")

cap = cv2.VideoCapture(ip_webcam_url)

print("[i] 웹캠 시작됨. 손 detect 후, 스페이스바 누르면 crop된 손 저장. ESC로 종료.")

last_saved_time = 0 # 마지막 저장 시간

padding = 80

while True:

    # ====== iPWebcam 프레임 동기화용 코드 ======
    for _ in range(5):
        cap.grab()

    ret, frame = cap.retrieve()
    if not ret:
        print("❌ 프레임을 불러올 수 없습니다.")
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    successed = hands.process(frame_rgb)

    hand_crop = None

    if successed.multi_hand_landmarks:
        for hand_landmarks in successed.multi_hand_landmarks:
            # 손 관절 좌표로 bounding box 계산
            x_list = [lm.x for lm in hand_landmarks.landmark]
            y_list = [lm.y for lm in hand_landmarks.landmark]
            xmin = int(min(x_list) * w) - padding
            ymin = int(min(y_list) * h) - padding
            xmax = int(max(x_list) * w) + padding
            ymax = int(max(y_list) * h) + padding

            xmin, ymin = max(xmin, 0), max(ymin, 0)
            xmax, ymax = min(xmax, w), min(ymax, h)

            hand_crop = frame[ymin:ymax, xmin:xmax]

            # 손 위치 디버그용으로 표시
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # 화면 표시
    # OpenCV 이미지 → PIL 이미지로 변환
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    text = f"Label: {label} | New: {img_count}"
    draw.text((10, 10), text, font=font, fill=(0, 255, 0))

    # 다시 OpenCV 이미지로 변환
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
    cv2.imshow("Hand Detection Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    #로그 확인용 코드
    if key == ord(' ') and (time.time() - last_saved_time) > 0.05:
        print("[Debug] 스페이스바 감지됨")
        if hand_crop is not None:
            print(f"[Debug] hand_crop.size = {hand_crop.size}")
        else:
            print("[Debug] hand_crop is None")

        if hand_crop is not None and hand_crop.size > 0:
            filename = f"{label}_{existing + img_count:04d}.jpg"
            save_path = os.path.join(save_dir, filename)
            successed, encoded_img = cv2.imencode('.jpg', hand_crop)
            if successed:
                with open(save_path, mode='wb') as f:
                    encoded_img.tofile(f)
                print(f"[✓] 저장성공: {filename}")
                img_count += 1
            else:
                print(f"[✗] 저장 실패 (인코딩 실패): {filename}")
            last_saved_time = time.time()

            if img_count >= max_additional:
                print(f"[i] {max_additional}장 저장 완료. 자동 종료합니다.")
                sys.exit()
        else:
            print("[!] 손이 인식되지 않아 저장할 수 없습니다.")

    elif key == 27:  # ESC 종료
        print("[i] 사용자에 의해 종료됨.")
        break
cap.release()
cv2.destroyAllWindows()
