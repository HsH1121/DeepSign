import cv2
import mediapipe as mp
import numpy as np
import os
import csv

# 폴더 경로 설정
dataset_dir = 'dataset_raw'  # 저장해둔 이미지 폴더 최상위 경로
output_csv = ('hand_landmarks_with_han.csv')  # 최종 CSV 파일 이름

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# CSV 파일 생성
with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)

    # 첫 줄 (헤더) 작성
    header = []
    for i in range(21):
        header += [f'x{i}', f'y{i}']
    header.append('label')
    writer.writerow(header)
    
    # 레이블별 인식 성공률 리스트
    success_rate_dict = {}
    
    # 폴더 내 모든 이미지 순회
    for label_name in sorted(os.listdir(dataset_dir)):
        label_path = os.path.join(dataset_dir, label_name)
        if not os.path.isdir(label_path):
            continue

        success_count = 0
        count = 0

        for img_name in sorted(os.listdir(label_path)):
            img_path = os.path.join(label_path, img_name)

            # ---- 한글 경로 호환 방식으로 이미지 로드 ----
            try:
                with open(img_path, 'rb') as f:
                    img_array = np.asarray(bytearray(f.read()), dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"[!] 파일 열기 실패: {img_path} — {e}")
                continue

            if img is None:
                print(f"[!] 이미지 로드 실패: {img_path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img_rgb)

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                row = []
                for lm in hand.landmark:
                    row.append(lm.x)
                    row.append(lm.y)
                row.append(label_name)
                writer.writerow(row)
                print(f"[✓] 좌표 저장 완료: {img_name}")
                success_count += 1
            else:
                print(f"[!] 손 인식 실패: {img_name} (스킵)")

            count += 1

        # 레이블당 인식 성공률
        success_rate_dict[label_name] = success_count / count * 100

    print("레이블당 인식 성공률")
    for label_name, success_rate in success_rate_dict.items():
        print(f"[{label_name}] 인식률: {success_rate}%")

print(f"\n[i] CSV 저장 완료: {output_csv}")
