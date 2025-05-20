import cv2
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense

# 1. 모델 정의 (학습과 동일한 구조)
input_tensor = Input(shape=(224, 224, 3))
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=input_tensor)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(10, activation='softmax')(x)
model = Model(inputs=input_tensor, outputs=output)

# 2. 가중치 불러오기
model.load_weights("final_effnet_weights.h5")
print("[✓] 가중치 로드 완료")

# 3. 클래스 라벨 정의 (숫자 1~10)
class_labels = [str(i) for i in range(1, 11)]

# 4. WebCam 실행
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 5. 중앙 사각형 ROI 추출
    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2
    size = min(h, w) // 2
    roi = frame[cy - size//2:cy + size//2, cx - size//2:cx + size//2]

    # 6. 모델 입력 처리
    input_image = cv2.resize(roi, (224, 224))
    input_array = np.expand_dims(preprocess_input(input_image.astype(np.float32)), axis=0)

    # 7. 예측
    pred = model.predict(input_array)
    pred_label = class_labels[np.argmax(pred)]
    confidence = np.max(pred)

    # 8. 결과 출력
    cv2.rectangle(frame, (cx - size//2, cy - size//2), (cx + size//2, cy + size//2), (0, 255, 0), 2)
    cv2.putText(frame, f"Pred: {pred_label} ({confidence:.2f})", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("EfficientNet Sign Prediction", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키 종료
        break

cap.release()
cv2.destroyAllWindows()
