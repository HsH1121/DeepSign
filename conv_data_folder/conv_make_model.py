import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

# 1. CSV 로드
df = pd.read_csv('../hand_landmarks.csv')

# 2. X, y 분리
X = df.iloc[:, :-1].values  # (N, 42)
y = df.iloc[:, -1].values   # 라벨 (1~10)

# 3. 라벨 정규화 (0~9)
y = y - 1

# 4. (21, 2) 형태로 reshape → Conv1D 입력 준비
X = X.reshape(-1, 21, 2)

# 5. one-hot 인코딩
y_cat = to_categorical(y, num_classes=10)

# 6. 학습/검증 분리
X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# 7. Conv1D 모델 구성
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(21, 2)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(128, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

# 8. 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 9. 학습
model.fit(X_train, y_train, epochs=50, batch_size=32,
          validation_data=(X_val, y_val))

# 10. 저장
model.save('hand_pose_conv1d_model.h5')
