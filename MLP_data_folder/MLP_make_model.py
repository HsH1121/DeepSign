import pandas as pd
import numpy as np
from keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.initializers import HeNormal

# 1. 데이터 로드
df = pd.read_csv('../hand_landmarks.csv')

# 2. 특징/레이블 분리
X = df.iloc[:, :-1].values  # 42개 좌표
y = df.iloc[:, -1].values   # 정수 라벨 (1~10)

# ⚠️ 라벨을 0~9로 맞추기 위해 -1 (필요에 따라 수정)
y = y - 1

# 3. one-hot encoding
y_cat = to_categorical(y, num_classes=10)

# 4. 훈련/검증 분리
X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.15, random_state=42)

# 5. MLP 모델 구성
model = Sequential([
    Dense(256, activation='relu', input_shape=(42,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

# 6. 컴파일
model.compile(optimizer='nadam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 7. 학습
model.fit(X_train, y_train, epochs=20, batch_size=64,
          validation_data=(X_val, y_val))

# 8. 저장
model.save('hand_pose_classifier_20_64_nadam.h5')
