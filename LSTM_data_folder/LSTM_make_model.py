import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical

# 1. 데이터 로드
df = pd.read_csv('../hand_landmarks.csv')

# 2. X, y 분리
X = df.iloc[:, :-1].values  # 42개 좌표
y = df.iloc[:, -1].values   # 정수 라벨 (1~10)

# 3. 라벨 정리 (0~9로 맞추기)
y = y - 1

# 4. 입력 데이터를 (21,2) 형태로 변환
X = X.reshape(-1, 21, 2)

# 5. one-hot encoding
y_cat = to_categorical(y, num_classes=10)

# 6. 학습/검증 분리
X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# 7. LSTM 모델 구성
model = Sequential([
    LSTM(64, activation='tanh', return_sequences=True, input_shape=(21, 2)),
    Dropout(0.3),
    LSTM(32, activation='tanh'),
    Dropout(0.3),
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

# 10. 모델 저장
model.save('hand_pose_lstm_model.h5')
