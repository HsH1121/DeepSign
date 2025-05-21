import pandas as pd
import numpy as np
import joblib

from keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.initializers import HeNormal
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


# 1. 데이터 로드
df = pd.read_csv('hand_landmarks.csv', encoding = "cp949")

# 2. 특징/레이블 분리
X = df.iloc[:, :-1].values  # 42개 좌표
y = df.iloc[:, -1].values   # 정수 라벨 (숫자, 한글)
num_labels = len(np.unique(y))

# 3. 한글 레이블 데이터를 정수로 매핑
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ⚠️ 라벨을 0~9로 맞추기 위해 -1 (필요에 따라 수정)
#y = y - 1

# 3. one-hot encoding
y_cat = to_categorical(y_encoded, num_classes=len(np.unique(y_encoded)))

# 4. 훈련/검증 분리
X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.15, random_state=42)

# 5. MLP 모델 구성
model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_labels, activation='softmax')
])

# 6. 컴파일
model.compile(optimizer='nadam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 7. 학습
model.fit(X_train, y_train, epochs=20, batch_size=64,
          validation_data=(X_val, y_val))

# 8. 저장
model.save('hand_pose_classifier_20_64_nadam_with_han.h5')
joblib.dump(le, "label_encoder.pkl")