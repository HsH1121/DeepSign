import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Add, Activation, BatchNormalization
from tensorflow.keras.utils import to_categorical
import joblib

# 1. 데이터 불러오기
df = pd.read_csv("hand_features_with_orientation.csv")  # 예: 15~16개 특징 + label

X = df.drop(columns=["label"]).values
y = df["label"].values

# 라벨 인코딩 및 원-핫 인코딩
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

# 정규화
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_cat, test_size=0.2, stratify=y_cat, random_state=42)

# 저장
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")

# 2. Residual Block 정의
def residual_block(x, units, dropout_rate=0.3):
    shortcut = x
    x = Dense(units)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(units)(x)
    x = Add()([shortcut, x])  # 잔차 연결
    x = Activation('relu')(x)
    return x

# 3. 모델 구성
input_dim = X.shape[1]
inputs = Input(shape=(input_dim,))
x = Dense(256)(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# Residual Block 2개
x = residual_block(x, 256, dropout_rate=0.4)
x = residual_block(x, 256, dropout_rate=0.3)

x = Dropout(0.3)(x)
outputs = Dense(y_cat.shape[1], activation='softmax')(x)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. 학습
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32)

# 5. 저장
model.save("residual_mlp_model.h5")
print("[✓] 모델 저장 완료")
