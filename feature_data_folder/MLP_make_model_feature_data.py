import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# 1. 데이터 로드
df = pd.read_csv("../hand_features_with_orientation.csv")
X = df.drop(columns=['label']).values
y = df['label'].values  # 1 ~ 10

# 2. Label 인코딩 (1~10 → 0~9)
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # ex: [1,2,...,10] → [0,1,...,9]
num_classes = len(np.unique(y_encoded))
y_cat = to_categorical(y_encoded, num_classes)

# 3. 정규화
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 4. 데이터 분할
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_cat, test_size=0.3, random_state=42, stratify=y_encoded)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=np.argmax(y_temp, axis=1))

# 5. 모델 구성
model = Sequential([
    Dense(64, input_dim=X.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])

# 6. 학습
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=16, callbacks=[early_stop])

# 7. 평가
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {acc:.4f}")

# 8. 저장
model.save("sign_language_model.h5")         # 모델
joblib.dump(scaler, "../scaler.pkl")            # 정규화 스케일러
joblib.dump(le, "../label_encoder.pkl")         # 라벨 인코더

print("\n📁 모델, 스케일러, 라벨 인코더 저장 완료!")
