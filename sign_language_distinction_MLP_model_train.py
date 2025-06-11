import pandas as pd
import numpy as np

from keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd
# 1. 데이터 로드
df = pd.read_csv('digit_feature.csv', encoding = "cp949")
print(df['label'].value_counts())

# 2. 특징/레이블 분리
X = df.iloc[:, :-1].values  # 42개 좌표
y = df.iloc[:, -1].values   # 정수 라벨 (숫자, 한글)
num_labels = len(np.unique(y))

# scaler.pkl 파일 재생성
scaler = MinMaxScaler()
scaler.fit(X)  # 여기서 X는 25개의 feature가 들어있는 데이터셋
X = scaler.transform(X)

# 3. 한글 레이블 데이터를 정수로 매핑
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("레이블 인덱스 순서:", list(le.classes_))

# ⚠️ 라벨을 0~9로 맞추기 위해 -1 (필요에 따라 수정)
#y = y - 1

# 3. one-hot encoding
y_cat = to_categorical(y_encoded, num_classes=len(np.unique(y_encoded)))

# 4. 훈련/검증 분리
X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat, test_size=0.15, random_state=42, stratify=y_encoded
)

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
model.fit(X_train, y_train, epochs=30, batch_size=64,
          validation_data=(X_val, y_val))

# 8. 레이블당 정확도
y_val_pred = model.predict(X_val)
y_pred_labels = np.argmax(y_val_pred, axis=1)
y_true_labels = np.argmax(y_val, axis=1)

unique_labels = np.unique(y_true_labels)

print("레이블당 정확도")
for label in unique_labels:
    indices = (y_true_labels == label)
    acc = accuracy_score(y_true_labels[indices], y_pred_labels[indices])
    label_name = le.inverse_transform([label])[0]  # 한글 라벨 디코딩
    print(f"[{label_name}] 정확도: {acc * 100:.2f}%")

# 9. 저장
model.save('model_digit.h5')
joblib.dump(le, "label_encoder_digit.pkl")
joblib.dump(scaler, "scaler_digit.pkl")
