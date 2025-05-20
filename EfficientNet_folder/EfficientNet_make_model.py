import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# 1. 데이터셋 경로
base_dir = "../dataset_raw"  # 폴더 안에 1~10 하위 폴더가 있어야 함

# 2. 하이퍼파라미터
img_size = (224, 224)
batch_size = 32
num_classes = 10
epochs = 15

# 3. 데이터 증강 및 정규화
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='sparse',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation',
    shuffle=False
)

# 4. 모델 구성 (Functional API)
input_tensor = Input(shape=(224, 224, 3))
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=input_tensor)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=input_tensor, outputs=output)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 5. 콜백 설정 (가중치만 저장)
callbacks = [
    ModelCheckpoint(
        filepath="best_effnet_weights.h5",
        save_best_only=True,
        save_weights_only=True,  # 핵심!
        monitor='val_accuracy',
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
]

# 6. 학습
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=callbacks
)

# 7. 최종 가중치 저장
model.save_weights("final_effnet_weights.h5")

# 8. 클래스 인덱스 출력
print("클래스 인덱스 매핑:", train_generator.class_indices)
