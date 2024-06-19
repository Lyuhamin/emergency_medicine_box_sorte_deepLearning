import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# 데이터 경로
data_dir = "c:/Users/KNUT/train_medi"

# 데이터 생성기
img_width, img_height = (244, 244)
batch_size = 32
epochs = 30
learning_rate = 0.0001

# 데이터 제너레이터 생성
datagen = ImageDataGenerator(
    rescale=1.0 / 255, 
    rotation_range=40,  # 이미지 회전 범위
    width_shift_range=0.2,  # 이미지 수평 이동 범위
    height_shift_range=0.2,  # 이미지 수직 이동 범위
    shear_range=0.2,  # 이미지 전단 변환 범위
    zoom_range=0.2,  # 이미지 확대 범위
    horizontal_flip=True,  # 이미지 수평 반전
    fill_mode="nearest",  # 변환 중 생기는 빈 공간을 채우는 방식
    validation_split = 0.2 # 20%의 데이터를 검증 데이터로 사용
) 

# 학습 데이터 생성기
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="binary",
    subset="training",
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="binary",
    subset="validation",
)

# VGG16 모델 불러오기 (사전 학습된 가중치 사용, 최상위 층은 포함하지 않음)
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))

# 새로운 레이어 추가
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)  # Dropout 레이어 추가하여 과적합 방지
predictions = Dense(1, activation="sigmoid")(x)  # 이진 분류에 맞게 수정

# 모델 정의
model = Model(inputs=base_model.input, outputs=predictions)

# 기본 VGG16 모델의 가중치를 고정
for layer in base_model.layers:
    layer.trainable = False

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=learning_rate), loss="binary_crossentropy", metrics=["accuracy"])

# 모델 학습
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    epochs=epochs,
)

# 그래프 이미지를 저장할 폴더 경로
save_dir = "c:/Users/KNUT/images_gr"  # 폴더 경로를 원하는 경로로 변경

# 학습 및 검증 정확도와 손실 값을 Python 리스트로 변환
def save_plot(history, save_dir):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    # 학습 및 검증 정확도와 손실 그래프 그리기
    epochs_range = range(epochs)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")
    plt.savefig(os.path.join(save_dir, 'Validation Accuracy4.png'))

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.savefig(os.path.join(save_dir, 'Validation Loss4.png'))

    # 최종 학습 및 검증 정확도 출력
    print(f"Final Training Accuracy: {acc[-1] * 100:.2f}%")
    print(f"Final Validation Accuracy: {val_acc[-1] * 100:.2f}%")

# 그래프 이미지를 저장합니다
save_plot(history, save_dir)