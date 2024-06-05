import os
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
import matplotlib.pyplot as plt

# Qt 플랫폼 플러그인 문제 해결을 위한 백엔드 설정
plt.switch_backend("Agg")

# 데이터셋 경로
dataset_path = "d:/train_medi"

# 데이터 증강 및 전처리
datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

train_datagen = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="training",
)

validation_datagen = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation",
)

'''
# 테스트 데이터 생성 - 문제 해결
test_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.1)
test_data = test_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation",
)
'''

# GoogLeNet 모델 불러오기
base_model = InceptionV3(weights="imagenet", include_top=False)

# 모델 구조 쌓기
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
predictions = Dense(train_datagen.num_classes, activation="softmax")(x)

# 최종 모델 정의
model = Model(inputs=base_model.input, outputs=predictions)

# 모델 컴파일
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 모델 학습
history = model.fit(
    train_datagen,
    epochs=20,  # 여기에서 epochs를 조절합니다.
    validation_data=validation_datagen,
)

# 정확도 그래프
plt.plot(history.history["accuracy"], label="train accuracy")
plt.plot(history.history["val_accuracy"], label="validation accuracy")
plt.title("Model accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="upper left")
plt.savefig("accuracy.png")  # 그래프를 파일로 저장

# 손실 그래프
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="validation loss")
plt.title("Model loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="upper left")
plt.savefig("loss.png")  # 그래프를 파일로 저장


'''
# 테스트 데이터 평가
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
'''