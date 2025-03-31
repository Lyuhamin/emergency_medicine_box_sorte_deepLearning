import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
import matplotlib.pyplot as plt

# 데이터 경로 설정
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
test_data_dir = 'data/test'

# 하이퍼 파라미터 설정
img_width, img_height = 299, 299  # InceptionV3는 기본적으로 299x299 크기의 이미지를 사용하지만 224x224로 줄여서 사용
batch_size = 32
epochs = 30

# 데이터 전처리 및 증강
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# InceptionV3 모델 불러오기 (사전 학습된 가중치 사용, 최상위 레이어는 포함하지 않음)
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# 최상위 레이어 추가
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  # 완전 연결 레이어
predictions = Dense(10, activation='softmax')(x)  # 10개의 클래스로 출력

# 전체 모델 구성
model = Model(inputs=base_model.input, outputs=predictions)

# 기본 모델의 가중치를 고정 (훈련되지 않도록)
for layer in base_model.layers:
    layer.trainable = False

# 모델 컴파일
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 훈련
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator)

# 하이퍼 파라미터 조정을 위한 주석
# 하이퍼 파라미터 조정 시 아래 코드 주석을 해제하고 값을 조정
# for layer in model.layers[:249]:
#     layer.trainable = False
# for layer in model.layers[249:]:
#     layer.trainable = True
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

# 모델 저장
model.save('my_model.h5')

# 그래프 이미지를 저장할 폴더 경로
save_dir = "d:/images_gr"  # 폴더 경로를 원하는 경로로 변경

# 폴더가 존재하지 않으면 생성
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 훈련 및 검증 결과 시각화 및 저장
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(os.path.join(save_dir, 'model_accuracy.png'))  # 그래프 이미지 저장
plt.show()

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(os.path.join(save_dir, 'model_loss.png'))  # 그래프 이미지 저장
plt.show()
