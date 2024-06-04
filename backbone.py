import os
'''
import splitfolders
splitfolders.ratio('./hmtest1/helmet_test/helmet_crop_fin', 
                   output="./hmtest1/helmet_test/helmet_crop_123/", seed=22,ratio=(0.8, 0.1, 0.1))
'''
# 이미지 처리용 라이브러리
from tensorflow.keras.preprocessing.image import ImageDataGenerator # 전처리
#from tensorflow.keras.preprocessing.image import array_to_img
#from tensorflow.keras.preprocessing.image import img_to_array
#from tensorflow.keras.preprocessing.image import load_img
# 0~1 사이값으로 픽셀값을 변환
base_dir = "C:/Users/KNUT/Desktop/deep_learning_pic/Wfill_person"  #진행상황(현재 l2=0.001. 드롭아웃 풀링레이어, 댄스 추가)

train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")
validation_dir = os.path.join(base_dir, "val")

train_yes_helmet_dir = os.path.join(train_dir, "L")
train_no_helmet_dir = os.path.join(train_dir, "M")
train_Cap_dir = os.path.join(train_dir, "S")

test_yes_helmet_dir = os.path.join(test_dir, "L")
test_no_helmet_dir = os.path.join(test_dir, "M")
test_Cap_dir = os.path.join(test_dir, "S")

validation_yes_helmet_dir = os.path.join(validation_dir, "L")
validation_no_helmet_dir = os.path.join(validation_dir, "M")
validation_Cap_dir = os.path.join(validation_dir, "S")

# 각 폴더에 있는 파일의 개수를 출력
print("훈련용 L 데이터 개수 : ", len(os.listdir(train_yes_helmet_dir)))
print("훈련용 M 데이터 개수 : ", len(os.listdir(train_no_helmet_dir)))
print("훈련용 S 데이터 개수 : ", len(os.listdir(train_Cap_dir)))

print("테스트용 L 데이터 개수 : ", len(os.listdir(test_yes_helmet_dir)))
print("테스트용 M 데이터 개수 : ", len(os.listdir(test_no_helmet_dir)))
print("테스트용 S 데이터 개수 : ", len(os.listdir(test_Cap_dir)))

print("검증용 L 데이터 개수 : ", len(os.listdir(validation_yes_helmet_dir)))
print("검증용 M 데이터 개수 : ", len(os.listdir(validation_no_helmet_dir)))
print("검증용 S 데이터 개수 : ", len(os.listdir(validation_Cap_dir)))

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# 이미지 전처리
# 폴더에 있는 이미지를 전처리

train_generator = train_datagen.flow_from_directory(

    # 폴더명
    train_dir,
    # 이미지 크기를 동일한 크기로 변환
    target_size = (150, 150),
    # 한 번에 전처리할 이미지의 수
    batch_size = 32,
    # 라벨링 : binary 이진라벨링, categorical 다중라벨링
    # 라벨링 방법 : 폴더명의 첫 문자의 알파벳으로 0부터 부여
    class_mode = "sparse",
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(

    # 폴더명
    validation_dir,
    # 이미지 크기를 동일한 크기로 변환
    target_size = (150, 150),
    # 한 번에 전처리할 이미지의 수
    batch_size = 32,
    # 라벨링 : binary 이진라벨링, categorical 다중라벨링
    # 라벨링 방법 : 폴더명의 첫 문자의 알파벳으로 0부터 부여
    class_mode = "sparse",
    shuffle=True
)
real_test_generator = test_datagen.flow_from_directory(

    # 폴더명
    test_dir,
    # 이미지 크기를 동일한 크기로 변환
    target_size = (150, 150),
    # 한 번에 전처리할 이미지의 수
    batch_size = 32,
    # 라벨링 : binary 이진라벨링, categorical 다중라벨링
    # 라벨링 방법 : 폴더명의 첫 문자의 알파벳으로 0부터 부여
    class_mode = "sparse",
    shuffle=True
)

#for data_batch, labels_batch in train_generator:
#    print('배치데이터 크기:',data_batch.shape)
#    print('배치레이블 크기:',labels_batch.shape)
#    print(train_generator.class_indices)
#    print(test_generator.class_indices)
#    break

import tensorflow as tf
import h5py
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionV3

from tensorflow.keras import regularizers

sgd = keras.optimizers.SGD(lr=0.01, momentum=0.0, nesterov=False)
rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-7)
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False)
opt = keras.optimizers.RMSprop(lr=0.002, rho=0.9, epsilon=1e-7)
 



'''
#LeNet+l2
model1 = Sequential()
model1.add(Conv2D(32, kernel_size=(3, 3), padding="same", activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(150, 150, 3)))
model1.add(MaxPooling2D((2,2), padding='same'))
model1.add(Conv2D(32, kernel_size=(3, 3), padding="same", activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model1.add(MaxPooling2D((2,2), padding='same'))
model1.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model1.add(MaxPooling2D((2,2), padding='same'))
model1.add(Flatten())
model1.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model1.add(Dense(3, activation='softmax', kernel_regularizer=regularizers.l2(0.001)))
'''

'''
#VGG16+드롭아웃
from keras.applications.vgg16 import VGG16
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
conv_base.trainable = False

model1 = Sequential()
model1.add(conv_base)
model1.add(Flatten())  # 이전 CNN 레이어에서 나온 3차원 배열은 1차원으로 뽑아줍니다
model1.add(Dense(units=64, activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(units=3, activation="softmax"))
'''

'''
#LeNet+dropout
model1 = Sequential()
model1.add(Conv2D(32, kernel_size=(3, 3), padding="same", activation='relu', input_shape=(150, 150, 3)))
model1.add(MaxPooling2D((2,2), padding='same'))
model1.add(Dropout(0.25))
model1.add(Conv2D(32, kernel_size=(3, 3), padding="same", activation='relu'))
model1.add(MaxPooling2D((2,2), padding='same'))
model1.add(Dropout(0.25))
model1.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation='relu'))
model1.add(MaxPooling2D((2,2), padding='same'))
model1.add(Dropout(0.25))
model1.add(Flatten())
model1.add(Dense(64, activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(3, activation='softmax'))
'''


#LeNet
model1 = Sequential()
model1.add(Conv2D(32, kernel_size=(3, 3), padding="same", activation='relu', input_shape=(150, 150, 3)))
model1.add(MaxPooling2D((2,2), padding='same'))
model1.add(Conv2D(32, kernel_size=(3, 3), padding="same", activation='relu'))
model1.add(MaxPooling2D((2,2), padding='same'))
model1.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation='relu'))
model1.add(MaxPooling2D((2,2), padding='same'))
model1.add(Flatten())
model1.add(Dense(64, activation='relu'))
model1.add(Dense(3, activation='softmax'))



'''
base_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# ResNet50의 일부 레이어 동결
for layer in base_resnet.layers:
    layer.trainable = False

# 모델 정의
model1 = Sequential([
    base_resnet,
    GlobalAveragePooling2D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model1.summary()
'''


'''
base_inception = InceptionV3(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# InceptionV3
for layer in base_inception.layers:
    layer.trainable = False

# 모델 정의
model1 = Sequential([
    base_inception,
    GlobalAveragePooling2D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])
'''


'''
#xception
base_xception = Xception(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
# Xception의 일부 레이어 동결
for layer in base_xception.layers:
    layer.trainable = False
# 모델 정의
model1 = Sequential([
    base_xception,
    GlobalAveragePooling2D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model1.summary()
'''


model1.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["acc"])
# generator: 제너레이터를 설정
h1 = model1.fit_generator(generator=train_generator, epochs=30, steps_per_epoch=20,
                          validation_data=test_generator)


#new_model=tf.keras.models.load_model("model1.h5")
import matplotlib.pyplot as plt

acc = h1.history["acc"] 
val_acc = h1.history["val_acc"]
loss = h1.history["loss"] 
val_loss = h1.history["val_loss"]

plt.plot(range(1, len(acc)+1), acc, label="train_acc")
plt.plot(range(1, len(acc)+1), val_acc, label="val_acc")
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.figure()

plt.plot(range(1, len(acc)+1), loss, label='Training loss')
plt.plot(range(1, len(acc)+1), val_loss, label='Validation loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
#Model().save_weights('model1.h5')
score = model1.evaluate(real_test_generator, verbose=1)
print('정답률 = ', score[1],'loss=', score[0])

import os
import matplotlib.pyplot as plt

# 그래프 이미지를 저장할 폴더 경로
save_dir = "C:/Users/KNUT/Desktop/그래프 모음/"  # 폴더 경로를 원하는 경로로 변경



# 그래프를 저장할 함수
def save_plot(history, save_dir):
    acc = history.history["acc"]
    val_acc = history.history["val_acc"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    # 정확도 그래프
    plt.figure()
    plt.plot(range(1, len(acc) + 1), acc, label="train_acc")
    plt.plot(range(1, len(acc) + 1), val_acc, label="val_acc")
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.savefig(os.path.join(save_dir, 'accuracy.png'))

    # 손실 그래프
    plt.figure()
    plt.plot(range(1, len(loss) + 1), loss, label='Training loss')
    plt.plot(range(1, len(loss) + 1), val_loss, label='Validation loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(save_dir, 'loss.png'))

# 그래프 이미지를 저장합니다
save_plot(h1, save_dir)



import os
import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt

base_dir = "C:/Users/KNUT/Desktop/deep_learning_pic/pic"
train_dir = os.path.join(base_dir, "train")

train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical',
    shuffle=False
)

# VGG16 모델 불러오기
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# 중간층의 출력을 특징 벡터로 사용하기 위한 모델 생성
feature_extractor = Model(inputs=vgg16.input, outputs=vgg16.get_layer('block5_conv2').output)

# 특징 벡터 추출
features = feature_extractor.predict(train_generator)

# 클래스 확인
classes = train_generator.classes

# PCA를 사용하여 2차원으로 차원 축소
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features.reshape(features.shape[0], -1))

# 산점도 그리기
plt.figure(figsize=(10, 8))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=classes, cmap='viridis')
plt.colorbar()
plt.title('Scatter Plot of PCA Features')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# 이미지 파일로 저장
save_dir = "C:/Users/KNUT/Desktop/scatter_graph"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'scatter_plot.png')
plt.savefig(save_path)
plt.show()