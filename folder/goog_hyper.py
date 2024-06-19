import tensorflow as tf
from keras.applications import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing import image
import numpy as np

# 모델 로드
base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(244, 244, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(10, activation="softmax")(x)  # 클래스 수에 맞게 수정
model = Model(inputs=base_model.input, outputs=predictions)

# 모델 컴파일
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])

# 모델 가중치 로드
model.load_weights('path_to_your_model_weights.h5')

def predict(image_path):
    img = image.load_img(image_path, target_size=(244, 244))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # 이미지 정규화
    prediction = model.predict(img_array)
    return np.argmax(prediction, axis=1)[0]  # 예측된 클래스 반환
