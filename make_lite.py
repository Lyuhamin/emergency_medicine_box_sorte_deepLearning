import tensorflow as tf
import numpy as np
import os
from keras.applications import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

# 1. 저장 위치 설정
SAVE_DIR = "D:/git/window_deep"
os.makedirs(SAVE_DIR, exist_ok=True)

# 2. 클래스 수 정의 (학습할 때 사용한 클래스 수와 동일해야 함)
NUM_CLASSES = 10

# 3. 모델 구조 복원
def build_model():
    base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(244, 244, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(NUM_CLASSES, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# 4. 모델 로드
model = build_model()
model.load_weights("D:/git/window_deep/path_to_your_model_weights.h5") 
model.summary()

# 5. Float32 TFLite 저장
converter_fp32 = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_fp32 = converter_fp32.convert()
with open(os.path.join(SAVE_DIR, "model_float32.tflite"), "wb") as f:
    f.write(tflite_fp32)
print("✅ Float32 모델 저장 완료")

# 6. Float16 양자화
converter_fp16 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
converter_fp16.target_spec.supported_types = [tf.float16]
tflite_fp16 = converter_fp16.convert()
with open(os.path.join(SAVE_DIR, "model_float16.tflite"), "wb") as f:
    f.write(tflite_fp16)
print("✅ Float16 모델 저장 완료")

# 7. Uint8 양자화
def representative_data_gen():
    for _ in range(100):
        yield [np.random.rand(1, 244, 244, 3).astype(np.float32)]

converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_data_gen
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type = tf.uint8
converter_int8.inference_output_type = tf.uint8
tflite_int8 = converter_int8.convert()
with open(os.path.join(SAVE_DIR, "model_uint8.tflite"), "wb") as f:
    f.write(tflite_int8)
print("✅ Uint8 모델 저장 완료")
