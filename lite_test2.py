import time
import numpy as np
import tensorflow as tf
from PIL import Image
import os

def measure_inference_time(model_path, image_path, input_dtype, runs=100):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    input_index = input_details[0]['index']
    expected_dtype = input_details[0]['dtype']
    input_shape = input_details[0]['shape']  # e.g. (1, 244, 244, 3)

    # 이미지 로드 및 전처리
    img = Image.open(image_path).convert("RGB")
    img = img.resize((input_shape[1], input_shape[2]))
    img_array = np.array(img)

    if expected_dtype == np.float32:
        img_array = img_array.astype(np.float32) / 255.0
    elif expected_dtype == np.uint8:
        img_array = img_array.astype(np.uint8)
    else:
        raise ValueError(f"지원되지 않는 입력 dtype: {expected_dtype}")

    input_data = np.expand_dims(img_array, axis=0).astype(expected_dtype)

    # 워밍업
    for _ in range(10):
        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()

    # 추론 시간 측정
    start = time.time()
    for _ in range(runs):
        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
    end = time.time()

    total_time = end - start
    avg_time = total_time / runs

    print(f"✅ 모델: {os.path.basename(model_path)}")
    print(f"총 소요 시간: {total_time:.4f}초")
    print(f"평균 추론 시간: {avg_time * 1000:.2f} ms")
    print("-" * 40)

# 모델 경로 및 입력 이미지 설정
models = {
    "float32": {
        "path": "D:/git/window_deep/model_float32.tflite",
        "dtype": np.float32
    },
    "float16": {
        "path": "D:/git/window_deep/model_float16.tflite",
        "dtype": np.float32  
    },
    "uint8": {
        "path": "D:/git/window_deep/model_uint8.tflite",
        "dtype": np.uint8
    }
}

# 테스트 이미지 1장 지정 (공통 사용)
image_directory = "D:/train_medi"
image_files = [f for f in os.listdir(image_directory) if f.endswith(('.jpg', '.jpeg', '.png'))]
if not image_files:
    raise FileNotFoundError("입력 이미지가 존재하지 않습니다.")
image_path = os.path.join(image_directory, image_files[0])  # 첫 번째 이미지 사용

# 측정 실행
print(f"📷 입력 이미지: {os.path.basename(image_path)}\n")
for name, config in models.items():
    measure_inference_time(config["path"], image_path, config["dtype"], runs=1000)
