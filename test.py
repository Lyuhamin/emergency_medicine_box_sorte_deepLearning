import os
import numpy as np
from PIL import Image
import tensorflow as tf
import random
from tqdm import tqdm 

# 모델 경로 설정
MODEL_PATHS = {
    "float": "D:/git/window_deep/model_float32.tflite",
    "uint8": "D:/git/window_deep/model_uint8.tflite"
}

# 테스트 이미지 폴더 경로
TEST_IMAGE_DIR = "D:/train_medi"

# 클래스 이름 자동 수집 (디렉토리만 필터링)
class_names = sorted([
    d for d in os.listdir(TEST_IMAGE_DIR)
    if os.path.isdir(os.path.join(TEST_IMAGE_DIR, d))
])

# 클래스당 평가 이미지 수
NUM_SAMPLES_PER_CLASS = 10

def evaluate_model(model_path, quantized=False):
    print(f"\n🔍 평가 시작: {'Quantized (uint8)' if quantized else 'Float32'} 모델")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_dtype = input_details[0]['dtype']
    input_shape = input_details[0]['shape']
    img_height, img_width = input_shape[1], input_shape[2]

    total = 0
    correct = 0

    for class_index, class_name in enumerate(class_names):
        class_dir = os.path.join(TEST_IMAGE_DIR, class_name)
        all_images = [
            f for f in os.listdir(class_dir)
            if os.path.isfile(os.path.join(class_dir, f))
        ]

        sampled_images = random.sample(all_images, min(len(all_images), NUM_SAMPLES_PER_CLASS))

        print(f"\n▶ 클래스 '{class_name}' 평가 중...")

        for img_file in tqdm(sampled_images, desc=f"{class_name}", leave=False):
            img_path = os.path.join(class_dir, img_file)

            try:
                image = Image.open(img_path).convert('RGB').resize((img_width, img_height))
                image = np.array(image, dtype=np.float32) / 255.0

                if quantized:
                    image = (image * 255).astype(np.uint8)

                input_data = np.expand_dims(image, axis=0).astype(input_dtype)

                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])[0]
                predicted = np.argmax(output)

                if predicted == class_index:
                    correct += 1
                total += 1
            except Exception as e:
                print(f"❗ 오류 발생: {img_path} -> {e}")
                continue

    accuracy = correct / total * 100 if total > 0 else 0
    return accuracy

# 정확도 측정 실행
float_acc = evaluate_model(MODEL_PATHS["float"], quantized=False)
uint8_acc = evaluate_model(MODEL_PATHS["uint8"], quantized=True)

print(f"\n✅ Float32 모델 정확도: {float_acc:.2f}%")
print(f"✅ UINT8 모델 정확도: {uint8_acc:.2f}%")
