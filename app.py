import os
from flask import Flask, request, render_template
import cv2
import numpy as np
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

# Ensure the upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# 모델 경로 설정 (SavedModel 형식)
model_path = "d:/train_medi"
model = load_model(model_path)

class_names = [
    "메가인",
    "밴드",
    "베아제",
    "알러샷",
    "제감골드",
    "타이레놀",
    "판콜A",
    "판피린",
    "후시딘",
    "훼스탈",
]

def preprocess_image(img):
    img = cv2.resize(img, (299, 299))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(model, img):
    preprocessed_img = preprocess_image(img)
    predictions = model.predict(preprocessed_img)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_class, confidence

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = file.filename
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
            img = cv2.imread(file_path)
            predicted_class, confidence = predict_image(model, img)
            return render_template(
                "index.html",
                prediction=predicted_class,
                confidence=confidence,
                image_path=file_path,
            )
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
