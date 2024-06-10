<<<<<<< Updated upstream
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from goog_hyper import predict

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        prediction = predict(file_path)
        return jsonify({'class': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
=======
import os
from flask import Flask, request, render_template
import cv2
import numpy as np
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.layers import TFSMLayer
from tensorflow.keras.models import Sequential

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

# Ensure the upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# 모델 경로 설정 (SavedModel 형식)
model_path = "/home/lyuha/training_05_21/train"
model = Sequential([TFSMLayer(model_path, call_endpoint="serving_default")])

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
    img = cv2.resize(img, (244, 244))
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
>>>>>>> Stashed changes
    app.run(debug=True)
