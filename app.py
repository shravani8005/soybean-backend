from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# =========================
# LOAD MODEL (SAFE PATH)
# =========================
model_path = os.path.join(os.path.dirname(__file__), "vgg16_seed_model.h5")
model = tf.keras.models.load_model(model_path)

IMG_SIZE = 224

# =========================
# PREPROCESS
# =========================
def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# =========================
# PREDICT
# =========================
def predict(img):
    x = preprocess(img)
    prob = model.predict(x, verbose=0)[0][0]

    if prob > 0.5:
        return "GOOD", float(prob)
    else:
        return "DEFECTIVE", float(1 - prob)

# =========================
# API ROUTE
# =========================
@app.route("/predict", methods=["POST"])
def predict_api():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]

    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    label, confidence = predict(img)

    return jsonify({
        "prediction": label,
        "confidence": round(confidence, 2)
    })

# =========================
# ROOT ROUTE (for testing)
# =========================
@app.route("/")
def home():
    return "Soybean Seed API is running 🚀"

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)