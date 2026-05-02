from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
import threading

# =========================
# LIMIT TF RESOURCE USAGE
# =========================
tf.config.set_visible_devices([], 'GPU')

app = Flask(__name__)

# =========================
# LOAD MODEL
# =========================
model_path = os.path.join(os.path.dirname(__file__), "vgg16_seed_model.h5")
model = tf.keras.models.load_model(model_path, compile=False)

print("✅ Model loaded successfully")

# =========================
# THREAD LOCK
# =========================
lock = threading.Lock()

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
# PREDICT (UNCHANGED)
# =========================
def predict(img):
    with lock:
        x = preprocess(img)
        prob = model.predict(x, verbose=0)[0][0]

    if prob > 0.5:
        return "GOOD", float(prob)
    else:
        return "DEFECTIVE", float(1 - prob)

# =========================
# API ROUTE (FINAL FIX)
# =========================
@app.route("/predict", methods=["POST"])
def predict_api():
    print("FILES:", request.files)
    print("DATA LENGTH:", len(request.data))

    file_bytes = None

    # ✅ Case 1: form-data
    if "image" in request.files:
        file = request.files["image"]
        file_bytes = np.frombuffer(file.read(), np.uint8)

    # ✅ Case 2: any file key fallback
    elif len(request.files) > 0:
        file = list(request.files.values())[0]
        file_bytes = np.frombuffer(file.read(), np.uint8)

    # ✅ Case 3: binary/raw
    elif len(request.data) > 0:
        file_bytes = np.frombuffer(request.data, np.uint8)

    else:
        return jsonify({"error": "No image provided"}), 400

    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    img = cv2.resize(img, (224, 224))

    label, confidence = predict(img)

    return jsonify({
        "prediction": label,
        "confidence": round(confidence, 2)
    })

# =========================
# ROOT ROUTE
# =========================
@app.route("/")
def home():
    return "Soybean Seed API is running 🚀"

# =========================
# RUN
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)