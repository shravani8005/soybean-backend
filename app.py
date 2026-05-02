from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
import threading

app = Flask(__name__)

# =========================
# LOAD MODEL (SAFE PATH)
# =========================
model_path = os.path.join(os.path.dirname(__file__), "vgg16_seed_model.h5")
model = tf.keras.models.load_model(model_path)

print("✅ Model loaded successfully")

# =========================
# WARM-UP (RUN ONCE)
# =========================
dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
model.predict(dummy)
print("✅ Warm-up done")

# =========================
# THREAD LOCK (IMPORTANT)
# =========================
lock = threading.Lock()

IMG_SIZE = 224

# =========================
# PREPROCESS (UNCHANGED)
# =========================
def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# =========================
# PREDICT (UNCHANGED LOGIC)
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

    # Resize once (keep this only here OR preprocess — keeping both is fine but this avoids extra load)
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
# RUN (RENDER SAFE)
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)