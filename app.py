from flask import Flask, render_template, request
import cv2
import numpy as np
import joblib

app = Flask(__name__)

# -----------------------------
# Load SVM model and scaler
# -----------------------------
svm_model = joblib.load("svm_model.pkl")
scaler = joblib.load("svm_scaler.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    # -----------------------------
    # READ IMAGE
    # -----------------------------
    file = request.files["image"]
    image = cv2.imdecode(
        np.frombuffer(file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    # -----------------------------
    # PREPROCESSING
    # -----------------------------
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # -----------------------------
    # CONTOUR DETECTION
    # -----------------------------
    contours, _ = cv2.findContours(
        clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    total_grains = 0
    broken_grains = 0

    # -----------------------------
    # FEATURE EXTRACTION
    # -----------------------------
    feature_list = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Remove noise
        if area < 20:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h if h != 0 else 0

        feature_list.append([area, aspect_ratio, h])
        total_grains += 1

    if len(feature_list) > 0:
        features = np.array(feature_list)

        # -----------------------------
        # SCALE FEATURES (IMPORTANT)
        # -----------------------------
        features_scaled = scaler.transform(features)

        # -----------------------------
        # SVM PREDICTION
        # -----------------------------
        predictions = svm_model.predict(features_scaled)

        # broken = 0, full = 1
        broken_grains = int(np.sum(predictions == 0))

    # -----------------------------
    # SEND RESULTS TO UI
    # -----------------------------
    return render_template(
        "index.html",
        result={
            "total": total_grains,
            "broken": broken_grains,
            "accuracy": "74.6"
        }
    )

if __name__ == "__main__":
    app.run(debug=True)
