import os
import cv2
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ----------------------------
# Feature Extraction
# ----------------------------
def extract_features(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(
        clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    features = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h if h != 0 else 0
        features.append([area, aspect_ratio, h])

    return features

# ----------------------------
# Build Dataset
# ----------------------------
X, y = [], []

for file in os.listdir():
    if file.startswith("broken_grain"):
        feats = extract_features(file)
        X.extend(feats)
        y.extend([0] * len(feats))  # broken = 0

    elif file.startswith("full_grain"):
        feats = extract_features(file)
        X.extend(feats)
        y.extend([1] * len(feats))  # full = 1

X = np.array(X)
y = np.array(y)

print("Total samples:", len(X))

# ----------------------------
# Train Model
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Training accuracy:", accuracy)

# ----------------------------
# Save Model
# ----------------------------
joblib.dump(model, "rf_model.pkl")
print("Model saved as rf_model.pkl")
