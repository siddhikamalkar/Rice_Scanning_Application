import os
import cv2
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

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
        y.extend([0] * len(feats))   # broken

    elif file.startswith("full_grain"):
        feats = extract_features(file)
        X.extend(feats)
        y.extend([1] * len(feats))   # full

X = np.array(X)
y = np.array(y)

print("Total samples:", len(X))

# ----------------------------
# Train/Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# Feature Scaling (VERY IMPORTANT FOR SVM)
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# Train SVM
# ----------------------------
svm_model = SVC(kernel="rbf", C=1.0, gamma="scale")
svm_model.fit(X_train_scaled, y_train)

accuracy = svm_model.score(X_test_scaled, y_test)
print("SVM Test Accuracy:", accuracy)

# ----------------------------
# Save Model + Scaler
# ----------------------------
joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(scaler, "svm_scaler.pkl")

print("SVM model saved as svm_model.pkl")
