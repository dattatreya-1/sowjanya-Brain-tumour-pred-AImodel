import numpy as np
import joblib
from PIL import Image

# Load model
lgb_model = joblib.load('lgbm_model.joblib')

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

EXPECTED_FEATURES = lgb_model.n_features_in_  # 20480

def preprocess_image(img_path):
    img = Image.open(img_path).convert("L")

    # 🔥 Adjust size to match feature count
    # 20480 = 128 × 160
    img = img.resize((160, 128))   # WIDTH, HEIGHT

    img = np.array(img) / 255.0

    # Flatten
    img = img.flatten()

    # Safety check
    if img.shape[0] != EXPECTED_FEATURES:
        img = img[:EXPECTED_FEATURES]

    img = img.reshape(1, -1)

    return img

def predict_image(img_path):
    img = preprocess_image(img_path)
    pred = lgb_model.predict(img)
    return class_names[int(pred[0])]
