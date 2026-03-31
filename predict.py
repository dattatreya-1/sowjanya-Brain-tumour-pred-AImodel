import numpy as np
import joblib
from PIL import Image

# Load model
lgb_model = joblib.load('lgbm_model.joblib')

# ✅ Correct class mapping
class_names = ['Haemorrhagic', 'Ischemic', 'Normal']

EXPECTED_FEATURES = lgb_model.n_features_in_  # should be 20480

def preprocess_image(img_path):
    # Convert to grayscale (same as training)
    img = Image.open(img_path).convert("L")

    # Resize EXACTLY to match training (20480 = 160×128)
    img = img.resize((160, 128))  # (width, height)

    # Normalize
    img = np.array(img) / 255.0

    # Flatten
    img = img.flatten()

    # Ensure correct feature size
    if img.shape[0] != EXPECTED_FEATURES:
        img = img[:EXPECTED_FEATURES]

    img = img.reshape(1, -1)

    return img

def predict_image(img_path):
    img = preprocess_image(img_path)
    pred = lgb_model.predict(img)
    return class_names[int(pred[0])]
