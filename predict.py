import numpy as np
import joblib
from PIL import Image

# Load model
lgb_model = joblib.load('lgbm_model.joblib')

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

def preprocess_image(img_path):
    # Convert to grayscale (IMPORTANT)
    img = Image.open(img_path).convert("L")

    # Resize EXACTLY like training
    img = img.resize((128, 128))

    # Convert to array
    img = np.array(img)

    # Normalize (if you did during training)
    img = img / 255.0

    # Flatten (VERY IMPORTANT)
    img = img.flatten().reshape(1, -1)

    return img

def predict_image(img_path):
    img = preprocess_image(img_path)
    pred = lgb_model.predict(img)
    return class_names[int(pred[0])]
