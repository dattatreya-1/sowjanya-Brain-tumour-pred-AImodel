import joblib
import numpy as np
from PIL import Image

# Load model
lgb_model = joblib.load('lightgbm_mri_classifier.pkl')

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

IMG_SIZE = 224

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = img.flatten().reshape(1, -1)
    return img

def predict_image(img_path):
    img = preprocess_image(img_path)
    pred = lgb_model.predict(img)
    return class_names[int(pred[0])]
