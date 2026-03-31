import numpy as np
import joblib
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load model
lgb_model = joblib.load('lgbm_model.joblib')

# Load EfficientNet WITHOUT pooling
feature_extractor = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3))

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224,224))
    img = np.array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(img_path):
    img = preprocess_image(img_path)

    features = feature_extractor.predict(img)

    # Flatten SAME as training
    features = features.reshape(1, -1)

    pred = lgb_model.predict(features)

    return class_names[int(pred[0])]
