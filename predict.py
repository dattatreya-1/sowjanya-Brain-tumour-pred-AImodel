import numpy as np
import joblib
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, Model

# Load LightGBM model
lgb_model = joblib.load('lgbm_model.joblib')

# Load EfficientNet
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = layers.GlobalAveragePooling2D()(base_model.output)
feature_extractor = Model(inputs=base_model.input, outputs=x)

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

    # Extract features
    features = feature_extractor.predict(img)

    # Predict using LightGBM
    pred = lgb_model.predict(features)

    return class_names[int(pred[0])]
