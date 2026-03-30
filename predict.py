%%writefile predict.py
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, Model

# Load models
cnn_model = load_model('cnn_model.h5')
lgb_model = joblib.load('lgb_model.pkl')

# EfficientNet feature extractor
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = layers.GlobalAveragePooling2D()(base_model.output)
feature_extractor = Model(inputs=base_model.input, outputs=x)

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

def predict_image(img_array):
    # CNN prediction
    cnn_pred = cnn_model.predict(img_array)
    cnn_class = class_names[np.argmax(cnn_pred)]

    # Feature extraction
    features = feature_extractor.predict(img_array)

    # LightGBM prediction
    lgb_pred = lgb_model.predict(features)
    lgb_class = class_names[int(lgb_pred[0])]

    return cnn_class, lgb_class
