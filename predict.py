import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# ✅ Load model (handle space in filename)
model = load_model('AI Model.h5')

# ✅ Correct class names
class_names = ['Haemorrhagic', 'Ischemic', 'Normal']

def preprocess_image(img_path):
    # Load image
    img = Image.open(img_path).convert("L")  # grayscale
    img = img.resize((128, 128))

    # Normalize
    img = np.array(img) / 255.0

    # 🔥 BiLSTM expects sequence input
    # Convert (128,128) → (128 timesteps, 128 features)
    img = img.reshape(128, 128)

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img

def predict_image(img_path):
    img = preprocess_image(img_path)

    pred = model.predict(img)
    pred_class = np.argmax(pred)
    confidence = np.max(pred)

    return class_names[pred_class], confidence
