import numpy as np
from PIL import Image

IMG_SIZE = 224

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = img.flatten().reshape(1, -1)
    return img
