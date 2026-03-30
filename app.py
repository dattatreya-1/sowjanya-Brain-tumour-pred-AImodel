
import streamlit as st
from preprocess import preprocess_image
from predict import predict_image
from PIL import Image
import tempfile

st.title("Brain Tumor Detection App")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save temp image
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        image.save(tmp.name)
        img_array = preprocess_image(tmp.name)

    cnn_class, lgb_class = predict_image(img_array)

    st.write(f"### CNN Prediction: {cnn_class}")
    st.write(f"### LightGBM Prediction: {lgb_class}")
