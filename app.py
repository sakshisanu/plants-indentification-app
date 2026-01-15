import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

st.title("🌿 Plant Identification App")

# Load model
model = load_model("keras_model.h5", compile=False)

# Load labels
class_names = open("labels.txt", "r").readlines()

uploaded_file = st.file_uploader("Upload a plant image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.success(f"🌱 Plant: {class_name}")
    st.info(f"Confidence: {confidence_score:.4f}")