import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(page_title="Plant Identification App 🌿")
st.title("🌿 Plant Identification App")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("keras_model.h5", compile=False)

model = load_model()

class_names = [line.strip() for line in open("labels.txt", "r")]

uploaded_file = st.file_uploader("Upload a plant image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image).astype(np.float32)
    normalized_image_array = (image_array / 127.5) - 1

    data = np.expand_dims(normalized_image_array, axis=0)

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    st.success(f"🌱 Prediction: {class_name}")
    st.info(f"Confidence: {confidence_score:.2f}"












