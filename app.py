import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

st.title("🌱 Plant Identification App")
#model=tf.keras.models.load_model("keras_model.h5")
with open("labels.txt","r") as f:class_names = f.read().splitlines()

uploaded_file = st.file_uploader(
    "Upload a plant image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    st.image(image, caption="Uploaded Image", use_column_width=True)
   





