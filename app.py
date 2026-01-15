import streamlit as st
import numpy as np
from PIL import Image, ImageOps

st.title("🌱 Plant Identification App")

uploaded_file = st.file_uploader("Upload a plant image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Resize
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Show image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to array
    image_array = np.asarray(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Fake prediction (for demo)
    st.subheader("Prediction")
    st.success("🌿 Plant identified successfully!")
    st.write("Plant: *Plumeria Alba*")
    st.write("Confidence: *92%*")





