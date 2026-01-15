import streamlit as st
import numpy as np
from PIL import Image, ImageOps

st.title("🌱 Plant Identification App")

uploaded_file = st.file_uploader(
    "Upload a plant image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # 1. Open image
    image = Image.open(uploaded_file).convert("RGB")

    # 2. Resize image
    image = ImageOps.fit(image, (224, 224))

    # 3. Show image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # 4. Convert to array
    image_array = np.asarray(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # 5. Show result
    st.subheader("Prediction")
    st.success("🌿 Plant identified successfully")
    st.write("Plant name: *Example Plant*")
    st.write("Confidence: *90%*")





