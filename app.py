import streamlit as st
from PIL import Image, ImageOps

st.title("🌿 Plant Identification App")

uploaded_file = st.file_uploader("Upload a plant image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
st.image(image, caption="Uploaded Image", use_column_width=True)
   


