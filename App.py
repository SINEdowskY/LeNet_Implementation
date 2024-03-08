import streamlit as st
from streamlit_drawable_canvas import st_canvas, st_image
import numpy as np
from PIL import Image, ImageOps

st.header(
    "LeNet Implementation"
)

col1, col2 = st.columns(2)

with col1:
    canva = st_canvas(
        fill_color="#FFFFF",
        height=256,
        width=256
    )

with col2:
    if canva.image_data is not None:
        resized = Image.fromarray(canva.image_data)
        resized = resized.resize((28,28))
        img_array = np.array(resized)
        img_array = np.mean(img_array, axis=2, keepdims=True)
        img_array = img_array / 255.0
        img_array = np.mean(img_array, axis=2, keepdims=True)
        st.image(img_array)
        st.text(f"{img_array.shape}")