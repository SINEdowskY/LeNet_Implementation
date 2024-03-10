import streamlit as st
from streamlit_drawable_canvas import st_canvas, st_image
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
import torch
from ModelEvaluator import Evaluator

st.header(
    "LeNet Implementation"
)

col1, col2 = st.columns(2)

with col1:
    canva = st_canvas(
        stroke_color="black",
        height=256,
        width=256
    )
    button_predict = st.button("Predict")
with col2:
    if canva.image_data is not None:
        image_data = canva.image_data
        image = Image.fromarray(canva.image_data)
        
        ev = Evaluator(image=image)
        st.write("stock image")
        st.image(ev.image, output_format="PNG")
        st.write(np.array(ev.image).shape)
        st.write("preprocessed image")
        st.image(ev.preprocessed_image)
        st.write(np.array(ev.preprocessed_image).shape)
        st.write("tensor image")
        st.image(ev.tensor_image_view)
        st.write(np.array(ev.tensor_image_view).shape)
        if button_predict:
            classes, probabilities = ev.predict()
            st.write(f"Result:{classes}")
            st.write(f"Probs:{probabilities}")


    
with col1:
    if button_predict:
        st.dataframe(
        data={
            "Classes":range(10),
            "Probabilities":np.array(probabilities).round(2)*100
        }
)