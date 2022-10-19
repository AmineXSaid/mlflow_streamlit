import streamlit as st
from tensorflow.keras.utils import load_img
from PIL import Image
import numpy as np
from tensorflow import keras

app_mode = st.sidebar.selectbox('Select Page',['Home','Prediction'])

if app_mode=="Home":
    st.title("IS IT A CAR OR A PLANE ?")
    st.image(Image.open(r"web_app\carvsplane.jpg"))

elif app_mode=="Prediction":
    model=keras.models.load_model(r"Model1")
    image=st.file_uploader("Upload Your photo")
    if image is not None :
        image = load_img(image, target_size=(224, 224))
        img = np.array(image)
        img = img / 255.0
        img = img.reshape(1,224,224,3)
        label = model.predict(img)
        if (label[0][0]<0.7) and (label[0][1]<0.7) :
            st.error("This is NOT a car NOR a plane")
        elif label[0][0]<label[0][1]  :
            st.success("Predicted Class Planes")
        else :
            st.success("Predicted Class Cars")  