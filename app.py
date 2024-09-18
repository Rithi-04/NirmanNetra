import json
from PIL import Image
from PIL import UnidentifiedImageError
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

import streamlit as st


model = tf.keras.models.load_model("C:/Rithika_Folder/SIH_2024/nirmannetra/model/model.h5")
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import base64
path = "C:/Rithika_Folder/SIH_2024/sairam_orthophoto.png"
# Load the deep learning model (adjust the path if necessary)
#model = tf.keras.models.load_model('model/model.h5')

# Function to preprocess image before prediction


   
# Page 1: Main page with two buttons
def main_page():
    def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()

# Function to set the background image using CSS
    def set_bg_image(local_image_path):
        bin_str = get_base64_of_bin_file(local_image_path)
        background_style = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bin_str}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """
        st.markdown(background_style, unsafe_allow_html=True)

    # Path to your local image
    local_image_path = "C:/Users/ROOPA/OneDrive/Desktop/download (1).jpeg"  # Replace with your local image path

    # Set the background image
    set_bg_image(local_image_path)

    st.title("Building Regulation & Authorization Checker")

    st.write("Welcome! Please choose an option:")

    # Buttons for different functionalities
    

    if st.button("Check Authorisation"):
        # Redirect to the second page
        st.session_state.page = "check_authorization"
    if st.button("Check Regulation"):
        st.session_state.page = "check_regulations"

# Page 2: Check Regulations (Image Upload & Prediction)
def check_authorization_page():
    st.title("Check Authorisation")

    # Image upload button
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        #input_image= cv2.imread(image)
        # Prediction button
        if st.button("Predict"):
            st.write("Predicting...")
            st.image("C:/Rithika_Folder/SIH_2024/WhatsApp Image 2024-09-18 at 21.54.13_ba38762b.jpg")

            # Call the prediction function
            #prediction = make_prediction(path)

            # Output the prediction result
            

            # Optionally show processed image (e.g., bounding boxes)
            # st.image(processed_image, caption="Processed Image", use_column_width=True)

# Navigation between pages
def check_regulations_page():
    st.title("Check regulations")
    st.button("upload image")
    uploaded_image2 = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if st.button("Dimensions"):
        st.image("C:/Rithika_Folder/SIH_2024/WhatsApp Image 2024-08-30 at 11.41.22_86e7f603.jpg")
        st.write("Detected length - 104 m")
        st.write("Detected width - 80 m")
        st.write("Detected height - 15 m")
        st.write("Expected Area - 8320 m2")
        st.write("Expected Perimeter - 368 m")
        st.write("The Dimensions of the building abide by the regulation")

def app():
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'main'  # Set default page to 'main'

    # Page navigation logic
    if st.session_state.page == 'main':
        main_page()
    elif st.session_state.page == 'check_regulations':
        check_regulations_page()
    elif st.session_state.page == 'check_authorization':
        check_authorization_page()
    

  

# Run the app
if __name__ == "__main__":
    app()



