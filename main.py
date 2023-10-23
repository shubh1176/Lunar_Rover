import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the pre-trained model
model = tf.keras.models.load_model("model/final_model.h5")  

# Define some constants
IMAGE_SIZE = (256, 256)

# Function to preprocess the user's image
def preprocess_image(image):
    image = cv2.resize(image, IMAGE_SIZE)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to make predictions
def make_prediction(image):
    image = preprocess_image(image)
    prediction = model.predict(image)
    return prediction

# Streamlit app
st.title("Lunar Surface Obstacle Detection")
st.write("Upload image of lunar surface")

# File uploader
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Check if the user clicked the "Segment" button
    if st.button("Segment"):
        # Process and make a prediction
        image = np.array(image)
        prediction = make_prediction(image)

        # Display the segmented image
        st.image(prediction[0], caption="", use_column_width=True)

