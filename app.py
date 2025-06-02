import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Load the model once
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('emotion_model.h5')
    return model

model = load_model()

# Emotion labels for FER-2013
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def preprocess_image(image: Image.Image) -> np.ndarray:
    # Convert to grayscale, resize to 48x48, scale pixels
    img = ImageOps.grayscale(image)
    img = img.resize((48, 48))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 48, 48, 1)  # batch, height, width, channels
    return img_array

st.title("FER-2013 Emotion Recognition")

uploaded_file = st.file_uploader("Upload a face image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess image
    processed_img = preprocess_image(image)
    
    # Predict emotion
    prediction = model.predict(processed_img)
    emotion_idx = np.argmax(prediction)
    confidence = prediction[0][emotion_idx]
    
    st.write(f"**Predicted emotion:** {emotion_labels[emotion_idx]}")
    st.write(f"**Confidence:** {confidence:.2f}")
