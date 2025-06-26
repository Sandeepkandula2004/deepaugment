import streamlit as st
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# Load the trained model
model = load_model("model.h5")

# Class labels
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Helper function to predict tumor type
def predict_tumor(image):
    IMAGE_SIZE = 128
    img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    if class_labels[predicted_class_index] == 'notumor':
        return "No Tumor", confidence_score
    else:
        return f"Tumor: {class_labels[predicted_class_index]}", confidence_score

# Streamlit UI
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")
st.title("🧠 Brain Tumor Classification")

uploaded_file = st.file_uploader("Upload a brain MRI image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI", width=300)  # Adjust width here

    result, confidence = predict_tumor(image)

    st.success(f"**Prediction:** {result}")
    st.info(f"**Confidence:** {confidence*100:.2f}%")

else:
    st.info("Please upload an image to proceed.")
