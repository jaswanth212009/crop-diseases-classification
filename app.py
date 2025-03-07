import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Title of the app
st.title("Crop Disease Classifier :)")

# Instructions
st.write("Select a crop and upload an image to detect diseases.")

# Dropdown for crop selection
crop_options = ["Sorghum", "SugarCane", "Wheat"]  # Add more crops as needed
selected_crop = st.selectbox("Select Crop", crop_options)

# Define model paths and disease labels for each crop
model_configs = {
    "Sorghum": {
        "path": "sorghum_disease_model.pt",
        "labels": [
            "Anthracnose and Red Rot",
            "Cereal Grain molds",
            "Covered Kernel smut",
            "Head Smut",
            "Rust",
            "Loose smut"
        ]
    },
    "SugarCane": {
        "path": "sugarcane_disease_model.pt",  # Update this path
        "labels": [
            "healthy",
            "mosaic",
            "red_rot", 
            "rust", 
            "yellow_leaf"
        ]
    },
    "Wheat": {
        "path": "wheat_disease_model.pt",
        "labels": [
            "root rot",
            "leaf rust",
            "stem rust",
            "mildew",
            "lead blotch",
            "stripe rust",
            "fusarium head blight",
            "loose smut",
            "aphids"
        ]
    }, 
 
}

# Load the selected model
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# Load the model and labels based on the selected crop
model = load_model(model_configs[selected_crop]["path"])
disease_labels = model_configs[selected_crop]["labels"]

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Function to preprocess the image (resize longest side to 640, preserve aspect ratio)
def preprocess_image(image):
    img = np.array(image)
    height, width = img.shape[:2]
    longest_side = max(height, width)
    scale = 640 / longest_side
    new_height = int(height * scale)
    new_width = int(width * scale)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif img.shape[-1] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

# Process and predict
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption=f"Uploaded {selected_crop} Image", use_column_width=True)

    # Preprocess and predict
    processed_image = preprocess_image(image)
    results = model(processed_image)
    prediction = results[0]

    # Extract class and confidence
    predicted_class = prediction.probs.top1
    confidence = prediction.probs.top1conf.item() * 100

    # Display result
    st.subheader("Prediction")
    st.write(f"Predicted Disease: **{disease_labels[predicted_class]}**")
    st.write(f"Confidence: **{confidence:.2f}%**")
else:
    st.write(f"Please upload a {selected_crop} image to get a prediction.")