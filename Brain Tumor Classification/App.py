# Import necessary libraries
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import openai

# Set up the Streamlit page
st.title("Brain Tumor Classifier")
st.write("Upload a Brain MRI image, and our AI will classify it.")

# Load the pre-trained model
@st.cache_resource
def load_tumor_model():
    model = load_model('C:\\Users\\tanje\\Documents\\Brain Tumor Classification\\brain_tumor_model_vgg16.keras')
    return model

model = load_tumor_model()

# Define tumor class labels
class_labels = ['Pituitary', 'No Tumor', 'Meningioma', 'Glioma']

# Image preprocessing function
def preprocess_image(image):
    image = image.resize((150, 150))  # Resize image for the model input shape
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Prediction function
def predict_tumor(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction)  # Get index of highest confidence
    predicted_class = class_labels[predicted_class_index]
    confidence = prediction[0][predicted_class_index] * 100
    return predicted_class, confidence


# Upload an image
uploaded_image = st.file_uploader("Upload Brain MRI Image", type=["png", "jpg", "jpeg"])

if uploaded_image:
    # Display uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # Classify tumor when the button is clicked
    if st.button("Classify Tumor"):
        predicted_class, confidence = predict_tumor(image)
        st.write(f"**Prediction:** {predicted_class} ({confidence:.2f}% confidence)")

        # User prompt input
        user_prompt = st.text_input("Enter your question or ask for an explanation:")

        # Generate explanation based on prediction and user prompt
        if user_prompt:
            explanation = generate_explanation(predicted_class, user_prompt)
            st.write("### Explanation:")
            st.write(explanation)
