import streamlit as st
import numpy as np
import cv2
import pickle
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

# Set the directory for model files
model_dir = 'model'

# Load models
with open(os.path.join(model_dir, 'knn_classifier.pkl'), 'rb') as f:
    knn = pickle.load(f)
with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
    label_encoder = pickle.load(f)

# Load pre-trained InceptionResNetV2 model
facenet_model = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg')

# Function to get face embeddings from an image
def get_face_embedding(image):
    image = cv2.resize(image, (299, 299))  # Resize to the model's input size
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    embedding = facenet_model.predict(image)
    return embedding.flatten()

# Streamlit app interface
st.title("Who are you madafakah!!! 🫵🏻")
st.title("Poopy or Paia? 🤷🏻‍♀️")
st.write("Upload an image to recognize poopy or paia")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    if image is not None:
        # Display the uploaded image before processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption='Original Uploaded Image', use_column_width=True)

        # Show the spinner after displaying the image
        with st.spinner('Processing...'):
            # Get face embedding
            face_embedding = get_face_embedding(image)

            # Make prediction
            prediction = knn.predict([face_embedding])
            predicted_label = label_encoder.inverse_transform(prediction)[0]

            # Calculate confidence score based on the proportion of neighbors agreeing with the prediction
            distances, indices = knn.kneighbors([face_embedding], n_neighbors=knn.n_neighbors)
            neighbor_labels = label_encoder.inverse_transform(knn.predict([face_embedding for i in indices[0]]))
            predicted_neighbors = sum(neighbor == predicted_label for neighbor in neighbor_labels)
            confidence = (predicted_neighbors / knn.n_neighbors) * 100

            # Display the prediction and confidence
            if predicted_label == 'Prapty':
                st.write("Prediction: Bubblegum Babu 🐣")
            else:
                st.write(f"Prediction: {predicted_label}")

            st.write(f"Confidence: {confidence:.2f}%")
