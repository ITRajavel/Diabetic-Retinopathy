import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, Flatten, Dense, Dropout, BatchNormalization, ReLU, MaxPooling2D
from tensorflow.keras.utils import load_img, img_to_array
import os

# Ensure TensorFlow runs on CPU if GPU/CUDA is unavailable
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    st.success("✅ CUDA detected — running on GPU!")
else:
    st.warning("")

# Define a compatible model architecture
MODEL_PATH = "diabetic_retinopathy_model_v2.h5"

if not os.path.exists(MODEL_PATH):
    st.warning("")

    model = Sequential([
        DepthwiseConv2D((3, 3), depth_multiplier=1, padding='same', input_shape=(150, 150, 3)),
        BatchNormalization(),
        ReLU(),
        MaxPooling2D((2, 2)),

        DepthwiseConv2D((3, 3), depth_multiplier=1, padding='same'),
        BatchNormalization(),
        ReLU(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(5, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    st.success("✨ Model architecture built successfully!")
else:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    st.success("✅ Model loaded successfully!")

# Define class labels and treatments
CLASS_NAMES = [
    "No Diabetic Retinopathy",
    "Mild Non-Proliferative Diabetic Retinopathy (NPDR)",
    "Moderate Non-Proliferative Diabetic Retinopathy (NPDR)",
    "Severe Non-Proliferative Diabetic Retinopathy (NPDR)",
    "Proliferative Diabetic Retinopathy (PDR)"
]

TREATMENTS = {
    "No Diabetic Retinopathy": "Maintain a healthy lifestyle and regular check-ups.",
    "Mild Non-Proliferative Diabetic Retinopathy (NPDR)": "Monitor closely and control diabetes.",
    "Moderate Non-Proliferative Diabetic Retinopathy (NPDR)": "Frequent monitoring and possible medical therapy.",
    "Severe Non-Proliferative Diabetic Retinopathy (NPDR)": "Urgent intervention with potential laser therapy.",
    "Proliferative Diabetic Retinopathy (PDR)": "Immediate medical attention required, potential surgery."
}

# Streamlit app setup
st.title("🩺 Diabetic Retinopathy Classification & Treatment")
st.write("Upload an eye fundus image to classify its severity and get treatment suggestions.")

# File uploader
uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="📌 Uploaded Image", use_column_width=True)
    st.write("🔄 **Processing the image...**")

    try:
        # Preprocess the image
        img = load_img(uploaded_file, target_size=(150, 150))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = model.predict(img_array)

        if predictions.shape[1] != len(CLASS_NAMES):
            st.error("❌ Mismatch between model output and class labels. Please check the model.")
            st.stop()

        # Get predicted class and confidence
        predicted_class_idx = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])

        # Display prediction
        st.success(f"**Predicted Class:** {predicted_class} ✅")
        st.write(f"**Confidence:** {confidence:.2%}")

        # Display treatment suggestion
        st.markdown("## 📋 Recommended Treatment Plan")
        st.info(TREATMENTS[predicted_class])

        # Show class probabilities as a bar chart
        st.write("### 📊 Class Probabilities:")
        st.bar_chart({CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))})

    except Exception as e:
        st.error(f"❌ An error occurred while processing the image: {e}")

# Optional footer
st.markdown("---")
st.write("🔍 **Note:** This app is for educational purposes only. Always consult a medical professional for accurate diagnosis.")