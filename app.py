import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import os

# Load the trained model
MODEL_PATH = r"D:\prithivi main project\multi_class_DR_model.h5"

# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

model = load_model(MODEL_PATH)

# Define class labels with full names
CLASS_NAMES = [
    "No Diabetic Retinopathy",
    "Mild Non-Proliferative Diabetic Retinopathy (NPDR)",
    "Moderate Non-Proliferative Diabetic Retinopathy (NPDR)",
    "Severe Non-Proliferative Diabetic Retinopathy (NPDR)",
    "Proliferative Diabetic Retinopathy (PDR)"
]

# Treatment suggestions for each class
TREATMENTS = {
    "No Diabetic Retinopathy": (
        "✅ **Lifestyle & Prevention:**\n"
        "- Maintain good blood sugar control (A1C < 7%).\n"
        "- Regular eye check-ups every 1-2 years.\n"
        "- Healthy diet (low sugar, high fiber) and regular exercise.\n"
        "- Stop smoking and limit alcohol consumption.\n"
        "- Monitor blood pressure and cholesterol.\n"
    ),
    "Mild Non-Proliferative Diabetic Retinopathy (NPDR)": (
        "✅ **Lifestyle & Monitoring:**\n"
        "- Control blood sugar, blood pressure, and cholesterol.\n"
        "- Regular eye exams every 6-12 months.\n"
        "- No immediate treatment but close monitoring.\n"
        "- Manage diabetes with diet, exercise, and medications.\n"
        "⚠️ **Potential Early Interventions:**\n"
        "- If macular edema is suspected, anti-VEGF therapy may be considered.\n"
        "- Discuss with an ophthalmologist for early signs of progression.\n"
    ),
    "Moderate Non-Proliferative Diabetic Retinopathy (NPDR)": (
        "⚠️ **More Frequent Monitoring & Medical Therapy:**\n"
        "- Eye exams every 3-6 months to track progression.\n"
        "- Intensified blood sugar and blood pressure control.\n"
        "- Lifestyle modifications (healthy eating, exercise, quitting smoking).\n"
        "⚠️ **Medical & Laser Therapy Considerations:**\n"
        "- Anti-VEGF injections (e.g., Ranibizumab, Bevacizumab) if macular edema is present.\n"
        "- Consider focal/grid laser photocoagulation if swelling affects vision.\n"
    ),
    "Severe Non-Proliferative Diabetic Retinopathy (NPDR)": (
        "🚨 **High Risk of Vision Loss – Urgent Intervention Needed!**\n"
        "⚠️ **Monitoring & Medications:**\n"
        "- Frequent monitoring every 1-3 months.\n"
        "- Anti-VEGF therapy if macular edema is present.\n"
        "- Intensive diabetes management (HbA1c <7%, strict BP control).\n"
        "⚠️ **Advanced Treatment Options:**\n"
        "- **Laser Photocoagulation:** Panretinal photocoagulation (PRP) to prevent new blood vessel growth.\n"
        "- **Corticosteroids:** Intravitreal injections (e.g., Dexamethasone implant) for macular edema.\n"
    ),
    "Proliferative Diabetic Retinopathy (PDR)": (
        "🚨 **URGENT MEDICAL ATTENTION REQUIRED!**\n"
        "🔴 **Immediate Interventions to Prevent Blindness:**\n"
        "- **Laser Therapy (Panretinal Photocoagulation - PRP):** Shrinks abnormal blood vessels.\n"
        "- **Intravitreal Anti-VEGF Injections:** Prevents new blood vessel growth (e.g., Ranibizumab, Aflibercept).\n"
        "- **Vitrectomy Surgery:** Removes blood and scar tissue from the retina if there’s severe bleeding.\n"
        "- **Steroid Injections:** Used in some cases to reduce swelling.\n"
        "🛑 **Preventive Measures & Long-Term Care:**\n"
        "- Tight glucose, BP, and lipid control to prevent worsening.\n"
        "- Close follow-up with a retina specialist.\n"
    )
}

# Streamlit app title and description
st.title("🩺 Diabetic Retinopathy Classification & Treatment")
st.write("Upload an eye fundus image to classify its severity and get treatment suggestions.")

# File uploader for image upload
uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="📌 Uploaded Image", use_column_width=True)
    st.write("🔄 **Processing the image...**")

    try:
        # Preprocess the uploaded image
        img = load_img(uploaded_file, target_size=(150, 150))  # Resize image
        img_array = img_to_array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make a prediction
        predictions = model.predict(img_array)

        if predictions.shape[1] != len(CLASS_NAMES):
            st.error("❌ Mismatch between model output and class labels. Please check the model.")
            st.stop()

        predicted_class_idx = np.argmax(predictions)  # Get class index with highest probability
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])  # Confidence score

        # Display the prediction results
        st.success(f"**Predicted Class:** {predicted_class} ✅")
        st.write(f"**Confidence:** {confidence:.2%}")

        # Display treatment suggestions
        st.markdown("## 📋 Recommended Treatment Plan")
        st.info(TREATMENTS[predicted_class])

        # Display full probability scores for each class
        st.write("### 📊 Class Probabilities:")
        for i, class_name in enumerate(CLASS_NAMES):
            st.write(f"**{class_name}:** {float(predictions[0][i]):.2%}")

    except Exception as e:
        st.error(f"❌ An error occurred while processing the image: {e}")
