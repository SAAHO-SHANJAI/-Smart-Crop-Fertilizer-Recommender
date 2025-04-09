import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle

# ✅ Set page config FIRST
st.set_page_config(page_title="Crop & Fertilizer Recommender", layout="centered")

# ✅ Neon water background effect
def set_neon_water_background():
    st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(circle at center, #001f3f, #001a35, #001226);
        background-size: 400% 400%;
        animation: neonMove 15s ease infinite;
    }

    @keyframes neonMove {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    body {
        margin: 0;
        overflow-x: hidden;
    }

    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background: radial-gradient(circle at center, rgba(0,255,255,0.1), transparent 70%);
        pointer-events: none;
        transition: background-position 0.1s ease;
    }
    </style>

    <script>
    document.addEventListener('mousemove', function(e) {
        const x = e.clientX / window.innerWidth * 100;
        const y = e.clientY / window.innerHeight * 100;
        document.documentElement.style.setProperty('--mouse-x', `${x}%`);
        document.documentElement.style.setProperty('--mouse-y', `${y}%`);
    });
    </script>
    """, unsafe_allow_html=True)

# 🔥 Set the fancy background
set_neon_water_background()

# ✅ Load models and encoders
try:
    with open('model/fertilizer_model.pkl', 'rb') as f:
        fert_model = pickle.load(f)
    with open('model/soil_encoder.pkl', 'rb') as f:
        soil_encoder = pickle.load(f)
    with open('model/crop_encoder.pkl', 'rb') as f:
        crop_encoder = pickle.load(f)
    with open('model/fert_encoder.pkl', 'rb') as f:
        fert_encoder = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load model or encoders: {e}")
    st.stop()

# ✅ App layout and title
st.title("🌿 Smart Crop & Fertilizer Recommendation System")

# ✅ Input section
st.header("📥 Enter Field Parameters")
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        temp = st.number_input("🌡️ Temperature (°C)", 0.0, 60.0, step=0.1)
        humidity = st.number_input("💧 Humidity (%)", 0.0, 100.0, step=0.1)
        moisture = st.number_input("🌊 Moisture (%)", 0.0, 100.0, step=0.1)
        nitrogen = st.number_input("🧪 Nitrogen (N)", 0, 100)

    with col2:
        potassium = st.number_input("🧪 Potassium (K)", 0, 100)
        phosphorus = st.number_input("🧪 Phosphorus (P)", 0, 100)
        soil_type = st.selectbox("🧱 Soil Type", soil_encoder.classes_)
        crop_type = st.selectbox("🌱 Crop Type", crop_encoder.classes_)

    uploaded_image = st.file_uploader("🖼️ Upload Plant Image (Optional)", type=["jpg", "jpeg", "png"])
    submit = st.form_submit_button("🔍 Get Recommendations")

# ✅ Prediction
if submit:
    try:
        # Transform inputs
        input_data = pd.DataFrame([[
            temp,
            humidity,
            moisture,
            soil_encoder.transform([soil_type])[0],
            crop_encoder.transform([crop_type])[0],
            nitrogen,
            potassium,
            phosphorus
        ]], columns=['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous'])

        # Predict fertilizer
        fert_pred = fert_model.predict(input_data)[0]
        fert_name = fert_encoder.inverse_transform([fert_pred])[0]

        # Output
        st.success(f"🌾 Recommended Fertilizer: **{fert_name}** for {crop_type} in {soil_type} soil.")

        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Plant Image", use_container_width=True)
            st.info("🧠 Plant disease detection feature coming soon!")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
