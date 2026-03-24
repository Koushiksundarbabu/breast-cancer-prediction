import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.title("Breast Cancer Prediction System")

st.write(
    "Enter tumor measurements below to predict whether it is malignant or benign."
)

st.markdown("---")

# Input section
st.subheader("📊 Enter Tumor Features")

col1, col2 = st.columns(2)

with col1:
    radius_worst = st.number_input("Radius Worst", value=10.0)
    perimeter_worst = st.number_input("Perimeter Worst", value=50.0)
    area_worst = st.number_input("Area Worst", value=500.0)

with col2:
    concave_points_worst = st.number_input("Concave Points Worst", value=0.1)
    radius_mean = st.number_input("Radius Mean", value=10.0)
    texture_mean = st.number_input("Texture Mean", value=20.0)
    concavity_mean = st.number_input("Concavity Mean", value=0.1)

st.markdown("---")

# Prediction
if st.button("🔍 Predict"):

    input_data = np.array([[
        radius_worst,
        perimeter_worst,
        area_worst,
        concave_points_worst,
        radius_mean,
        texture_mean,
        concavity_mean
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("📢 Prediction Result")

    if prediction[0] == 1:
        st.error(f"⚠️ Malignant (Cancer) - Risk: {probability:.2%}")
    else:
        st.success(f"✅ Benign (No Cancer) - Risk: {probability:.2%}")