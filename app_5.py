import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page setup
st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="centered")

st.title("🍷 Wine Quality Prediction Dashboard")
st.markdown("Enter the exact chemical laboratory metrics to evaluate the wine profile.")

# 1. Load your saved pipeline artifacts safely
@st.cache_resource
def load_model_artifacts():
    with open("WineQualityPrediction.pkl", "rb") as f:
        return pickle.load(f)

try:
    artifacts = load_model_artifacts()
    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]
    scaler = artifacts["scaler"]
except FileNotFoundError:
    st.error("❌ 'WineQualityPrediction.pkl' not found. Run your main training script first!")
    st.stop()

st.subheader("🧪 Chemical Laboratory Metrics")

# 2. Using two columns with number inputs instead of glitchy sliders
col1, col2 = st.columns(2)

with col1:
    alcohol = st.number_input("Alcohol (% by volume)", min_value=8.0, max_value=15.0, value=10.5, step=0.1)
    sulphates = st.number_input("Sulphates (g/dm³)", min_value=0.3, max_value=2.0, value=0.6, step=0.01)
    volatile_acidity = st.number_input("Volatile Acidity (g/dm³)", min_value=0.1, max_value=1.6, value=0.5, step=0.01)
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide (mg/dm³)", min_value=6.0, max_value=289.0, value=46.0, step=1.0)
    fixed_acidity = st.number_input("Fixed Acidity (g/dm³)", min_value=4.0, max_value=16.0, value=8.3, step=0.1)
    pH = st.number_input("pH Level", min_value=2.7, max_value=4.0, value=3.3, step=0.01)

with col2:
    chlorides = st.number_input("Chlorides (g/dm³)", min_value=0.01, max_value=0.6, value=0.08, step=0.001, format="%.3f")
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide (mg/dm³)", min_value=1.0, max_value=72.0, value=15.0, step=1.0)
    citric_acid = st.number_input("Citric Acid (g/dm³)", min_value=0.0, max_value=1.0, value=0.27, step=0.01)
    residual_sugar = st.number_input("Residual Sugar (g/dm³)", min_value=0.9, max_value=15.5, value=2.5, step=0.1)
    density = st.number_input("Density (g/cm³)", min_value=0.9900, max_value=1.0040, value=0.9968, step=0.0001, format="%.4f")

# Create structure matching original DataFrame columns
input_data = pd.DataFrame([{
    "fixed acidity": fixed_acidity,
    "volatile acidity": volatile_acidity,
    "citric acid": citric_acid,
    "residual sugar": residual_sugar,
    "chlorides": chlorides,
    "free sulfur dioxide": free_sulfur_dioxide,
    "total sulfur dioxide": total_sulfur_dioxide,
    "density": density,
    "pH": pH,
    "sulphates": sulphates,
    "alcohol": alcohol
}])

st.markdown("---")

# 3. Handle model logic on button trigger
if st.button("🚀 Evaluate Wine Quality Profile", use_container_width=True):
    processed_input = preprocessor.transform(input_data)
    scaled_input = scaler.transform(processed_input)
    
    prediction = model.predict(scaled_input)
    probabilities = model.predict_proba(scaled_input)
    confidence = probabilities[0][prediction[0]] * 100

    if prediction[0] == 1:
        st.success("### 🌟 Premium Quality Wine Predicted!")
        st.metric(label="Model Confidence Score", value=f"{confidence:.2f}%")
    else:
        st.warning("### 🪵 Standard/Low Quality Wine Predicted")
        st.metric(label="Model Confidence Score", value=f"{confidence:.2f}%")
