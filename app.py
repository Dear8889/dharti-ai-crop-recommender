import streamlit as st
import pandas as pd
import joblib

# 1. Load the AI model we trained yesterday
# We use try-except just in case the file isn't in the exact right spot
try:
    model = joblib.load('crop_model.pkl')
except FileNotFoundError:
    st.error("Model not found! Make sure 'crop_model.pkl' is in the same folder.")

# 2. Set up the UI Headers
st.set_page_config(page_title="DHARTI AI: Crop Recommender", page_icon="🌱")
st.title("🌱 DHARTI AI: Climate-Resilient Crop Recommender")
st.markdown("Enter the local soil metrics and weather data below to get an AI-driven crop recommendation for marginal farmers.")

# 3. Create the input sliders for the user
st.sidebar.header("Input Farm Data")

# We use sliders with realistic ranges for these agricultural metrics
N = st.sidebar.slider("Nitrogen (N)", 0, 140, 50)
P = st.sidebar.slider("Phosphorus (P)", 0, 145, 50)
K = st.sidebar.slider("Potassium (K)", 0, 205, 50)
temperature = st.sidebar.slider("Temperature (°C)", 5.0, 50.0, 25.0)
humidity = st.sidebar.slider("Humidity (%)", 10.0, 100.0, 60.0)
ph = st.sidebar.slider("Soil pH", 3.0, 10.0, 6.5)
rainfall = st.sidebar.slider("Rainfall (mm)", 20.0, 300.0, 100.0)

# 4. The Prediction Engine
if st.button("Predict Best Crop"):
    # Organize the user inputs into the exact format the AI expects
    user_input = pd.DataFrame({
        'N': [N],
        'P': [P],
        'K': [K],
        'temperature': [temperature],
        'humidity': [humidity],
        'ph': [ph],
        'rainfall': [rainfall]
    })
    
    # Ask the AI to predict
    prediction = model.predict(user_input)
    
    # Display the result to the user!
    st.success(f"🌾 **Recommended Crop to Plant:** {prediction[0].upper()}")
    st.balloons() # A little Streamlit magic for the presentation
    