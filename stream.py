import streamlit as st
import pandas as pd
import pickle


with open('weather_predictor.pkl', 'rb') as f:
    model = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

with open('label_encoder_cloud.pkl', 'rb') as f:
    label_encoder_cloud = pickle.load(f)

with open('label_encoder_location.pkl', 'rb') as f:
    label_encoder_location = pickle.load(f)

# ... (rest of your code remains the same))

st.title("Weather Type Prediction")
temperature = st.number_input('Temperature (°C)', min_value=-50, max_value=50, value=20)
humidity = st.number_input('Humidity (%)', min_value=0, max_value=100, value=50)
wind_speed = st.number_input('Wind Speed (m/s)', min_value=0.0, max_value=50.0, value=5.0)
precipitation = st.number_input('Precipitation (%)', min_value=0.0, max_value=100.0, value=10.0)
cloud_cover = st.selectbox('Cloud Cover', label_encoder_cloud.classes_)
atmospheric_pressure = st.number_input('Atmospheric Pressure (hPa)', min_value=900.0, max_value=1100.0, value=1013.0)
uv_index = st.number_input('UV Index', min_value=0, max_value=11, value=5)
visibility = st.number_input('Visibility (km)', min_value=0.0, max_value=50.0, value=10.0)
location = st.selectbox('Location', label_encoder_location.classes_)
cloud_cover_encoded = label_encoder_cloud.transform([cloud_cover])[0]
location_encoded = label_encoder_location.transform([location])[0]
input_data = pd.DataFrame({
    'Temperature': [temperature],
    'Humidity': [humidity],
    'Wind Speed': [wind_speed],
    'Precipitation (%)': [precipitation],
    'Cloud Cover Encoded': [cloud_cover_encoded],
    'Atmospheric Pressure': [atmospheric_pressure],
    'UV Index': [uv_index],
    'Visibility (km)': [visibility],
    'Location Encoded': [location_encoded]
})
input_data = input_data[feature_names]
if st.button('Predict Weather Type'):
    prediction = model.predict(input_data)
    st.write(f"The predicted weather type is: {prediction[0]}")