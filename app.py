import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Load the model and scaler
model = joblib.load('gbmreg_model.pkl')
scaler = joblib.load('scaler.pkl')

# 2. Set up the Streamlit UI
st.set_page_config(page_title="Insurance Charges Predictor", layout="centered")
st.title("Health Insurance Charges Prediction")
st.write("Enter details below to estimate medical insurance charges based on patient demographics and history.")

# 3. Create input containers
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        claim_amount = st.number_input("Claim Amount (₹)", min_value=0.0, value=30000.0, step=500.0)
        past_consultations = st.number_input("Past Consultations Count", min_value=0, value=15, step=1)
        hospital_expenditure = st.number_input("Hospital Expenditure (₹)", min_value=0.0, value=5000000.0, step=10000.0)

    with col2:
        annual_salary = st.number_input("Annual Salary (₹)", min_value=0.0, value=50000000.0, step=100000.0)
        children = st.slider("Number of Dependent Children", 0, 5, 0)
        smoker = st.selectbox("Smoker Status", options=["No", "Yes"])

    submit_button = st.form_submit_button(label="Calculate Charges")

# 4. Processing and Prediction
if submit_button:
    # Convert smoker status back to binary encoding used during training
    smoker_val = 1 if smoker == "Yes" else 0
    
    # Create a dataframe for the input
    input_dict = {
        'claim_amount': [claim_amount],
        'past_consultations': [past_consultations],
        'hospital_expenditure': [hospital_expenditure],
        'annual_salary': [annual_salary],
        'children': [children],
        'smoker': [smoker_val]
    }
    input_df = pd.DataFrame(input_dict)

    # Apply scaling using the loaded scaler
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)

    # Display Result
    st.markdown("---宣")
    st.subheader(f"Estimated Insurance Charges: ₹ {prediction[0]:,.2f}")
    st.info("Note: This is an AI-generated estimate based on historical data patterns.")
