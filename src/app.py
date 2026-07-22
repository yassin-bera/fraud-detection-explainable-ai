
import streamlit as st
import joblib
import numpy as np


model = joblib.load("models/fraud_detection_xgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")


st.title("Fraud Detection with Explainable AI")

st.write(
    "Enter transaction features to predict fraud probability"
)


features = []

for i in range(30):
    value = st.number_input(
        f"Feature {i+1}",
        value=0.0
    )
    features.append(value)


if st.button("Predict"):

    data = np.array(features).reshape(1,-1)

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0]

    probability = model.predict_proba(data_scaled)[0][1]


    if prediction == 1:
        st.error("Fraudulent Transaction Detected")
    else:
        st.success("Normal Transaction")


    st.write(
        f"Fraud Probability: {probability:.4f}"
    )
