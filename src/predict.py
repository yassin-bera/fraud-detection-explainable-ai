
import joblib
import numpy as np


# Load model and scaler
model = joblib.load("fraud_detection_xgb_model.pkl")
scaler = joblib.load("scaler.pkl")


def predict_transaction(transaction):

    transaction = np.array(transaction).reshape(1, -1)

    transaction_scaled = scaler.transform(transaction)

    prediction = model.predict(transaction_scaled)[0]

    probability = model.predict_proba(transaction_scaled)[0][1]

    return {
        "prediction": int(prediction),
        "fraud_probability": float(probability)
    }
