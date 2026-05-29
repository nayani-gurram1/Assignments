import streamlit as st
import pandas as pd
import pickle
import os

# Load files
st.write("Files:", os.listdir())

model = pickle.load(open("fraud_rf.pkl", "rb"))
scaler = pickle.load(open("fraud_scaler.pkl", "rb"))
columns = pickle.load(open("fraud_columns.pkl", "rb"))

st.set_page_config(page_title="Fraud Detection", page_icon="💳")

st.title("💳 AI Credit Card Fraud Detection")
st.write("Predict whether a transaction is fraudulent")

st.sidebar.header("Transaction Details")

# Inputs
time = st.sidebar.number_input("Transaction Time", value=10000.0)
amount = st.sidebar.number_input("Transaction Amount", value=1000.0)

features = {}
for i in range(1, 29):
    features[f"V{i}"] = st.sidebar.number_input(f"V{i}", value=0.0)

if st.button("Detect Fraud"):

    input_data = {
        "Time": time,
        "Amount": amount
    }

    input_data.update(features)

    df = pd.DataFrame([input_data])

    # Match exact training column order
    df = df.reindex(columns=columns, fill_value=0)

    # Scale Amount
    df["Amount"] = scaler.transform(df[["Amount"]])

    probability = model.predict_proba(df)[0][1] * 100

    risk = "🔴 FRAUD DETECTED" if probability > 70 else "🟢 LEGITIMATE"

    st.subheader("Prediction Result")
    st.metric("Fraud Probability", f"{probability:.2f}%")
    st.success(f"Status: {risk}")