import streamlit as st
import pandas as pd
import pickle
import os

# Load files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "random_forest.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
columns = pickle.load(open(os.path.join(BASE_DIR, "columns.pkl"), "rb"))

st.title("AI-Powered Employee Attrition Prediction")

# Inputs
age = st.number_input("Age", 18, 60, 35)
income = st.number_input("Monthly Income", 1000, 50000, 4500)
job_satisfaction = st.slider("Job Satisfaction", 1, 4, 2)

if st.button("Predict Attrition Risk"):

    # Create full feature dataframe with zeros
    input_data = pd.DataFrame(0, index=[0], columns=columns)

    # Fill known features
    if "Age" in input_data.columns:
        input_data["Age"] = age

    if "MonthlyIncome" in input_data.columns:
        input_data["MonthlyIncome"] = income

    if "JobSatisfaction" in input_data.columns:
        input_data["JobSatisfaction"] = job_satisfaction

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prob = model.predict_proba(input_scaled)[0][1] * 100

    if prob > 70:
        risk = "HIGH"
    elif prob > 40:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    st.success(f"Attrition Probability: {prob:.2f}%")
    st.warning(f"Risk Level: {risk}")
