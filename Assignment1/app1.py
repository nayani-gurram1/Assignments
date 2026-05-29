import streamlit as st
import pandas as pd
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "random_forest.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
columns = pickle.load(open(os.path.join(BASE_DIR, "columns.pkl"), "rb"))

st.set_page_config(page_title="Employee Attrition Predictor", layout="centered")

st.title("🚀 Employee Attrition Prediction System")
st.write("Predict whether an employee will leave the company")

# ---------------- INPUT SECTION ----------------

age = st.number_input("Age", 18, 60, 30)
distance = st.number_input("Distance From Home", 1, 30, 5)
income = st.number_input("Monthly Income", 1000, 20000, 5000)
years = st.number_input("Years at Company", 0, 40, 3)
job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 2)
work_life = st.slider("Work Life Balance (1-4)", 1, 4, 2)
overtime = st.selectbox("OverTime", ["Yes", "No"])
gender = st.selectbox("Gender", ["Male", "Female"])
marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

# ---------------- PREPROCESS INPUT ----------------

input_dict = {
    "Age": age,
    "DistanceFromHome": distance,
    "MonthlyIncome": income,
    "YearsAtCompany": years,
    "JobSatisfaction": job_satisfaction,
    "WorkLifeBalance": work_life,
    "OverTime_Yes": 1 if overtime == "Yes" else 0,
    "Gender_Male": 1 if gender == "Male" else 0,
    "MaritalStatus_Married": 1 if marital == "Married" else 0,
    "MaritalStatus_Single": 1 if marital == "Single" else 0
}

# Convert to dataframe
input_df = pd.DataFrame([input_dict])

# Align with training columns
input_df = input_df.reindex(columns=columns, fill_value=0)

# Scale input
input_scaled = scaler.transform(input_df)

# ---------------- PREDICTION ----------------

if st.button("Predict Attrition Risk"):
    prob = model.predict_proba(input_scaled)[0][1] * 100

    if prob > 70:
        risk = "🔴 HIGH RISK"
    elif prob > 40:
        risk = "🟠 MEDIUM RISK"
    else:
        risk = "🟢 LOW RISK"

    st.subheader(f"Attrition Probability: {prob:.2f}%")
    st.subheader(f"Risk Level: {risk}")
