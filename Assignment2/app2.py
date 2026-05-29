import streamlit as st
import pandas as pd
import pickle

# Load model files
model = pickle.load(open("loan_rf.pkl", "rb"))
scaler = pickle.load(open("loan_scaler.pkl", "rb"))
columns = pickle.load(open("loan_columns.pkl", "rb"))

st.set_page_config(page_title="Loan Default Prediction", page_icon="💰")

st.title("💰 AI Loan Default Prediction System")
st.write("Predict whether a customer is likely to default on a loan.")

st.sidebar.header("Enter Applicant Details")

# Input fields
age = st.sidebar.number_input("Age", 18, 100, 30)
income = st.sidebar.number_input("Annual Income", 10000, 1000000, 50000)
loan_amount = st.sidebar.number_input("Loan Amount", 1000, 500000, 50000)
credit_score = st.sidebar.number_input("Credit Score", 300, 900, 650)
loan_term = st.sidebar.number_input("Loan Term (months)", 6, 360, 60)

# Example categorical inputs
employment_type = st.sidebar.selectbox(
    "Employment Type",
    ["Salaried", "Self-employed"]
)

marital_status = st.sidebar.selectbox(
    "Marital Status",
    ["Single", "Married"]
)

if st.button("Predict Default Risk"):

    # Create input dictionary
    input_data = {
        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "LoanTerm": loan_term,
        f"EmploymentType_{employment_type}": 1,
        f"MaritalStatus_{marital_status}": 1
    }

    # Convert to DataFrame
    df_input = pd.DataFrame([input_data])

    # Align with training columns
    df_input = df_input.reindex(columns=columns, fill_value=0)

    # Scale
    scaled_data = scaler.transform(df_input)

    # Predict
    probability = model.predict_proba(scaled_data)[0][1] * 100

    risk = "🔴 HIGH RISK" if probability > 60 else "🟢 LOW RISK"

    st.subheader("Prediction Result")
    st.metric("Default Probability", f"{probability:.2f}%")
    st.success(f"Risk Level: {risk}")