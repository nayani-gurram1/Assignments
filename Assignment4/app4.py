import streamlit as st
import pandas as pd
import pickle
import os

# --------------------------
# Load files
# --------------------------
st.write("Files:", os.listdir())

churn_model = pickle.load(open("churn_model.pkl", "rb"))
rev_model = pickle.load(open("revenue_model.pkl", "rb"))
scaler = pickle.load(open("churn_scaler.pkl", "rb"))
columns = pickle.load(open("churn_columns.pkl", "rb"))
cluster_model = pickle.load(open("cluster_model.pkl", "rb"))

# --------------------------
# Page config
# --------------------------
st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📞"
)

st.title("📞 AI Telecom Customer Churn Prediction Platform")
st.write("Predict churn probability, revenue loss, cluster, and retention recommendation.")

# --------------------------
# Sidebar inputs
# --------------------------
st.sidebar.header("Customer Details")

tenure = st.sidebar.number_input("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges", 0.0, 500.0, 70.0)
total_charges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 1000.0)

contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

# --------------------------
# Recommendation logic
# --------------------------
def retention_recommendation(prob):
    if prob > 80:
        return "🔥 Offer 30% discount + priority support"
    elif prob > 50:
        return "🎁 Offer loyalty rewards"
    else:
        return "✅ Maintain regular engagement"

# --------------------------
# Prediction
# --------------------------
if st.button("Predict Customer Risk"):

    input_data = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        f"Contract_{contract}": 1,
        f"InternetService_{internet_service}": 1,
        f"PaymentMethod_{payment_method}": 1
    }

    df = pd.DataFrame([input_data])

    # Match training columns
    df = df.reindex(columns=columns, fill_value=0)

    scaled = scaler.transform(df)

    churn_prob = churn_model.predict_proba(scaled)[0][1] * 100
    revenue_loss = rev_model.predict(df)[0]
    cluster = cluster_model.predict(df)[0]

    recommendation = retention_recommendation(churn_prob)

    st.subheader("Prediction Results")

    st.metric("Churn Probability", f"{churn_prob:.2f}%")
    st.metric("Expected Revenue Loss", f"${revenue_loss:.2f}")
    st.metric("Customer Cluster", cluster)

    st.success(f"Recommendation: {recommendation}")