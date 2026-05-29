import streamlit as st
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "random_forest.pkl"))
columns = joblib.load(os.path.join(BASE_DIR, "columns.pkl"))

st.set_page_config(page_title="Employee Attrition Prediction", page_icon="📊")

st.title("AI-Powered Employee Attrition Prediction System")
st.write("Enter Employee Details")

age = st.number_input("Age", 18, 60, 35)
daily_rate = st.number_input("Daily Rate", value=1000)
distance_from_home = st.number_input("Distance From Home", value=5)
education = st.selectbox("Education", [1,2,3,4,5])
environment_satisfaction = st.selectbox("Environment Satisfaction", [1,2,3,4])
job_involvement = st.selectbox("Job Involvement", [1,2,3,4])
job_level = st.selectbox("Job Level", [1,2,3,4,5])
job_satisfaction = st.selectbox("Job Satisfaction", [1,2,3,4])
monthly_income = st.number_input("Monthly Income", value=4500)
num_companies_worked = st.number_input("Number Of Companies Worked", value=2)
percent_salary_hike = st.number_input("Percent Salary Hike", value=15)
performance_rating = st.selectbox("Performance Rating", [3,4])
relationship_satisfaction = st.selectbox("Relationship Satisfaction", [1,2,3,4])
stock_option_level = st.selectbox("Stock Option Level", [0,1,2,3])
total_working_years = st.number_input("Total Working Years", value=10)
training_times_last_year = st.number_input("Training Times Last Year", value=2)
work_life_balance = st.selectbox("Work Life Balance", [1,2,3,4])
years_at_company = st.number_input("Years At Company", value=5)
years_in_current_role = st.number_input("Years In Current Role", value=3)
years_since_last_promotion = st.number_input("Years Since Last Promotion", value=1)
years_with_curr_manager = st.number_input("Years With Current Manager", value=3)

if st.button("Predict Attrition"):

    # Create dataframe with exact training columns
    features = pd.DataFrame(0, index=[0], columns=columns)

    # Fill numeric features
    feature_values = {
        "Age": age,
        "DailyRate": daily_rate,
        "DistanceFromHome": distance_from_home,
        "Education": education,
        "EnvironmentSatisfaction": environment_satisfaction,
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobSatisfaction": job_satisfaction,
        "MonthlyIncome": monthly_income,
        "NumCompaniesWorked": num_companies_worked,
        "PercentSalaryHike": percent_salary_hike,
        "PerformanceRating": performance_rating,
        "RelationshipSatisfaction": relationship_satisfaction,
        "StockOptionLevel": stock_option_level,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training_times_last_year,
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_current_role,
        "YearsSinceLastPromotion": years_since_last_promotion,
        "YearsWithCurrManager": years_with_curr_manager
    }

    for col, val in feature_values.items():
        if col in features.columns:
            features[col] = val

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1] * 100

    if probability >= 75:
        risk = "HIGH"
    elif probability >= 40:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    st.subheader("Prediction Result")
    st.write("Attrition Probability:", round(probability, 2), "%")
    st.write("Risk Level:", risk)

    if prediction == 1:
        st.error("Employee likely to leave.")
    else:
        st.success("Employee likely to stay.")
