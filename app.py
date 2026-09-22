import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="ABC Ltd. — Attrition Risk Predictor", page_icon="📊", layout="centered")

# ---------- Load model artifacts ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("attrition_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, feature_columns

model, scaler, feature_columns = load_artifacts()

st.title("📊 ABC Ltd. — Employee Attrition Risk Predictor")
st.write(
    "Enter an employee's details below to estimate their probability of leaving the company. "
    "This tool uses a logistic regression model trained on historical HR data."
)

st.divider()

# ---------- Input form ----------
with st.form("employee_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 60, 30)
        distance = st.slider("Distance From Home (km)", 1, 30, 5)
        monthly_income = st.number_input("Monthly Income ($)", 1000, 20000, 5000, step=100)
        total_working_years = st.slider("Total Working Years", 0, 40, 8)
        years_at_company = st.slider("Years At Company", 0, 40, 5)
        years_in_role = st.slider("Years In Current Role", 0, 20, 3)
        years_since_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)
        years_with_manager = st.slider("Years With Current Manager", 0, 20, 3)
        num_companies = st.slider("Number Of Companies Worked", 0, 10, 2)

    with col2:
        job_satisfaction = st.select_slider("Job Satisfaction (1=Low, 4=High)", options=[1, 2, 3, 4], value=3)
        env_satisfaction = st.select_slider("Environment Satisfaction (1=Low, 4=High)", options=[1, 2, 3, 4], value=3)
        work_life_balance = st.select_slider("Work-Life Balance (1=Bad, 4=Best)", options=[1, 2, 3, 4], value=3)
        job_involvement = st.select_slider("Job Involvement (1=Low, 4=High)", options=[1, 2, 3, 4], value=3)
        job_level = st.select_slider("Job Level (1=Entry, 5=Senior)", options=[1, 2, 3, 4, 5], value=2)
        stock_option = st.select_slider("Stock Option Level", options=[0, 1, 2, 3], value=0)
        overtime = st.radio("Works Overtime?", ["No", "Yes"], horizontal=True)
        business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

    department = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])
    job_role = st.selectbox(
        "Job Role",
        ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director",
         "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"],
    )
    education_field = st.selectbox(
        "Education Field",
        ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"],
    )
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)

    submitted = st.form_submit_button("Predict Attrition Risk", use_container_width=True)

# ---------- Build feature row & predict ----------
if submitted:
    # Reasonable dataset-average defaults for fields not exposed in the simplified form
    row = {
        "Age": age,
        "DailyRate": 800,
        "DistanceFromHome": distance,
        "Education": 3,
        "EnvironmentSatisfaction": env_satisfaction,
        "HourlyRate": 65,
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobSatisfaction": job_satisfaction,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": 14000,
        "NumCompaniesWorked": num_companies,
        "PercentSalaryHike": 15,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 3,
        "StockOptionLevel": stock_option,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": 2,
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_role,
        "YearsSinceLastPromotion": years_since_promotion,
        "YearsWithCurrManager": years_with_manager,
        "BusinessTravel_Travel_Frequently": 1 if business_travel == "Travel_Frequently" else 0,
        "BusinessTravel_Travel_Rarely": 1 if business_travel == "Travel_Rarely" else 0,
        "Department_Research & Development": 1 if department == "Research & Development" else 0,
        "Department_Sales": 1 if department == "Sales" else 0,
        "EducationField_Life Sciences": 1 if education_field == "Life Sciences" else 0,
        "EducationField_Marketing": 1 if education_field == "Marketing" else 0,
        "EducationField_Medical": 1 if education_field == "Medical" else 0,
        "EducationField_Other": 1 if education_field == "Other" else 0,
        "EducationField_Technical Degree": 1 if education_field == "Technical Degree" else 0,
        "Gender_Male": 1 if gender == "Male" else 0,
        "JobRole_Human Resources": 1 if job_role == "Human Resources" else 0,
        "JobRole_Laboratory Technician": 1 if job_role == "Laboratory Technician" else 0,
        "JobRole_Manager": 1 if job_role == "Manager" else 0,
        "JobRole_Manufacturing Director": 1 if job_role == "Manufacturing Director" else 0,
        "JobRole_Research Director": 1 if job_role == "Research Director" else 0,
        "JobRole_Research Scientist": 1 if job_role == "Research Scientist" else 0,
        "JobRole_Sales Executive": 1 if job_role == "Sales Executive" else 0,
        "JobRole_Sales Representative": 1 if job_role == "Sales Representative" else 0,
        "MaritalStatus_Married": 1 if marital_status == "Married" else 0,
        "MaritalStatus_Single": 1 if marital_status == "Single" else 0,
        "OverTime_Yes": 1 if overtime == "Yes" else 0,
    }

    input_df = pd.DataFrame([row])
    # Ensure column order matches training exactly; fill any missing with 0
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    input_scaled = scaler.transform(input_df)
    probability = model.predict_proba(input_scaled)[0][1]

    st.divider()
    st.subheader("Prediction Result")

    if probability >= 0.5:
        st.error(f"⚠️ High attrition risk: **{probability:.1%}**")
    elif probability >= 0.3:
        st.warning(f"🟡 Moderate attrition risk: **{probability:.1%}**")
    else:
        st.success(f"✅ Low attrition risk: **{probability:.1%}**")

    st.progress(min(probability, 1.0))

    st.caption(
        "This is a statistical estimate based on historical patterns, not a certainty. "
        "Use it alongside your own judgment of the employee's situation."
    )

st.divider()
st.caption(
    "Built for ABC Ltd. managerial decision support — logistic regression model trained on anonymized HR data."
)
