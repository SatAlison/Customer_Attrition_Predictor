
# 💳 Streamlit App - Bank Customer Churn Predictor


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# Load model and scaler

model = joblib.load("xgb_churn_model.pkl")
scaler = joblib.load("scaler.pkl")


# Streamlit page config

st.set_page_config(
    page_title="Customer Attrition Forecasting in Banking",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom colors
primary_color = "#1f77b4"  # Blue
secondary_color = "#ff6f61"  # Salmon



# App header

st.markdown(
    f"<h1 style='text-align:center; color:{primary_color}'>💳 Customer Attrition Predictor</h1>",
    unsafe_allow_html=True
)
# Short summary / explanation
st.markdown(
    f"<p style='text-align:center; color:#333333; font-size:20px;'>"
    "Customer attrition (or churn) occurs when a client stops using a bank's services. "
    "Use this tool to quickly assess the likelihood of a customer leaving and take proactive steps.</p>",
    unsafe_allow_html=True
)
st.markdown(
    f"<p style='text-align:center; color:{secondary_color}; font-size:22px;'>"
    "Enter customer details to predict churn probability and risk.</p>",
    unsafe_allow_html=True
)
st.markdown("---")


# Input form in sidebar

with st.sidebar.form("customer_form"):
    st.header("Customer Details")
    Customer_Age = st.number_input("Customer Age", min_value=18, max_value=100, value=40)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Dependent_count = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
    Education_Level = st.selectbox("Education Level", ["Uneducated", "High School", "College", "Graduate", "Post-Graduate", "Doctorate"])
    Income_Category = st.selectbox("Income Category", ["Less than $40K", "$40K-$60K", "$60K-$80K", "$80K-$120K", "$120K +"])
    Months_on_book = st.number_input("Months on Book", min_value=1, max_value=120, value=30)
    Total_Relationship_Count = st.number_input("Total Relationship Count", min_value=1, max_value=10, value=3)
    Months_Inactive_12_mon = st.number_input("Months Inactive (Last 12)", min_value=0, max_value=12, value=2)
    Contacts_Count_12_mon = st.number_input("Contacts Count (Last 12)", min_value=0, max_value=10, value=1)
    Credit_Limit = st.number_input("Credit Limit", min_value=100.0, max_value=50000.0, value=5000.0)
    Total_Revolving_Bal = st.number_input("Total Revolving Balance", min_value=0.0, max_value=50000.0, value=1500.0)
    Total_Amt_Chng_Q4_Q1 = st.number_input("Total Amount Change Q4/Q1", min_value=0.0, value=1.0)
    Total_Trans_Amt = st.number_input("Total Transaction Amount", min_value=0.0, value=5000.0)
    Total_Trans_Ct = st.number_input("Total Transaction Count", min_value=0, value=50)
    Total_Ct_Chng_Q4_Q1 = st.number_input("Total Count Change Q4/Q1", min_value=0.0, value=1.0)
    Avg_Utilization_Ratio = st.number_input("Average Utilization Ratio", min_value=0.0, max_value=1.0, value=0.3)
    Marital_Status_Single = st.selectbox("Marital Status Single", ["Yes", "No"])
    submitted = st.form_submit_button("Predict Churn")

# Handle prediction

if submitted:
    # Encode categorical features
    education_mapping = {
        "Uneducated": 0, "High School": 1, "College": 2,
        "Graduate": 3, "Post-Graduate": 4, "Doctorate": 5
    }
    income_mapping = {
        "Less than $40K": 0, "$40K-$60K": 1, "$60K-$80K": 2,
        "$80K-$120K": 3, "$120K +": 4
    }

    input_df = pd.DataFrame({
        'Customer_Age': [Customer_Age],
        'Gender': [1 if Gender=="Male" else 0],
        'Dependent_count': [Dependent_count],
        'Education_Level': [education_mapping[Education_Level]],
        'Income_Category': [income_mapping[Income_Category]],
        'Months_on_book': [Months_on_book],
        'Total_Relationship_Count': [Total_Relationship_Count],
        'Months_Inactive_12_mon': [Months_Inactive_12_mon],
        'Contacts_Count_12_mon': [Contacts_Count_12_mon],
        'Credit_Limit': [Credit_Limit],
        'Total_Revolving_Bal': [Total_Revolving_Bal],
        'Total_Amt_Chng_Q4_Q1': [Total_Amt_Chng_Q4_Q1],
        'Total_Trans_Amt': [Total_Trans_Amt],
        'Total_Trans_Ct': [Total_Trans_Ct],
        'Total_Ct_Chng_Q4_Q1': [Total_Ct_Chng_Q4_Q1],
        'Avg_Utilization_Ratio': [Avg_Utilization_Ratio],
        'Marital_Status_Single': [1 if Marital_Status_Single=="Yes" else 0]
    })

    # Scale numeric columns
    numeric_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book', 
                    'Total_Relationship_Count', 'Months_Inactive_12_mon', 
                    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal', 
                    'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct', 
                    'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Prediction
    proba = model.predict_proba(input_df)[:,1][0]
    prediction = "Attrited" if proba >= 0.6 else "Existing"

    
    # Display results
   
    st.markdown("---")
    st.subheader("Prediction Result")
    st.markdown(
        f"<h2 style='color:{secondary_color}'>Customer is likely to be : <b>{prediction}</b></h2>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<p style='font-size:18px; color:{primary_color}'>Churn Probability: <b>{proba:.2f}</b></p>",
        unsafe_allow_html=True
    )

    
    # Visualize churn probability
    
    fig, ax = plt.subplots(figsize=(6,3))
    sns.barplot(x=[prediction], y=[proba], palette=[secondary_color], ax=ax)
    ax.set_ylim(0,1)
    ax.set_ylabel("Probability")
    ax.set_title("Churn Probability Visualization", color=primary_color)
    st.pyplot(fig)
