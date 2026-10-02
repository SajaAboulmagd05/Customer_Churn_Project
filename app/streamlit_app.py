import streamlit as st
import pandas as pd
import joblib

model = joblib.load('models/best_classifier.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')

st.title("Customer Churn Predictor")

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.slider("Monthly Charges", 18.0, 120.0, 70.0)
total_charges = st.slider("Total Charges", 0.0, 9000.0, 1000.0)

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment = st.selectbox("Payment Method", ["Bank transfer (automatic)", "Credit card (automatic)",
                                           "Electronic check", "Mailed check"])

senior = st.checkbox("Senior Citizen")
partner = st.checkbox("Has Partner")
dependents = st.checkbox("Has Dependents")
phone_service = st.checkbox("Phone Service", value=True)
multiple_lines = st.checkbox("Multiple Lines")
online_security = st.checkbox("Online Security")
online_backup = st.checkbox("Online Backup")
device_protection = st.checkbox("Device Protection")
tech_support = st.checkbox("Tech Support")
streaming_tv = st.checkbox("Streaming TV")
streaming_movies = st.checkbox("Streaming Movies")
paperless_billing = st.checkbox("Paperless Billing", value=True)

if st.button("Predict"):
    row = {col: 0 for col in feature_columns}

    row['Tenure Months'] = tenure
    row['Monthly Charges'] = monthly_charges
    row['Total Charges'] = total_charges
    row['Senior Citizen'] = int(senior)
    row['Partner'] = int(partner)
    row['Dependents'] = int(dependents)
    row['Phone Service'] = int(phone_service)
    row['Multiple Lines'] = int(multiple_lines)
    row['Online Security'] = int(online_security)
    row['Online Backup'] = int(online_backup)
    row['Device Protection'] = int(device_protection)
    row['Tech Support'] = int(tech_support)
    row['Streaming TV'] = int(streaming_tv)
    row['Streaming Movies'] = int(streaming_movies)
    row['Paperless Billing'] = int(paperless_billing)

    if contract == "One year":
        row['Contract_One year'] = 1
    elif contract == "Two year":
        row['Contract_Two year'] = 1

    if internet == "Fiber optic":
        row['Internet Service_Fiber optic'] = 1
    elif internet == "No":
        row['Internet Service_No'] = 1

    if payment == "Credit card (automatic)":
        row['Payment Method_Credit card (automatic)'] = 1
    elif payment == "Electronic check":
        row['Payment Method_Electronic check'] = 1
    elif payment == "Mailed check":
        row['Payment Method_Mailed check'] = 1

    X_new = pd.DataFrame([row])[feature_columns]
    X_new[['Monthly Charges', 'Total Charges', 'Tenure Months']] = scaler.transform(
        X_new[['Monthly Charges', 'Total Charges', 'Tenure Months']]
    )

    prob = model.predict_proba(X_new)[0][1]
    st.metric("Churn Probability", f"{prob:.1%}")
    if prob > 0.5:
        st.error("High risk of churn")
    else:
        st.success("Low risk of churn")