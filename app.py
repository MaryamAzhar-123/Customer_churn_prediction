import streamlit as st
import pandas as pd
import pickle

    #   Page
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

    #   Load
with open("churn.pkl", "rb") as file:
    model = pickle.load(file)

    # Title

st.title("📊 Customer Churn Prediction")
st.write(
    "Enter customer information below to predict whether the customer "
    "is likely to churn."
)

st.divider()

    # Customer Information

st.subheader("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )

with col2:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

with col3:
    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0
    )

st.divider()

    # Services

st.subheader("🌐 Services")

col1, col2 = st.columns(2)

with col1:
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

with col2:
    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

st.divider()

    # Contract & Payment
st.subheader("💳 Contract & Payment")

col1, col2 = st.columns(2)

with col1:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

with col2:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

st.divider()

    # Prediction
    
if st.button("🔍 Predict Churn", use_container_width=True):

    input_data = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "Contract": [contract],
        "PaymentMethod": [payment_method]
    })

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.divider()
    st.subheader("📈 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1 or prediction == "Yes":
            st.error("⚠️ Customer is likely to churn.")
        else:
            st.success("✅ Customer is unlikely to churn.")

    with col2:
        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

    st.progress(float(probability))

    if probability >= 0.5:
        st.warning(
            "The model estimates a higher likelihood of churn for this customer."
        )
    else:
        st.info(
            "The model estimates a lower likelihood of churn for this customer."
        )