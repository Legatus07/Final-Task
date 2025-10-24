import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

model = joblib.load("ML_Model.pkl")

st.title(" Credit Mix Prediction App")

st.write("""
This app predicts a user's **Credit Mix** (Good / Standard / Bad)
based on their financial profile.
""")

Age = st.number_input("Age", min_value=18, max_value=100, value=30)
Annual_Income = st.number_input("Annual Income (₹)", min_value=0, value=500000)
Credit_Score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
Loan_Amount = st.number_input("Loan Amount", min_value=0, value=100000)

Occupation = st.selectbox("Occupation", [
    "Scientist", "Teacher", "Engineer", "Lawyer", "Entrepreneur",
    "Doctor", "Artist", "Other"
])

Payment_Behaviour = st.selectbox("Payment Behaviour", [
    "High_spent_Small_value_payments", 
    "Low_spent_Large_value_payments",
    "Low_spent_Medium_value_payments"
])

input_data = pd.DataFrame({
    "Age": [Age],
    "Annual_Income": [Annual_Income],
    "Credit_Score": [Credit_Score],
    "Loan_Amount": [Loan_Amount],
    "Occupation": [Occupation],
    "Payment_Behaviour": [Payment_Behaviour]
})

le = LabelEncoder()
for col in input_data.select_dtypes(include="object").columns:
    input_data[col] = le.fit_transform(input_data[col])

if st.button("Predict Credit Mix"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Credit Mix: **{prediction[0]}**")