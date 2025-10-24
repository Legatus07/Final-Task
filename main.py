from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np


model = joblib.load("Logistic.pkl")
scaler = joblib.load("scaler.pkl")
le_y = joblib.load("label_encoder.pkl")
feature_names = joblib.load("feature_names.pkl")

app = FastAPI(
    title="Credit Mix Predictor API",
    description="API that predicts Credit Mix: Good, Standard, or Bad",
    version="1.0"
)

class CreditInput(BaseModel):
    Age: int
    Annual_Income: float
    Monthly_Inhand_Salary: float
    Num_Bank_Accounts: int
    Num_Credit_Card: int
    Num_of_Loan: int
    Interest_Rate: float
    Num_of_Delayed_Payment: int
    Outstanding_Debt: float
    Credit_Utilization_Ratio: float

@app.get("/")
def home():
    return {"message": "Welcome to the Credit Mix Prediction API!"}

@app.post("/predict")
def predict_credit_mix(data: CreditInput):
    # Convert input data to DataFrame
    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])

    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)
    label = le_y.inverse_transform(prediction)[0]

    return {
        "prediction": label,
        "inputs": input_dict
    }
