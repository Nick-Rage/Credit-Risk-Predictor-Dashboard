from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
import os

script_dir = os.path.dirname(__file__)  
model_path = os.path.join(script_dir, "loan_risk_model.pkl")
model = joblib.load(model_path)

app = FastAPI()

# Define input structure
class CreditRiskInput(BaseModel):
    credit_score: int
    credit_limit: int
    total_overdue_payments: int
    highest_balance: int
    current_balance: int
    income: int
    age: int
    employment_status: int
    duration_of_credit: int

# Define API endpoint
@app.post("/predict/")
def predict_risk(data: CreditRiskInput):
    input_data = pd.DataFrame([data.dict()])
    
    risk_score = model.predict(input_data)[0]
    
    return {"predicted_risk_score": risk_score}

# Run API with: uvicorn api:app --reload
