import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Load model
model_data = pickle.load(open("fraud_model_final.pkl", "rb"))
model = model_data["model"]
threshold = model_data["threshold"]

app = FastAPI()

# Input schema
class Transaction(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}

@app.post("/predict")
def predict(data: Transaction):
    arr = np.array(data.features).reshape(1, -1)
    prob = model.predict_proba(arr)[0][1]
    pred = 1 if prob > threshold else 0

    return {
        "fraud_probability": float(prob),
        "prediction": "FRAUD 🚨" if pred == 1 else "LEGIT ✅"
    }