import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

model_data = pickle.load(open("model.pkl", "rb"))
model = model_data["model"]
threshold = model_data["threshold"]

app = FastAPI(title="Fraud Detection API")

class Transaction(BaseModel):
    features: list

@app.get("/")
def home():
    return {"status": "Fraud API Running"}

@app.post("/predict")
def predict(data: Transaction):

    arr = np.array(data.features).reshape(1, -1)
    prob = model.predict_proba(arr)[0][1]
    pred = 1 if prob > threshold else 0

    return {
        "fraud_probability": float(prob),
        "prediction": "FRAUD 🚨" if pred == 1 else "LEGIT ✅"
    }