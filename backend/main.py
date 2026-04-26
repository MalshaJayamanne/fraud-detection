from fastapi import FastAPI
import numpy as np
import pickle
from pydantic import BaseModel
import pandas as pd
import os
from datetime import datetime

app = FastAPI()

# Load model
model_data = pickle.load(open("fraud_model_final.pkl", "rb"))
model = model_data["model"]
threshold = model_data["threshold"]

# Log file
LOG_FILE = "fraud_logs.csv"

if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["timestamp", "probability", "prediction"]).to_csv(LOG_FILE, index=False)

class Transaction(BaseModel):
    features: list

@app.get("/")
def home():
    return {"status": "API running"}

@app.post("/predict")
def predict(data: Transaction):

    arr = np.array(data.features).reshape(1, -1)

    prob = model.predict_proba(arr)[0][1]
    pred = 1 if prob > threshold else 0

    # Save log
    new_entry = pd.DataFrame([{
        "timestamp": datetime.now(),
        "probability": float(prob),
        "prediction": int(pred)
    }])
    new_entry.to_csv(LOG_FILE, mode='a', header=False, index=False)

    return {
        "fraud_probability": float(prob),
        "prediction": "FRAUD 🚨" if pred else "LEGIT ✅"
    }