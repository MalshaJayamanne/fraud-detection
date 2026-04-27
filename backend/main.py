import pickle
import numpy as np
import pandas as pd
import os
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel


# LOAD MODEL
MODEL_PATH = "fraud_model_final.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found")

with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)

model = data["model"]
threshold = data["threshold"]

# APP INIT
app = FastAPI(title="Fraud Detection API")


# INPUT SCHEMA
class Transaction(BaseModel):
    features: list

# LOG FILE
LOG_FILE = "fraud_logs.csv"

if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["time", "probability", "prediction"]).to_csv(LOG_FILE, index=False)

# ------------------------------------
# HEALTH CHECK
# ====================================
@app.get("/")
def home():
    return {"status": "API running"}

# --------------------------------
# PREDICT ENDPOINT
# ================================
@app.post("/predict")
def predict(data: Transaction):

    try:
        if len(data.features) != 30:
            return {"error": "Expected 30 features"}

        arr = np.array(data.features).reshape(1, -1)

        prob = model.predict_proba(arr)[0][1]
        pred = "FRAUD" if prob > threshold else "LEGIT"

        # -----------------------
        # LOG RESULTS
        # =======================
        log = pd.DataFrame([{
            "time": datetime.now(),
            "probability": float(prob),
            "prediction": pred
        }])

        log.to_csv(LOG_FILE, mode="a", header=False, index=False)

        return {
            "fraud_probability": float(prob),
            "prediction": pred
        }

    except Exception as e:
        return {"error": str(e)}