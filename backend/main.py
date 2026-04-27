import pickle
import numpy as np
import os
import pandas as pd
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel

# =========================================================
# ⚡ LOAD MODEL (SAFE)
# =========================================================
MODEL_PATH = "fraud_model_final.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found!")

with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
threshold = model_data["threshold"]

# =========================================================
# 🚀 INIT APP
# =========================================================
app = FastAPI(title="Aura Fraud Detection API")

# =========================================================
# 📦 INPUT SCHEMA
# =========================================================
class Transaction(BaseModel):
    features: list


# =========================================================
# 📁 LOG FILE (AUTO CREATE)
# =========================================================
LOG_FILE = "fraud_logs.csv"

if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["time", "probability", "prediction"]).to_csv(LOG_FILE, index=False)


# =========================================================
# ❤️ HEALTH CHECK
# =========================================================
@app.get("/")
def home():
    return {"status": "API running"}


# =========================================================
# 🔍 PREDICTION ENDPOINT
# =========================================================
@app.post("/predict")
def predict(data: Transaction):

    try:
        # Validate input length
        if len(data.features) != 30:
            return {"error": "Expected 30 features"}

        arr = np.array(data.features).reshape(1, -1)

        prob = model.predict_proba(arr)[0][1]
        pred = 1 if prob > threshold else 0

        label = "FRAUD 🚨" if pred == 1 else "LEGIT ✅"

        # ---------------- LOGGING ----------------
        new_row = pd.DataFrame([{
            "time": datetime.now(),
            "probability": prob,
            "prediction": label
        }])

        new_row.to_csv(LOG_FILE, mode='a', header=False, index=False)

        return {
            "fraud_probability": float(prob),
            "prediction": label
        }

    except Exception as e:
        return {"error": str(e)}