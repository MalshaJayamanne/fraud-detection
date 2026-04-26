import streamlit as st
import pandas as pd
import numpy as np
import requests
import shap
import pickle

st.set_page_config(page_title="Fraud Dashboard", layout="wide")

# ---------------- API ----------------
API_URL = "https://your-api.onrender.com/predict"  # CHANGE AFTER DEPLOY

def call_api(features):
    try:
        r = requests.post(API_URL, json={"features": features})
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# ---------------- LOAD MODEL FOR SHAP ----------------
model_data = pickle.load(open("fraud_model_final.pkl", "rb"))
model = model_data["model"]

explainer = shap.Explainer(model)

# ---------------- UI ----------------
st.title("💳 Fraud Detection System")

menu = st.sidebar.radio("Menu", ["Live Monitor", "Manual Audit"])

# ---------------- LIVE MONITOR ----------------
if menu == "Live Monitor":
    st.subheader("📊 Live Monitoring")

    try:
        df = pd.read_csv("../backend/fraud_logs.csv")

        st.metric("Total", len(df))
        st.metric("Fraud", df["prediction"].sum())

        st.line_chart(df["probability"])
        st.dataframe(df.tail())

    except:
        st.warning("No logs yet.")

# ---------------- MANUAL ----------------
if menu == "Manual Audit":

    features = [st.number_input(f"Feature {i}", 0.0) for i in range(30)]

    if st.button("Predict"):

        result = call_api(features)

        if "error" in result:
            st.error(result["error"])
        else:
            prob = result["fraud_probability"]
            prediction = result["prediction"]

            st.metric("Probability", prob)
            st.write(prediction)

            # SHAP
            st.subheader("Explainability")
            shap_values = explainer(np.array(features).reshape(1, -1))
            st.bar_chart(shap_values.values[0])