import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go

# 🔗 CHANGE THIS AFTER DEPLOY
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Aura Fraud Guard", layout="wide")

# ---------------- API ----------------
def call_api(features):
    try:
        r = requests.post(API_URL, json={"features": features})
        return r.json()
    except:
        return {"error": "API not reachable"}

# ---------------- SIDEBAR ----------------
st.sidebar.title("💳 Aura Fraud Guard")
menu = st.sidebar.radio("Navigation", ["Dashboard", "Manual Audit", "Batch Scan"])

st.title(menu)

# =========================================================
# 📊 DASHBOARD
# =========================================================
if menu == "Dashboard":

    st.subheader("📊 System Overview")

    try:
        df = pd.read_csv("../backend/fraud_logs.csv")

        total = len(df)
        fraud = df["prediction"].str.contains("FRAUD").sum()
        legit = total - fraud

        c1, c2, c3 = st.columns(3)
        c1.metric("Total", total)
        c2.metric("Fraud", fraud)
        c3.metric("Legit", legit)

        st.markdown("### 📈 Risk Trend")
        df["index"] = range(len(df))
        fig = px.line(df, x="index", y="probability")
        st.plotly_chart(fig, use_container_width=True)

    except:
        st.info("No data yet. Run predictions.")

# =========================================================
# 🔍 MANUAL AUDIT
# =========================================================
elif menu == "Manual Audit":

    st.subheader("🔍 Single Transaction Check")

    features = []

    col1, col2, col3 = st.columns(3)

    with col1:
        features.append(st.number_input("Amount", 0.0))
        features.append(st.number_input("Time", 0.0))
        for i in range(1, 9):
            features.append(st.number_input(f"V{i}", 0.0))

    with col2:
        for i in range(9, 19):
            features.append(st.number_input(f"V{i}", 0.0))

    with col3:
        for i in range(19, 29):
            features.append(st.number_input(f"V{i}", 0.0))

    if st.button("🚀 Predict"):

        result = call_api(features)

        if "error" in result:
            st.error(result["error"])
        else:
            prob = result["fraud_probability"]
            pred = result["prediction"]

            st.metric("Fraud Probability", f"{prob:.4f}")
            st.progress(prob)

            if "FRAUD" in pred:
                st.error(pred)
            else:
                st.success(pred)

            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={'text': "Risk %"},
                gauge={'axis': {'range': [0, 100]}}
            ))
            st.plotly_chart(fig)

# =========================================================
# 📂 BATCH SCAN
# =========================================================
elif menu == "Batch Scan":

    file = st.file_uploader("Upload CSV", type="csv")

    if file:
        df = pd.read_csv(file)

        if len(df) > 1000:
            df = df.head(1000)
            st.warning("Using first 1000 rows only")

        st.dataframe(df.head())

        if st.button("Run Detection"):

            probs, preds = [], []
            progress = st.progress(0)

            for i, row in enumerate(df.values):
                result = call_api(row.tolist())

                if "fraud_probability" in result:
                    probs.append(result["fraud_probability"])
                    preds.append(result["prediction"])
                else:
                    probs.append(0)
                    preds.append("ERROR")

                progress.progress((i + 1) / len(df))

            df["Fraud Probability"] = probs
            df["Prediction"] = preds

            st.success("Done ✅")

            # Charts
            fig = px.histogram(df, x="Fraud Probability")
            st.plotly_chart(fig)

            st.dataframe(df.head())

            st.download_button(
                "Download",
                df.to_csv(index=False),
                "results.csv"
            )