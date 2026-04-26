import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import time

# ================= API =================
API_URL = "http://127.0.0.1:8000/predict"

def call_api(features):
    try:
        r = requests.post(API_URL, json={"features": features})
        return r.json()
    except:
        return {"error": "API not reachable"}

# ================= SHAP SETUP =================
import pickle

model_data = pickle.load(open("../backend/model.pkl", "rb"))
model = model_data["model"]
explainer = shap.TreeExplainer(model)

# ================= UI =================
st.set_page_config(page_title="Fraud AI System", layout="wide")

menu = st.sidebar.radio(
    "Control Panel",
    ["Dashboard", "Manual Check", "Batch Scan", "Explain AI", "Live Stream"]
)

st.title("🏦 Fraud Detection Intelligence System")

# =====================================================
# 📊 DASHBOARD
# =====================================================
if menu == "Dashboard":

    st.subheader("📊 System Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions", "1.2M")
    c2.metric("Fraud Rate", "0.42%")
    c3.metric("Model Accuracy", "99.3%")

    df = pd.DataFrame({
        "hour": range(24),
        "fraud": np.random.randint(0, 50, 24)
    })

    fig = px.line(df, x="hour", y="fraud", title="Fraud Trend")
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 🔍 MANUAL CHECK
# =====================================================
elif menu == "Manual Check":

    st.subheader("🔍 Transaction Analysis")

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

    if st.button("Predict"):

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

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={'text': "Risk Score"}
            ))
            st.plotly_chart(fig)

# =====================================================
# 📂 BATCH SCAN
# =====================================================
elif menu == "Batch Scan":

    st.subheader("📂 Bulk Fraud Detection")

    file = st.file_uploader("Upload CSV", type="csv")

    if file:

        df = pd.read_csv(file)

        if len(df) > 1000:
            df = df.head(1000)
            st.warning("Using first 1000 rows")

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

            st.success("Done")

            fig = px.histogram(df, x="Fraud Probability")
            st.plotly_chart(fig)

            st.dataframe(df.head())

# =====================================================
# 🧠 SHAP EXPLAINABILITY
# =====================================================
elif menu == "Explain AI":

    st.subheader("🧠 SHAP Explainability Dashboard")

    sample = np.random.rand(1, 29)

    shap_values = explainer.shap_values(sample)

    st.write("### Global Feature Impact")

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, sample, show=False)
    st.pyplot(fig)

    st.write("### Single Prediction Explanation")

    fig2, ax2 = plt.subplots()
    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        sample,
        matplotlib=True
    )
    st.pyplot(fig2)

# =====================================================
# 📡 LIVE STREAMING
# =====================================================
elif menu == "Live Stream":

    st.subheader("📡 Real-Time Fraud Detection")

    placeholder = st.empty()

    for i in range(30):

        features = list(np.random.rand(29))
        result = call_api(features)

        with placeholder.container():

            st.write(f"Transaction {i+1}")

            if "fraud_probability" in result:

                prob = result["fraud_probability"]
                pred = result["prediction"]

                st.metric("Fraud Probability", f"{prob:.3f}")

                if "FRAUD" in pred:
                    st.error(pred)
                else:
                    st.success(pred)

                st.progress(prob)

            else:
                st.warning("API error")

        time.sleep(1)