import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go

# 🔗 CHANGE AFTER DEPLOY
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Aura Fraud Guard", layout="wide")

# =========================================================
# ⚡ CACHE API CALLS (IMPORTANT)
# =========================================================
@st.cache_data(ttl=60)
def cached_api_call(features_tuple):
    try:
        r = requests.post(API_URL, json={"features": list(features_tuple)})
        return r.json()
    except:
        return {"error": "API not reachable"}

def call_api(features):
    return cached_api_call(tuple(features))


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("💳 Aura Fraud Guard")
menu = st.sidebar.radio("Navigation", ["Dashboard", "Manual Audit", "Batch Scan"])


# =========================================================
# HEADER
# =========================================================
st.title(menu)

with st.expander("📘 How to Use"):
    st.markdown("""
1. Use **Manual Audit** for single transaction  
2. Use **Batch Scan** for CSV analysis  
3. Use **Dashboard** to view trends  

💡 Tips:
- High Amount → higher risk  
- V14 negative → strong fraud signal  
""")


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
        st.info("No data yet. Run predictions first.")


# =========================================================
# 🔍 MANUAL AUDIT
# =========================================================
elif menu == "Manual Audit":

    st.subheader("🔍 Single Transaction Check")

    # Quick presets
    colA, colB = st.columns(2)

    with colA:
        if st.button("⚡ Fraud Example"):
            st.session_state["preset"] = "fraud"

    with colB:
        if st.button("✅ Legit Example"):
            st.session_state["preset"] = "legit"

    preset = st.session_state.get("preset", None)

    def get_val(default, fraud_val=None):
        if preset == "fraud" and fraud_val is not None:
            return fraud_val
        return default

    features = []

    col1, col2, col3 = st.columns(3)

    with col1:
        features.append(st.number_input(
            "Amount 💰", value=get_val(0.0, 2500.0),
            help="Transaction amount"
        ))

        features.append(st.number_input(
            "Time ⏱️", value=get_val(0.0, 10000.0),
            help="Time from first transaction"
        ))

        for i in range(1, 9):
            features.append(st.number_input(f"V{i}", value=0.0))

    with col2:
        for i in range(9, 19):
            if i == 14:
                features.append(st.number_input(
                    "V14 🔥 (Fraud Sensitive)",
                    value=get_val(0.0, -10.0),
                    help="Strong fraud indicator"
                ))
            else:
                features.append(st.number_input(f"V{i}", value=0.0))

    with col3:
        for i in range(19, 29):
            features.append(st.number_input(f"V{i}", value=0.0))

    if st.button("🚀 Predict"):

        if len(features) != 30:
            st.error("Invalid input size")
            st.stop()

        result = call_api(features)

        if "error" in result:
            st.error(result["error"])
        else:
            prob = result["fraud_probability"]
            pred = result["prediction"]

            st.metric("Fraud Probability", f"{prob:.4f}")
            st.progress(prob)

            # Risk level
            if prob < 0.3:
                st.success("🟢 Low Risk")
            elif prob < 0.7:
                st.warning("🟡 Medium Risk")
            else:
                st.error("🔴 High Risk")

            if "FRAUD" in pred:
                st.error(pred)
            else:
                st.success(pred)

            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={'text': "Fraud Risk %"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'steps': [
                        {'range': [0, 30], 'color': "green"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"},
                    ],
                }
            ))
            st.plotly_chart(fig, use_container_width=True)


# =========================================================
# 📂 BATCH SCAN
# =========================================================
elif menu == "Batch Scan":

    st.subheader("📂 Bulk Fraud Detection")

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

            st.success("Detection Completed ✅")

            fig = px.histogram(df, x="Fraud Probability")
            st.plotly_chart(fig)

            st.dataframe(df.head())

            st.download_button(
                "⬇ Download Results",
                df.to_csv(index=False),
                "results.csv"
            )