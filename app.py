import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection System",
    layout="wide"
)

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    with open("fraud_model_final.pkl", "rb") as f:
        data = pickle.load(f)
    return data["model"], data["threshold"]

model, threshold = load_model()

# ---------------------------------------------------------
# PREDICT FUNCTION
# ---------------------------------------------------------
@st.cache_data
def predict(features_tuple):
    arr = np.array(features_tuple).reshape(1, -1)
    prob = model.predict_proba(arr)[0][1]
    pred = "FRAUD" if prob > threshold else "LEGIT"
    return float(prob), pred

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.title("Fraud Detection System")

menu = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Single Prediction", "Batch Processing"]
)

st.title("Fraud Detection System")

with st.expander("How to Use"):
    st.markdown("""
- Dashboard: system overview  
- Single Prediction: analyze one transaction  
- Batch Processing: upload CSV file  

Model expects 30 features:
Amount, Time, V1–V28
""")

# =========================================================
# DASHBOARD (UPDATED PROFESSIONAL VERSION)
# =========================================================
if menu == "Dashboard":

    st.subheader("System Overview")

    st.markdown("""
### Fraud Detection System Overview

This system performs **offline transaction anomaly detection** using structured financial data.

It analyzes credit card transactions with **30 features**:
- Amount
- Time
- V1 – V28 (anonymized behavioral signals)

---

### Fraud Detection Scope

The model is designed for **credit card fraud detection**, including:

- Card-not-present fraud (online transactions)
- Stolen or compromised card usage
- Abnormal spending behavior patterns
- Unusual transaction timing or velocity

---

### Model Behavior

The system uses a **supervised machine learning model (XGBoost)** trained on historical labeled data.

It learns:
- Legitimate transaction patterns
- Fraudulent transaction patterns

Then outputs:
- Fraud probability score (0 to 1)
- Final classification (LEGIT / FRAUD)

---

### Important Note

This is a **risk scoring system**, not a rule-based banking system.
It predicts probability of fraud based on learned statistical patterns.
""")

    # -----------------------------------------------------
    # MODEL INFO CARD
    # -----------------------------------------------------
    st.markdown("### Model Information")

    st.info(f"""
Model Type: XGBoost Classifier  
Problem Type: Binary Classification  
Features: 30 (Amount, Time, V1–V28)  
Threshold: {threshold:.3f}  
Output: Fraud Probability + Decision  
""")

    # -----------------------------------------------------
    # LOG ANALYSIS
    # -----------------------------------------------------
    try:
        df = pd.read_csv("fraud_logs.csv")

        total = len(df)
        fraud = df["Prediction"].str.contains("FRAUD").sum()
        legit = total - fraud

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Transactions", total)
        c2.metric("Fraud Cases", fraud)
        c3.metric("Legit Cases", legit)

        df["Index"] = range(len(df))

        fig = px.line(df, x="Index", y="Probability", title="Fraud Risk Trend")
        st.plotly_chart(fig, use_container_width=True)

    except:
        st.info("No prediction logs available yet.")

# =========================================================
# SINGLE PREDICTION
# =========================================================
elif menu == "Single Prediction":

    st.subheader("Transaction Analysis")

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

    if st.button("Run Prediction"):

        prob, pred = predict(tuple(features))

        c1, c2 = st.columns(2)

        with c1:
            st.metric("Fraud Probability", round(prob, 4))
            st.progress(prob)

            if pred == "FRAUD":
                st.error("Fraud Detected")
            else:
                st.success("Transaction is Legitimate")

        with c2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Risk Score"},
                gauge={"axis": {"range": [0, 100]}}
            ))
            st.plotly_chart(fig, use_container_width=True)

        # LOG
        log = pd.DataFrame([{
            "Probability": prob,
            "Prediction": pred
        }])

        try:
            old = pd.read_csv("fraud_logs.csv")
            updated = pd.concat([old, log], ignore_index=True)
        except:
            updated = log

        updated.to_csv("fraud_logs.csv", index=False)

# =========================================================
# BATCH PROCESSING (OPTIMIZED)
# =========================================================
elif menu == "Batch Processing":

    st.subheader("Bulk Transaction Analysis")

    file = st.file_uploader("Upload CSV", type="csv")

    if file:

        df = pd.read_csv(file)

        st.write("Original Dataset Shape:", df.shape)

        # -----------------------------------------------------
        # LIMIT DATA FOR PERFORMANCE (IMPORTANT FIX)
        # -----------------------------------------------------
        MAX_SAMPLES = 1500

        if len(df) > MAX_SAMPLES:
            df = df.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)
            st.warning(f"Dataset too large. Randomly selected {MAX_SAMPLES} records for faster processing.")

        # Ensure only 30 features
        df = df.iloc[:, :30]

        st.dataframe(df.head())

        if st.button("Run Batch Prediction"):

            probabilities = []
            predictions = []

            progress = st.progress(0)

            for i, row in df.iterrows():

                prob, pred = predict(tuple(row.values))

                probabilities.append(prob)
                predictions.append(pred)

                progress.progress((i + 1) / len(df))

            df["Probability"] = probabilities
            df["Prediction"] = predictions

            st.success("Batch Processing Completed")

            # -----------------------------------------------------
            # VISUALIZATION
            # -----------------------------------------------------
            st.markdown("### Fraud Probability Distribution")

            fig = px.histogram(
                df,
                x="Probability",
                nbins=30,
                title="Risk Score Distribution"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Sample Results")
            st.dataframe(df.head(20))

            st.download_button(
                "Download Full Results",
                df.to_csv(index=False),
                "fraud_results.csv"
            )