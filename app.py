import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Fraud Detection System",
    layout="wide",
    page_icon="💳"
)

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    with open("fraud_model_final.pkl", "rb") as f:
        data = pickle.load(f)
    return data["model"], data["threshold"]

model, threshold = load_model()

# =========================================================
# PREDICT FUNCTION
# =========================================================
@st.cache_data
def predict(features):
    arr = np.array(features).reshape(1, -1)
    prob = model.predict_proba(arr)[0][1]
    pred = "FRAUD" if prob > threshold else "LEGIT"
    return float(prob), pred

# =========================================================
# RISK LOGIC
# =========================================================
def get_risk(prob):
    if prob < 0.3:
        return "LOW", "#22c55e"
    elif prob < 0.7:
        return "MEDIUM", "#f59e0b"
    else:
        return "HIGH", "#ef4444"

# =========================================================
# UPDATED DEMO (better separation)
# =========================================================
def demo_low():
    return [20, 12000] + [0.05]*28

def demo_medium():
    return [300, 45000] + [0.6]*28

def demo_high():
    return [950, 90000] + [-2.0]*28

# =========================================================
# UI STYLE (UNCHANGED THEME)
# =========================================================
st.markdown("""
<style>
.block-container {padding-top: 2rem;}

.card {
    background: #111827;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #1f2937;
    color: white;
    text-align: center;
}

.risk {
    padding: 10px 14px;
    border-radius: 10px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.title("Fraud Detection System")
st.caption("Credit Card Fraud Risk Intelligence Engine")

# =========================================================
# SIDEBAR
# =========================================================
menu = st.sidebar.radio("Navigation", ["Dashboard", "Single Prediction", "Batch Processing"])

st.sidebar.markdown("---")
st.sidebar.markdown("""
### Model Info
XGBoost Classifier  
30 Features (Amount, Time, V1–V28)
""")

# =========================================================
# DASHBOARD
# =========================================================
if menu == "Dashboard":

    st.header("System Overview")

    # ================= STATS =================
    try:
        df_log = pd.read_csv("fraud_logs.csv")

        total = len(df_log)
        fraud = (df_log["Prediction"] == "FRAUD").sum()
        legit = (df_log["Prediction"] == "LEGIT").sum()
        avg_risk = df_log["Probability"].mean()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total", total)
        c2.metric("Fraud", fraud)
        c3.metric("Legit", legit)
        c4.metric("Avg Risk", f"{avg_risk:.3f}")

        # ================= LOG GRAPH =================
        df_log["Index"] = range(len(df_log))

        fig = px.line(
            df_log,
            x="Index",
            y="Probability",
            title="Fraud Risk Trend"
        )

        st.plotly_chart(fig, use_container_width=True)

    except:
        st.info("No logs available yet")

    # ================= MODEL INFO BUTTON =================
    if st.button("Model Information"):
        st.info("""
        XGBoost Classifier  
        30 features used  
        Detects credit card fraud using anomaly patterns  
        Output: Fraud probability + risk level
        """)

# =========================================================
# SINGLE PREDICTION
# =========================================================
elif menu == "Single Prediction":

    st.header("Transaction Analysis")

    # ONLY ONE DEMO BUTTON (kept)
    if st.button("Load Demo Transaction"):
        st.session_state.demo = demo_medium()

    if "demo" not in st.session_state:
        st.session_state.demo = [0]*30

    demo = st.session_state.demo

    features = []

    col1, col2, col3 = st.columns(3)

    with col1:
        features.append(st.number_input("Amount", value=float(demo[0])))
        features.append(st.number_input("Time", value=float(demo[1])))
        for i in range(1, 9):
            features.append(st.number_input(f"V{i}", value=float(demo[i+1])))

    with col2:
        for i in range(9, 19):
            features.append(st.number_input(f"V{i}", value=float(demo[i+1])))

    with col3:
        for i in range(19, 29):
            features.append(st.number_input(f"V{i}", value=float(demo[i+1])))

    if st.button("Run Prediction"):

        prob, pred = predict(features)
        risk, color = get_risk(prob)

        colA, colB = st.columns(2)

        with colA:
            st.metric("Fraud Probability", round(prob, 4))
            st.progress(prob)

            st.markdown(
                f"<div class='risk' style='background:{color};color:white;'>Risk Level: {risk}</div>",
                unsafe_allow_html=True
            )

        with colB:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Risk Score"},
                gauge={"axis": {"range": [0, 100]}}
            ))
            st.plotly_chart(fig, use_container_width=True)

# =========================================================
# BATCH PROCESSING (UPDATED ONLY HERE)
# =========================================================
elif menu == "Batch Processing":

    st.header("Bulk Transaction Analysis")

    file = st.file_uploader("Upload CSV File", type="csv")

    if file:

        df = pd.read_csv(file)

        if df.shape[1] > 30:
            df = df.iloc[:, :30]

        if df.shape[1] < 30:
            st.error("Dataset must have exactly 30 features")
            st.stop()

        # ================= NEW: SESSION STATE CONTROL =================
        if "batch_df" not in st.session_state:
            st.session_state.batch_df = df

        colA, colB = st.columns(2)

        # ================= RANDOM 1500 SAMPLE =================
        with colA:
            if st.button("Random 1500 Records"):
                st.session_state.batch_df = df.sample(
                    n=min(1500, len(df)),
                    random_state=np.random.randint(0, 9999)
                )

        # ================= REFRESH SAMPLE =================
        with colB:
            if st.button("Refresh Sample"):
                st.session_state.batch_df = df.sample(
                    n=min(1500, len(df)),
                    random_state=np.random.randint(0, 9999)
                )

        df = st.session_state.batch_df

        st.success(f"Processing {len(df)} records")

        st.dataframe(df.head())

        if st.button("Run Batch Prediction"):

            probs, preds = [], []

            progress = st.progress(0)

            for i in range(len(df)):

                try:
                    prob, pred = predict(df.iloc[i].values.tolist())
                except:
                    prob, pred = 0.0, "ERROR"

                probs.append(prob)
                preds.append(pred)

                progress.progress((i + 1) / len(df))

            df["Probability"] = probs
            df["Prediction"] = preds

            st.success("Completed")

            col1, col2 = st.columns(2)

            with col1:
                fig = px.histogram(df, x="Probability", title="Risk Distribution")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                pie = df["Prediction"].value_counts().reset_index()
                pie.columns = ["Type", "Count"]

                fig2 = px.pie(pie, names="Type", values="Count", title="Fraud vs Legit")
                st.plotly_chart(fig2, use_container_width=True)

            st.dataframe(df.head(20))

            st.download_button(
                "Download Results",
                df.to_csv(index=False),
                file_name="fraud_results.csv"
            )