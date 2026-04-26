
# Fraud Detection System

ML-based fraud detection using XGBoost + FastAPI + Streamlit.

## Features
- Real-time fraud prediction
- Dashboard with charts
- Batch processing
- API-based architecture

## Run locally

### Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

### Frontend
cd frontend
pip install -r requirements.txt
streamlit run app.py
