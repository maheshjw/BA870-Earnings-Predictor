# Earnings Surprise & Stock Return Predictor
### BA870 / AC820 — Financial Analytics | Boston University Questrom | Spring 2026
**Author:** Mahesh Wadhokar

## What it does
- Predicts whether a company will beat or miss analyst EPS consensus estimates
- Forecasts the 3-day Cumulative Abnormal Return (CAR) around earnings announcements
- Users enter any stock ticker and get a live beat/miss forecast + historical scorecard

## Data Sources
- WRDS IBES (1990–2024) — EPS estimates & actuals
- WRDS CRSP (1990–2024) — daily stock returns
- Compustat — firm fundamentals (size, leverage, ROE)
- yfinance — real-time data

## Models
- XGBoost — beat/miss classification
- Logistic Regression — beat/miss baseline
- Random Forest — 3-day CAR prediction
- OLS — CAR baseline

## How to run
pip install -r requirements.txt
streamlit run app.py
