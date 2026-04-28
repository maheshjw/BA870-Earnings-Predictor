import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Earnings Surprise Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Earnings Surprise & Stock Return Predictor")
st.markdown("*BA870 / AC820 — Mahesh Wadhokar*")
st.divider()

# ── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    lr      = pickle.load(open('model_lr.pkl', 'rb'))
    xgb_cls = pickle.load(open('model_xgb.pkl', 'rb'))
    ols     = pickle.load(open('model_ols.pkl', 'rb'))
    rf      = pickle.load(open('model_rf.pkl', 'rb'))
    scaler  = pickle.load(open('scaler.pkl', 'rb'))
    with open('feature_cols.json') as f:
        feature_cols = json.load(f)
    return lr, xgb_cls, ols, rf, scaler, feature_cols

lr, xgb_cls, ols, rf, scaler, FEATURE_COLS = load_models()

# ── Load Historical Data ──────────────────────────────────────────────────────
@st.cache_data
def load_history():
    df = pd.read_csv('earnings_dataset_clean.csv', parse_dates=['anndats_act'])
    return df

df_history = load_history()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("🔍 Stock Input")
ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL").upper()
predict_btn = st.sidebar.button("🚀 Predict", use_container_width=True)

st.sidebar.divider()
st.sidebar.markdown("### How it works")
st.sidebar.markdown("""
1. Enter a stock ticker
2. App fetches latest earnings data
3. ML model predicts beat/miss
4. Shows predicted 3-day return
""")

# ── Main Logic ────────────────────────────────────────────────────────────────
if predict_btn:
    with st.spinner(f'Fetching data for {ticker}...'):
        try:
            # Fetch live data from yfinance
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get earnings history
            earnings = stock.earnings_history
            if earnings is None or len(earnings) == 0:
                st.error(f"No earnings data found for {ticker}")
                st.stop()

            # Get historical earnings from our dataset
            hist = df_history[df_history['ticker'] == ticker].sort_values('anndats_act')

            # Build features from historical data
            if len(hist) < 2:
                st.warning(f"Limited history for {ticker} — using market averages for some features")
                prior_surprise  = df_history['surprise'].mean()
                prior_beat      = df_history['beat'].mean()
                avg_surprise_4q = df_history['surprise'].mean()
                prior_car       = df_history['car_3day'].mean()
                beat_streak_val = 0
            else:
                last = hist.iloc[-1]
                prior_surprise  = hist['surprise'].iloc[-1]
                prior_beat      = hist['beat'].iloc[-1]
                avg_surprise_4q = hist['surprise'].tail(4).mean()
                prior_car       = hist['car_3day'].iloc[-1]
                beat_streak_val = int(hist['beat'].tail(4).sum())

            # Get analyst estimates from yfinance
            try:
                analysts = stock.analyst_price_targets
                num_analysts = info.get('numberOfAnalystOpinions', 10)
                dispersion = 0.05  # default
            except:
                num_analysts = 10
                dispersion = 0.05

            # Get fundamentals
            log_assets = np.log(max(info.get('totalAssets', 1e9), 1))
            leverage   = info.get('debtToEquity', 0.5) / 100
            roe        = info.get('returnOnEquity', 0.1)

            # Build feature vector
            features = pd.DataFrame([{
                'prior_surprise':  prior_surprise,
                'prior_beat':      prior_beat,
                'avg_surprise_4q': avg_surprise_4q,
                'dispersion':      dispersion,
                'num_analysts':    num_analysts,
                'prior_car':       prior_car,
                'beat_streak':     beat_streak_val,
                'log_assets':      log_assets,
                'leverage':        leverage,
                'roe':             roe
            }])

            features_scaled = scaler.transform(features[FEATURE_COLS])

            # Predictions
            beat_prob   = xgb_cls.predict_proba(features[FEATURE_COLS])[0][1]
            beat_pred   = int(beat_prob >= 0.5)
            car_pred    = rf.predict(features[FEATURE_COLS])[0]
            lr_prob     = lr.predict_proba(features_scaled)[0][1]

            # ── Layout ───────────────────────────────────────────────────────
            st.subheader(f"Results for **{ticker}** — {info.get('longName', ticker)}")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Beat Probability (XGBoost)", f"{beat_prob:.1%}")
            col2.metric("Predicted 3-Day CAR", f"{car_pred:.2%}")
            col3.metric("Beat Probability (Logistic)", f"{lr_prob:.1%}")

            st.divider()

            # ── Row 1: Gauge + CAR Chart ──────────────────────────────────
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("#### 🎯 Beat/Miss Probability Gauge")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=beat_prob * 100,
                    title={'text': "Probability of Beating Estimates"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "green" if beat_prob >= 0.5 else "red"},
                        'steps': [
                            {'range': [0, 40],  'color': '#ffcccc'},
                            {'range': [40, 60], 'color': '#fff3cc'},
                            {'range': [60, 100],'color': '#ccffcc'}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)

            with c2:
                st.markdown("#### 📊 Predicted 3-Day CAR")
                # Show CAR with confidence interval
                car_std = df_history['car_3day'].std()
                fig_car = go.Figure()
                fig_car.add_trace(go.Bar(
                    x=[ticker],
                    y=[car_pred],
                    error_y=dict(type='constant', value=car_std),
                    marker_color='green' if car_pred > 0 else 'red',
                    name='Predicted CAR'
                ))
                fig_car.add_hline(y=0, line_dash='dash', line_color='black')
                fig_car.update_layout(
                    yaxis_title='3-Day Abnormal Return',
                    height=300,
                    yaxis_tickformat='.2%'
                )
                st.plotly_chart(fig_car, use_container_width=True)

            st.divider()

            # ── Row 2: Scorecard + SHAP ───────────────────────────────────
            c3, c4 = st.columns(2)

            with c3:
                st.markdown("#### 📋 Historical Accuracy Scorecard")
                if len(hist) > 0:
                    scorecard = hist.tail(8)[['anndats_act', 'meanest', 
                                              'actual', 'surprise', 
                                              'beat', 'car_3day']].copy()
                    scorecard.columns = ['Date', 'Consensus EPS', 
                                         'Actual EPS', 'Surprise',
                                         'Beat', '3-Day CAR']
                    scorecard['Date'] = scorecard['Date'].dt.strftime('%Y-%m-%d')
                    scorecard['Surprise'] = scorecard['Surprise'].map('{:.1%}'.format)
                    scorecard['3-Day CAR'] = scorecard['3-Day CAR'].map('{:.2%}'.format)
                    scorecard['Beat'] = scorecard['Beat'].map({1: '✅', 0: '❌'})
                    
                    hit_rate = hist['beat'].mean()
                    avg_sur  = hist['surprise'].mean()
                    avg_car  = hist['car_3day'].mean()
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Hit Rate", f"{hit_rate:.1%}")
                    m2.metric("Avg Surprise", f"{avg_sur:.2%}")
                    m3.metric("Avg CAR", f"{avg_car:.2%}")
                    
                    st.dataframe(scorecard, use_container_width=True, hide_index=True)
                else:
                    st.info("No historical data found for this ticker")

            with c4:
                st.markdown("#### 🔍 Feature Importance (SHAP)")
                importances = xgb_cls.feature_importances_
                feat_df = pd.DataFrame({
                    'Feature': FEATURE_COLS,
                    'Importance': importances
                }).sort_values('Importance', ascending=True).tail(10)

                fig_shap = px.bar(
                    feat_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    color='Importance',
                    color_continuous_scale='RdYlGn'
                )
                fig_shap.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_shap, use_container_width=True)

            st.divider()

            # ── Row 3: Scatter Plot ───────────────────────────────────────
            st.markdown("#### 📉 Prior EPS Surprise vs Realized 3-Day CAR")
            if len(hist) > 5:
                fig_scatter = px.scatter(
                    hist,
                    x='surprise',
                    y='car_3day',
                    color='beat',
                    color_discrete_map={1: 'green', 0: 'red'},
                    labels={
                        'surprise': 'EPS Surprise',
                        'car_3day': '3-Day CAR',
                        'beat': 'Beat'
                    },
                    title=f'{ticker} — Historical Surprise vs CAR',
                    hover_data=['anndats_act', 'actual', 'meanest']
                )
                # Add current prediction as highlighted point
                fig_scatter.add_trace(go.Scatter(
                    x=[prior_surprise],
                    y=[car_pred],
                    mode='markers',
                    marker=dict(size=20, color='blue', 
                                symbol='star', line=dict(width=2)),
                    name='Current Prediction'
                ))
                fig_scatter.add_hline(y=0, line_dash='dash', line_color='gray')
                fig_scatter.add_vline(x=0, line_dash='dash', line_color='gray')
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("Not enough history for scatter plot")

        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {str(e)}")
            st.exception(e)

else:
    # Landing page
    st.markdown("""
    ## Welcome to the Earnings Surprise & Stock Return Predictor
    
    This app predicts whether a company will **beat or miss** analyst EPS estimates
    and forecasts the **3-day abnormal stock return** around the earnings announcement.
    
    ### How to use:
    1. Enter a stock ticker in the sidebar (e.g. AAPL, MSFT, GOOGL)
    2. Click **Predict**
    3. View the beat/miss forecast, predicted CAR, and historical scorecard
    
    ### Models used:
    - **XGBoost** — beat/miss classification
    - **Logistic Regression** — beat/miss baseline
    - **Random Forest** — CAR prediction
    - **OLS** — CAR baseline
    
    ### Data sources:
    - WRDS IBES (1990–2024) — EPS estimates & actuals
    - WRDS CRSP (1990–2024) — daily stock returns
    - Compustat — firm fundamentals
    - yfinance — live data
    """)
    
    # Show sample stats
    st.divider()
    st.markdown("### Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Observations", f"{len(df_history):,}")
    c2.metric("Unique Tickers", f"{df_history['ticker'].nunique():,}")
    c3.metric("Overall Beat Rate", f"{df_history['beat'].mean():.1%}")
    c4.metric("Avg 3-Day CAR", f"{df_history['car_3day'].mean():.2%}")