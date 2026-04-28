import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import json
import plotly.graph_objects as go
import plotly.express as px
import time

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
    df = pd.read_csv('earnings_dataset_clean.csv', 
                     parse_dates=['anndats_act'])
    return df

df_history = load_history()

# ── Helper: Find ticker in history ───────────────────────────────────────────
def find_ticker_in_history(ticker, df):
    # Try exact match
    hist = df[df['ticker'] == ticker]
    if len(hist) > 0:
        return hist
    # Try oftic column
    if 'oftic' in df.columns:
        hist = df[df['oftic'] == ticker]
        if len(hist) > 0:
            return hist
    # Try without last letter (GOOGL → GOOG)
    if len(ticker) > 1:
        hist = df[df['ticker'] == ticker[:-1]]
        if len(hist) > 0:
            return hist
    return pd.DataFrame()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("🔍 Stock Input")
ticker = st.sidebar.text_input(
    "Enter Ticker Symbol",
    value="AAPL",
    help="Try: AAPL, AMGN, AMD, ADBE, BA, BAX, BMY, AXP"
).upper()

st.sidebar.markdown("**Tickers with full history:**")
st.sidebar.code("AAPL  AMGN  AMD\nADBE  BA    BAX\nBMY   AXP   AGN\nAVT   ADI   BDX")

predict_btn = st.sidebar.button("🚀 Predict", use_container_width=True)

st.sidebar.divider()
st.sidebar.markdown("### How it works")
st.sidebar.markdown("""
1. Enter a stock ticker
2. App fetches latest data via yfinance
3. ML model predicts beat/miss
4. Shows predicted 3-day return
""")

# ── Main Logic ────────────────────────────────────────────────────────────────
if predict_btn:
    with st.spinner(f'Fetching data for {ticker}...'):
        try:
            # Fetch live data from yfinance with retry
            stock = yf.Ticker(ticker)
            
            for attempt in range(3):
                try:
                    info = stock.info
                    break
                except Exception as e:
                    if attempt < 2:
                        st.warning(f'Rate limited — retrying in 5 seconds...')
                        time.sleep(5)
                    else:
                        raise e

            # Get historical earnings from our dataset
            hist = find_ticker_in_history(ticker, df_history)
            hist = hist.sort_values('anndats_act')

            # Build features
            if len(hist) < 2:
                st.warning(f"Limited history for {ticker} — using market averages")
                prior_surprise  = df_history['surprise'].mean()
                prior_beat      = df_history['beat'].mean()
                avg_surprise_4q = df_history['surprise'].mean()
                prior_car       = df_history['car_3day'].mean()
                beat_streak_val = 0
            else:
                prior_surprise  = hist['prior_surprise'].iloc[-1] \
                                  if 'prior_surprise' in hist.columns \
                                  else hist['surprise'].iloc[-2]
                prior_beat      = hist['prior_beat'].iloc[-1] \
                                  if 'prior_beat' in hist.columns \
                                  else float(hist['beat'].iloc[-2])
                avg_surprise_4q = hist['surprise'].tail(4).mean()
                prior_car       = hist['car_3day'].iloc[-1]
                beat_streak_val = int(hist['beat'].tail(4).sum())

            # Get fundamentals from yfinance
            log_assets = np.log(max(info.get('totalAssets', 1e9), 1))
            leverage   = min(max(info.get('debtToEquity', 50) / 100, -5), 5)
            roe        = min(max(info.get('returnOnEquity', 0.1), -5), 5)
            num_analysts = info.get('numberOfAnalystOpinions', 10) or 10
            dispersion = hist['dispersion'].iloc[-1] \
                         if len(hist) > 0 and 'dispersion' in hist.columns \
                         else 0.05

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
            beat_prob = xgb_cls.predict_proba(features[FEATURE_COLS])[0][1]
            car_pred  = rf.predict(features[FEATURE_COLS])[0]
            lr_prob   = lr.predict_proba(features_scaled)[0][1]

            # ── Results Header ────────────────────────────────────────────
            company_name = info.get('longName', ticker)
            st.subheader(f"Results for **{ticker}** — {company_name}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Beat Probability (XGBoost)", f"{beat_prob:.1%}",
                       delta="BEAT" if beat_prob >= 0.5 else "MISS")
            col2.metric("Predicted 3-Day CAR", f"{car_pred:.2%}")
            col3.metric("Beat Probability (Logistic)", f"{lr_prob:.1%}")

            st.divider()

            # ── Row 1: Gauge + CAR ────────────────────────────────────────
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("#### 🎯 Beat/Miss Probability")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=beat_prob * 100,
                    title={'text': "Probability of Beating Estimates (%)"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "green" if beat_prob >= 0.5 else "red"},
                        'steps': [
                            {'range': [0, 40],   'color': '#ffcccc'},
                            {'range': [40, 60],  'color': '#fff3cc'},
                            {'range': [60, 100], 'color': '#ccffcc'}
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
                    hit_rate = hist['beat'].mean()
                    avg_sur  = hist['surprise'].mean()
                    avg_car  = hist['car_3day'].mean()

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Hit Rate",    f"{hit_rate:.1%}")
                    m2.metric("Avg Surprise",f"{avg_sur:.2%}")
                    m3.metric("Avg CAR",     f"{avg_car:.2%}")

                    scorecard = hist.tail(8)[['anndats_act','meanest',
                                              'actual','surprise',
                                              'beat','car_3day']].copy()
                    scorecard.columns = ['Date','Consensus EPS',
                                         'Actual EPS','Surprise',
                                         'Beat','3-Day CAR']
                    scorecard['Date']     = pd.to_datetime(scorecard['Date']).dt.strftime('%Y-%m-%d')
                    scorecard['Surprise'] = scorecard['Surprise'].map('{:.1%}'.format)
                    scorecard['3-Day CAR']= scorecard['3-Day CAR'].map('{:.2%}'.format)
                    scorecard['Beat']     = scorecard['Beat'].map({1:'✅', 0:'❌'})
                    st.dataframe(scorecard, use_container_width=True, hide_index=True)
                else:
                    st.info("No historical data found for this ticker")

            with c4:
                st.markdown("#### 🔍 Feature Importance (XGBoost)")
                importances = xgb_cls.feature_importances_
                feat_df = pd.DataFrame({
                    'Feature':    FEATURE_COLS,
                    'Importance': importances
                }).sort_values('Importance', ascending=True)

                fig_shap = px.bar(
                    feat_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    color='Importance',
                    color_continuous_scale='RdYlGn',
                    title='Top Predictors'
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
                    color_discrete_map={1:'green', 0:'red'},
                    labels={
                        'surprise':  'EPS Surprise',
                        'car_3day':  '3-Day CAR',
                        'beat':      'Beat'
                    },
                    title=f'{ticker} — Historical Surprise vs CAR',
                    hover_data=['anndats_act','actual','meanest']
                )
                fig_scatter.add_trace(go.Scatter(
                    x=[prior_surprise],
                    y=[car_pred],
                    mode='markers',
                    marker=dict(size=20, color='blue',
                                symbol='star',
                                line=dict(width=2, color='white')),
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
    # ── Landing Page ──────────────────────────────────────────────────────
    st.markdown("""
    ## Welcome to the Earnings Surprise & Stock Return Predictor
    This app predicts whether a company will **beat or miss** analyst EPS estimates
    and forecasts the **3-day abnormal stock return** around the earnings announcement.

    ### How to use:
    1. Enter a stock ticker in the sidebar (e.g. AAPL, AMGN, AMD)
    2. Click **Predict**
    3. View the beat/miss forecast, predicted CAR, and historical scorecard

    ### Models used:
    - **XGBoost** — beat/miss classification (AUC: 0.713)
    - **Logistic Regression** — beat/miss baseline (AUC: 0.701)
    - **Random Forest** — CAR prediction (RMSE: 0.118)
    - **OLS** — CAR baseline (RMSE: 0.118)

    ### Data sources:
    - WRDS IBES (1990–2024) — EPS estimates & actuals
    - WRDS CRSP (1990–2024) — daily stock returns
    - Compustat — firm fundamentals
    - yfinance — live data
    """)

    st.divider()
    st.markdown("### Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Observations", f"{len(df_history):,}")
    c2.metric("Unique Tickers",     f"{df_history['ticker'].nunique():,}")
    c3.metric("Overall Beat Rate",  f"{df_history['beat'].mean():.1%}")
    c4.metric("Avg 3-Day CAR",      f"{df_history['car_3day'].mean():.2%}")