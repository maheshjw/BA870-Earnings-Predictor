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
    df = pd.read_csv('earnings_dataset_clean.csv')
    if 'anndats_act' not in df.columns:
        for col in df.columns:
            if 'ann' in col.lower() or 'date' in col.lower():
                df = df.rename(columns={col: 'anndats_act'})
                break
    df['anndats_act'] = pd.to_datetime(df['anndats_act'])
    return df

df_history = load_history()

# ── Helper: Find ticker in history ───────────────────────────────────────────
def find_ticker_in_history(ticker, df):
    hist = df[df['ticker'] == ticker]
    if len(hist) > 0:
        return hist
    if 'oftic' in df.columns:
        hist = df[df['oftic'] == ticker]
        if len(hist) > 0:
            return hist
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

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_predictor, tab_methodology = st.tabs(["📈 Predictor", "📖 Methodology"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTOR  (original, unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_predictor:

    if predict_btn:
        with st.spinner(f'Fetching data for {ticker}...'):
            try:
                stock = yf.Ticker(ticker)

                for attempt in range(3):
                    try:
                        info = stock.info
                        break
                    except Exception as e:
                        if attempt < 2:
                            st.warning('Rate limited — retrying in 5 seconds...')
                            time.sleep(5)
                        else:
                            raise e

                hist = find_ticker_in_history(ticker, df_history)
                if len(hist) > 0:
                    hist = hist.sort_values('anndats_act')

                if len(hist) < 2:
                    st.warning(f"Limited history for {ticker} — using market averages")
                    prior_surprise  = df_history['surprise'].mean()
                    prior_beat      = df_history['beat'].mean()
                    avg_surprise_4q = df_history['surprise'].mean()
                    prior_car       = df_history['car_3day'].mean()
                    beat_streak_val = 0
                    dispersion      = 0.05
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
                    dispersion      = hist['dispersion'].iloc[-1] \
                                      if 'dispersion' in hist.columns else 0.05

                log_assets   = np.log(max(info.get('totalAssets', 1e9), 1))
                leverage     = min(max(info.get('debtToEquity', 50) / 100, -5), 5)
                roe          = min(max(info.get('returnOnEquity', 0.1), -5), 5)
                num_analysts = info.get('numberOfAnalystOpinions', 10) or 10

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

                beat_prob = xgb_cls.predict_proba(features[FEATURE_COLS])[0][1]
                car_pred  = rf.predict(features[FEATURE_COLS])[0]
                lr_prob   = lr.predict_proba(features_scaled)[0][1]

                company_name = info.get('longName', ticker)
                st.subheader(f"Results for **{ticker}** — {company_name}")

                col1, col2, col3 = st.columns(3)
                col1.metric("Beat Probability (XGBoost)", f"{beat_prob:.1%}",
                            delta="BEAT" if beat_prob >= 0.5 else "MISS")
                col2.metric("Predicted 3-Day CAR", f"{car_pred:.2%}")
                col3.metric("Beat Probability (Logistic)", f"{lr_prob:.1%}")

                st.divider()

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

                    # Set y-axis range based on the prediction itself,
                    # not the error bars — otherwise a 0.46% bar on a
                    # ±10% axis is invisible.
                    pad = max(abs(car_pred) * 3, 0.02)   # at least ±2%
                    y_lo = min(car_pred - pad, -pad)
                    y_hi = max(car_pred + pad,  pad)

                    fig_car = go.Figure()
                    fig_car.add_trace(go.Bar(
                        x=[ticker],
                        y=[car_pred],
                        # error bars shown as a separate annotation so they
                        # don't blow out the axis scale
                        marker_color='green' if car_pred > 0 else 'red',
                        width=0.4,
                        name='Predicted CAR',
                        text=f"{car_pred:.2%}",
                        textposition='outside',
                        textfont=dict(size=14, color='white'),
                    ))
                    # ±1 std dev reference lines (thin, so they don't dominate)
                    fig_car.add_hline(y=car_std,  line_dash='dot',
                                      line_color='rgba(150,150,150,0.4)', line_width=1,
                                      annotation_text=f'+1σ {car_std:.1%}',
                                      annotation_position='right',
                                      annotation_font_size=10)
                    fig_car.add_hline(y=-car_std, line_dash='dot',
                                      line_color='rgba(150,150,150,0.4)', line_width=1,
                                      annotation_text=f'-1σ {-car_std:.1%}',
                                      annotation_position='right',
                                      annotation_font_size=10)
                    fig_car.add_hline(y=0, line_dash='dash',
                                      line_color='gray', line_width=1)
                    fig_car.update_layout(
                        yaxis_title='3-Day Abnormal Return',
                        yaxis=dict(range=[y_lo, y_hi], tickformat='.1%'),
                        height=300,
                        showlegend=False,
                    )
                    st.plotly_chart(fig_car, use_container_width=True)

                st.divider()

                c3, c4 = st.columns(2)

                with c3:
                    st.markdown("#### 📋 Historical Accuracy Scorecard")
                    if len(hist) > 0:
                        hit_rate = hist['beat'].mean()
                        avg_sur  = hist['surprise'].mean()
                        avg_car  = hist['car_3day'].mean()

                        m1, m2, m3 = st.columns(3)
                        m1.metric("Hit Rate",     f"{hit_rate:.1%}")
                        m2.metric("Avg Surprise", f"{avg_sur:.2%}")
                        m3.metric("Avg CAR",      f"{avg_car:.2%}")

                        scorecard = hist.tail(8)[['anndats_act','meanest',
                                                  'actual','surprise',
                                                  'beat','car_3day']].copy()
                        scorecard.columns = ['Date','Consensus EPS',
                                             'Actual EPS','Surprise',
                                             'Beat','3-Day CAR']
                        scorecard['Date']      = pd.to_datetime(
                            scorecard['Date']).dt.strftime('%Y-%m-%d')
                        scorecard['Surprise']  = scorecard['Surprise'].map('{:.1%}'.format)
                        scorecard['3-Day CAR'] = scorecard['3-Day CAR'].map('{:.2%}'.format)
                        scorecard['Beat']      = scorecard['Beat'].map({1:'✅', 0:'❌'})
                        st.dataframe(scorecard, use_container_width=True, hide_index=True)
                    else:
                        st.info("No historical data found for this ticker in our dataset")

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

                st.markdown("#### 📉 Prior EPS Surprise vs Realized 3-Day CAR")
                if len(hist) > 5:
                    fig_scatter = px.scatter(
                        hist,
                        x='surprise',
                        y='car_3day',
                        color='beat',
                        color_discrete_map={1:'green', 0:'red'},
                        labels={
                            'surprise': 'EPS Surprise',
                            'car_3day': '3-Day CAR',
                            'beat':     'Beat'
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


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — METHODOLOGY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_methodology:

    st.header("📖 Methodology")
    st.markdown("*How the Earnings Surprise & Stock Return Predictor works — data pipeline, model design, chart guide, and external event effects.*")
    st.divider()

    # ── 1. Problem Statement ──────────────────────────────────────────────────
    st.subheader("1. 🎯 What Problem Are We Solving?")
    st.markdown("""
    This app addresses two connected prediction problems in empirical finance:

    - **(1) Classification** — Will the company beat or miss analyst EPS consensus at the next earnings announcement?
    - **(2) Regression** — What 3-day cumulative abnormal return (CAR) will the stock produce around that announcement?

    Both tasks are framed as **out-of-sample forecasting** using only information available *before* the announcement date,
    evaluated with time-based cross-validation to prevent any look-ahead bias.
    """)
    st.divider()

    # ── 2. 3-Day CAR ──────────────────────────────────────────────────────────
    st.subheader("2. 📐 What is a 3-Day CAR?")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **Cumulative Abnormal Return (CAR)** isolates how much a stock moved *because of earnings news*,
        after stripping out what the broad market would have caused on the same days.

        **Step 1 — Estimate the market model** over a 252-trading-day window ending 10 days before the announcement:
        """)
        st.code("R_stock = α + β × R_market + ε", language=None)
        st.markdown("**Step 2 — Compute the daily Abnormal Return** on each event day:")
        st.code("AR_t  =  R_stock_t  −  (α̂ + β̂ × R_market_t)", language=None)
        st.markdown("**Step 3 — Sum over the 3-day window** (day −1, 0, +1):")
        st.code("CAR(−1, +1)  =  AR₋₁ + AR₀ + AR₊₁", language=None)

    with col_b:
        st.markdown("""
        **Why exactly 3 days?**

        | Day | Rationale |
        |-----|-----------|
        | **Day −1** | Earnings can leak via pre-announcements, options activity, or analyst whispers |
        | **Day 0** | Official announcement — the primary price-discovery event |
        | **Day +1** | Market continues processing analyst revisions and post-earnings call commentary |

        A wider window (±5 days) risks contamination from unrelated news.
        A narrower window (day 0 only) misses pre-leakage and post-call reactions.

        **Interpreting the value:**
        - **CAR = +5%** → stock returned 5% *more* than the market model predicted — strong positive earnings reaction
        - **CAR = −3%** → stock underperformed its expected return by 3% around the announcement
        - **CAR ≈ 0%** → announcement was already fully priced in by the market
        """)
    st.divider()

    # ── 3. Data Pipeline ──────────────────────────────────────────────────────
    st.subheader("3. 🗄️ Data Pipeline")
    st.markdown("""
    | # | Source | Period | What it provides |
    |---|--------|--------|-----------------|
    | 1 | **WRDS IBES** | 1990–2024 | Consensus mean EPS estimate, actual reported EPS, surprise magnitude, analyst count, and estimate dispersion per earnings event |
    | 2 | **WRDS CRSP** | 1990–2024 | Daily stock and market (S&P 500) returns — used to compute market-model CAR for every event in the training set |
    | 3 | **Compustat** | 1990–2024 | Firm-level fundamentals: total assets (size), debt-to-equity (leverage), return on equity — all lagged one quarter to prevent look-ahead bias |
    | 4 | **yfinance API** | Live | Latest firm fundamentals fetched at prediction time and mapped to the same feature schema as training data |

    **Key design rule:** All features are constructed from information dated *before* the announcement quarter.
    The CRSP market-model estimation window ends 10 days before each event date.
    """)
    st.divider()

    # ── 4. Features ───────────────────────────────────────────────────────────
    st.subheader("4. 🔧 Feature Engineering")
    feat_table = pd.DataFrame({
        'Feature':     ['prior_surprise', 'prior_beat', 'avg_surprise_4q', 'dispersion',
                        'num_analysts', 'prior_car', 'beat_streak', 'log_assets', 'leverage', 'roe'],
        'Source':      ['IBES','IBES','IBES','IBES','IBES','CRSP','IBES','Compustat','Compustat','Compustat'],
        'Lag':         ['1 quarter','1 quarter','4-quarter avg','Same quarter','Same quarter',
                        '1 quarter','Prior 4 quarters','1 quarter','1 quarter','1 quarter'],
        'Description': [
            'EPS surprise magnitude in the prior quarter (actual − consensus)',
            'Binary: did the firm beat estimates last quarter?',
            'Average EPS surprise over the prior 4 quarters — captures sustained tendency to beat',
            'Standard deviation of analyst EPS estimates — proxy for pre-announcement uncertainty',
            'Number of analysts covering the stock — proxy for information environment density',
            '3-day CAR from the prior earnings announcement',
            'Count of consecutive beats in the last 4 quarters (0–4)',
            'Natural log of total assets — size control variable',
            'Debt-to-equity ratio ÷ 100 — leverage control variable',
            'Return on equity — profitability control variable',
        ],
    })
    st.dataframe(feat_table, use_container_width=True, hide_index=True)
    st.divider()

    # ── 5. Models ─────────────────────────────────────────────────────────────
    st.subheader("5. 🤖 Models & Validation")
    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown("""
        #### Classification: Beat vs. Miss

        **XGBoost (primary)**
        Gradient-boosted tree ensemble tuned via 5-fold time-series cross-validation.
        Handles non-linear feature interactions, missing values, and class imbalance natively.
        → **AUC: 0.713**

        **Logistic Regression (baseline)**
        L2-regularised logistic model trained on scaled features.
        Interpretable probabilistic benchmark to validate whether the ensemble adds value.
        → **AUC: 0.701**

        *Validation:* Test periods never overlap training windows — strict temporal ordering enforced throughout.
        """)
    with mc2:
        st.markdown("""
        #### Regression: Predict 3-Day CAR

        **Random Forest (primary)**
        200-tree ensemble with max depth = 6.
        Captures the non-linear mapping from surprise magnitude to return and interaction effects.
        → **RMSE: 0.118**

        **OLS (baseline)**
        Standard linear regression on the same 10-feature set.
        Provides a stable, interpretable benchmark.
        → **RMSE: 0.118**

        *Note:* R² for CAR regression is intentionally modest — short-window returns are noisy.
        The model captures systematic patterns, not idiosyncratic noise.
        """)
    st.divider()

    # ── 6. Chart Guide ────────────────────────────────────────────────────────
    st.subheader("6. 📊 How to Read Each Chart")

    with st.expander("🎯 Beat/Miss Probability Gauge", expanded=True):
        st.markdown("""
        **What it shows:** The XGBoost model's probability (0–100%) that the company will beat analyst EPS consensus at its next earnings announcement.

        **Color zones:**
        - 🟢 **Green zone (60–100%)** — model is confident the stock will beat; historically associated with positive CARs
        - 🟡 **Yellow zone (40–60%)** — uncertain outcome; model has limited conviction either way
        - 🔴 **Red zone (0–40%)** — model leans toward a miss; historically associated with negative CARs

        **The delta (▲/▼ vs 50%)** tells you how far above or below the decision boundary the prediction sits.
        A reading of 73% means the model is 23 percentage points above the 50% toss-up line — a moderately strong signal.

        **Logistic Regression probability** is shown alongside as a sanity check. When both models agree (both above or both below 50%), the signal is more robust than when they diverge.
        """)

    with st.expander("📊 Predicted 3-Day CAR Bar Chart"):
        st.markdown("""
        **What it shows:** The Random Forest model's point estimate for the cumulative abnormal return over the 3-day window around the next earnings announcement.

        **How to read it:**
        - **Green bar (positive)** → model predicts the stock will outperform the market-model benchmark around earnings
        - **Red bar (negative)** → model predicts underperformance relative to expected return
        - **Error bars (±1 std dev)** show the historical spread of realized CARs for this ticker — a rough confidence band

        **Important caveat:** Predicted CARs tend to be small in magnitude (often under 1%) because the model predicts the *expected* abnormal return, not a tail outcome.
        The wide error bars reflect how noisy realized short-window returns are — even a small positive prediction is consistent with a wide range of actual outcomes.
        """)

    with st.expander("📋 Historical Accuracy Scorecard"):
        st.markdown("""
        **What it shows:** The last 8 quarters of actual earnings outcomes for the selected ticker, drawn from the WRDS IBES + CRSP training dataset.

        **Columns explained:**
        | Column | Meaning |
        |--------|---------|
        | Date | Earnings announcement date |
        | Consensus EPS | Mean analyst estimate at announcement time |
        | Actual EPS | Reported earnings per share |
        | Surprise | (Actual − Consensus) / |Consensus| — percentage beat or miss |
        | Beat | ✅ = beat consensus · ❌ = missed |
        | 3-Day CAR | Realized cumulative abnormal return over the ±1 day window |

        **How to use it:** Check the Hit Rate, Avg Surprise, and Avg CAR summary metrics above the table.
        A high hit rate (e.g. 75%+) combined with consistently positive CARs on beats supports trusting the current prediction.
        A volatile or inconsistent history should lower your confidence in the model's point estimate.
        """)

    with st.expander("🔍 Feature Importance (XGBoost)"):
        st.markdown("""
        **What it shows:** Each feature's **gain score** — the average improvement in the classification objective (log-loss) when that feature is used in a split, weighted by how often it appears across all trees.

        **Typical ranking and interpretation:**
        - **beat_streak** — the most powerful predictor; firms that have beaten 3–4 consecutive quarters tend to continue due to conservative guidance practices ("sandbagging")
        - **prior_surprise** — a large positive surprise last quarter predicts a beat this quarter; analysts systematically under-correct their estimates
        - **avg_surprise_4q** — firms with a sustained history of beating have embedded structural advantages in how they set expectations
        - **num_analysts** — heavily covered stocks have tighter consensus; surprises are harder to achieve with many eyes watching
        - **dispersion** — high analyst disagreement signals genuine pre-announcement uncertainty; the model uses this as a difficulty signal
        - **Fundamentals (leverage, ROE, log_assets)** — contribute less individually but provide firm-quality context that modulates the above signals
        """)

    with st.expander("📉 Prior EPS Surprise vs Realized 3-Day CAR Scatter"):
        st.markdown("""
        **What it shows:** A scatter plot of all historical earnings events for the selected ticker. Each dot is one past earnings announcement.

        **Axes:**
        - **X-axis (EPS Surprise):** How much the company beat or missed consensus — positive = beat, negative = miss
        - **Y-axis (3-Day CAR):** The realized abnormal return over the ±1 day window around that announcement

        **Color coding:** 🟢 Green = beat consensus · 🔴 Red = missed consensus

        **The blue star ★** is the *current prediction* — plotted at the prior quarter's surprise (X) and the predicted CAR (Y),
        overlaid on the historical cloud so you can see whether the forecast is consistent with past patterns.

        **Four quadrants to interpret:**
        | Quadrant | Surprise | CAR | What it means |
        |----------|----------|-----|---------------|
        | Top-right ↗ | Positive | Positive | Beat + rewarded — the normal, expected case |
        | Bottom-left ↙ | Negative | Negative | Miss + punished — the normal, expected case |
        | Top-left ↖ | Negative | Positive | Miss but stock rose — bad news was already priced in |
        | Bottom-right ↘ | Positive | Negative | Beat but stock fell — "buy the rumor, sell the news" |
        """)
    st.divider()

    # ── 7. Sentiment & External Events ───────────────────────────────────────
    st.subheader("7. 📰 Sentiment, News & External Event Effects on CAR")
    st.markdown("""
    Earnings surprises and stock price reactions don't happen in isolation.
    Several categories of external signals have well-documented effects on both
    the probability of a beat and the magnitude of the CAR — and are important context
    for interpreting the model's predictions.
    """)

    se1, se2 = st.columns(2)

    with se1:
        st.markdown("""
        #### A. 📈 The Expectations Treadmill
        Positive pre-earnings media coverage and analyst sentiment in the 30 days before an
        announcement is associated with **upward analyst estimate revisions**.
        When consensus rises, the hurdle for a beat rises too — making outperformance harder.

        This explains why stocks sometimes *fall* on strong earnings: the beat was already
        embedded in the elevated consensus. This is known as the **"expectations treadmill"** effect.

        > *Model implication:* The `dispersion` feature partially captures pre-announcement
        > uncertainty, but the model does not observe analyst revision direction.
        > A stock with rising consensus revisions faces a harder beat threshold
        > than the raw consensus level alone suggests.

        ---

        #### B. 🏛️ Macro Event Contamination
        Fed rate decisions, CPI prints, geopolitical shocks, and sector-wide regulatory
        announcements occurring **within the 3-day CAR window** can inflate or deflate
        measured abnormal returns independent of earnings quality.

        The market-model beta-adjustment removes market-wide moves, but **idiosyncratic
        sector shocks** (e.g. an FDA ruling on an industry peer) are not fully captured
        in a single-factor model.

        > *Model implication:* CARs measured during macro event windows should be interpreted
        > cautiously. A positive CAR during a Fed rate-cut day may reflect macro tailwinds
        > more than genuine earnings quality.
        """)

    with se2:
        st.markdown("""
        #### C. 🎙️ Management Guidance Tone (Earnings Call Effect)
        Academic research consistently shows that **negative forward guidance issued
        alongside a positive EPS beat** often produces a *negative* CAR — because investors
        price the forward outlook, not the backward-looking quarterly result.

        Conversely, a miss paired with strong forward guidance can produce a *positive* CAR
        as investors look past the current quarter.

        > *Current model limitation:* Guidance language and management tone are not captured.
        > Adding a **FinBERT-scored guidance tone feature** from earnings call transcripts
        > is the highest-impact single extension available for a future version.

        ---

        #### D. 📉 Short Interest & Options Positioning
        Elevated short interest before earnings can **amplify positive CARs** on a beat
        (forced short covering → price spike) and can dampen negative CARs on a miss
        if put buyers close positions rapidly.

        **Elevated implied volatility (IV)** before an announcement signals a large expected
        move — but the *realized* CAR is often smaller than the implied move once uncertainty
        resolves (IV crush), particularly for heavily covered mega-cap stocks.

        > *Current model:* The `dispersion` feature serves as a partial proxy for
        > uncertainty but does not capture directional short positioning.
        > Incorporating FINRA short interest data would improve tail-event CAR accuracy.
        """)

    st.divider()

    with st.expander("🔮 Future Extensions"):
        st.markdown("""
        - **NLP guidance tone** — FinBERT scoring of earnings call transcripts to capture forward guidance sentiment
        - **News sentiment score** — Alpha Vantage News API or RavenPack pre-announcement sentiment signal (30-day window)
        - **Options implied volatility** — IV term structure as a pre-announcement uncertainty and magnitude signal
        - **FINRA short interest** — directional short positioning as a CAR amplifier feature
        - **Sector-adjusted CAR** — replace single-factor market model with Fama-French 3-factor or industry-adjusted benchmark
        - **Whisper numbers** — unofficial EPS expectations that sometimes diverge significantly from published consensus
        - **Analyst revision momentum** — direction and magnitude of estimate changes in the 30 days before announcement
        """)

    st.caption("BA870 / AC820 · Mahesh Wadhokar · Data: WRDS IBES, CRSP, Compustat · Live: yfinance API")
