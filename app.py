import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import json
import plotly.graph_objects as go
import plotly.express as px
import time

# ── Page Config ───────────────────────────────────────────────────────────────
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

# ── Ticker lookup — tries multiple variations ─────────────────────────────────
def find_ticker_in_history(ticker, df):
    # 1. Exact match on ticker column
    hist = df[df['ticker'] == ticker]
    if len(hist) > 0:
        return hist, ticker
    # 2. oftic column (official ticker used in IBES)
    if 'oftic' in df.columns:
        hist = df[df['oftic'] == ticker]
        if len(hist) > 0:
            return hist, ticker
    # 3. Drop last letter  e.g. GOOGL → GOOG, BRK.B → BRK
    alt = ticker[:-1]
    hist = df[df['ticker'] == alt]
    if len(hist) > 0:
        return hist, alt
    # 4. Nothing found
    return pd.DataFrame(), ticker

# ── Safe yfinance fetch with retries & hard fallbacks ────────────────────────
def fetch_yfinance_info(ticker):
    """Returns (info_dict, warning_message_or_None)."""
    stock = yf.Ticker(ticker)
    for attempt in range(4):
        try:
            info = stock.info
            # yfinance sometimes returns a nearly-empty dict
            if info and len(info) > 5:
                return info, None
        except Exception as e:
            err = str(e)
            if attempt < 3:
                wait = (attempt + 1) * 6   # 6 s, 12 s, 18 s
                time.sleep(wait)
            else:
                return {}, f"Yahoo Finance unavailable ({err[:80]}). Using dataset averages for live features."
    return {}, "Yahoo Finance returned empty data. Using dataset averages for live features."

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("🔍 Stock Input")
ticker = st.sidebar.text_input(
    "Enter Ticker Symbol",
    value="AAPL",
    help="Best results: AAPL, AMGN, AMD, ADBE, BA, BAX, BMY, AXP, AGN, BDX"
).upper().strip()

st.sidebar.markdown("**Tickers with full history (135+ quarters):**")
st.sidebar.code("AAPL  AMGN  AMD\nADBE  BA    BAX\nBMY   AXP   AGN\nAVT   ADI   BDX")

st.sidebar.markdown("**Works but no history (model uses averages):**")
st.sidebar.code("TSLA  MSFT  GOOGL\nJPM   WMT   META")

predict_btn = st.sidebar.button("🚀 Predict", use_container_width=True)

st.sidebar.divider()
st.sidebar.markdown("### How it works")
st.sidebar.markdown("""
1. Enter a stock ticker
2. App fetches latest fundamentals via yfinance
3. XGBoost predicts beat/miss probability
4. Random Forest predicts 3-day CAR
5. Historical scorecard from WRDS IBES + CRSP
""")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_predictor, tab_methodology = st.tabs(["📈 Predictor", "📖 Methodology"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab_predictor:

    if predict_btn:
        with st.spinner(f'Fetching data for {ticker}...'):
            try:
                # ── 1. Live data ──────────────────────────────────────────
                info, yf_warning = fetch_yfinance_info(ticker)
                if yf_warning:
                    st.warning(f"⚠️ {yf_warning}")

                # ── 2. Historical earnings data ───────────────────────────
                hist, matched_ticker = find_ticker_in_history(ticker, df_history)
                has_history = len(hist) >= 2

                if len(hist) > 0:
                    hist = hist.sort_values('anndats_act')
                    if matched_ticker != ticker:
                        st.info(f"ℹ️ Showing history for **{matched_ticker}** (closest match for {ticker} in our dataset)")
                else:
                    st.info(f"ℹ️ **{ticker}** is not in our WRDS training dataset (1990–2024). "
                            f"Predictions still run using dataset-wide averages for historical features "
                            f"and live yfinance data for firm fundamentals. "
                            f"Tickers with full history: AAPL, AMGN, AMD, ADBE, BA, BAX, BMY, AXP.")

                # ── 3. Build features ─────────────────────────────────────
                if not has_history:
                    prior_surprise  = df_history['surprise'].mean()
                    prior_beat      = df_history['beat'].mean()
                    avg_surprise_4q = df_history['surprise'].mean()
                    prior_car       = df_history['car_3day'].mean()
                    beat_streak_val = 0
                    dispersion      = df_history['dispersion'].mean() if 'dispersion' in df_history.columns else 0.05
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

                # Live fundamentals — fall back to dataset averages if yfinance failed
                log_assets = np.log(max(info.get('totalAssets',
                                   df_history['log_assets'].apply(np.exp).median()
                                   if 'log_assets' in df_history.columns else 1e10), 1))
                leverage   = min(max(info.get('debtToEquity', 50) / 100, -5), 5)
                roe        = min(max(info.get('returnOnEquity', 0.1), -5), 5)
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

                # ── 4. Results header ─────────────────────────────────────
                company_name = info.get('longName', ticker)
                st.subheader(f"Results for **{ticker}** — {company_name}")

                col1, col2, col3 = st.columns(3)
                col1.metric("Beat Probability (XGBoost)", f"{beat_prob:.1%}",
                            delta="BEAT" if beat_prob >= 0.5 else "MISS")
                col2.metric("Predicted 3-Day CAR", f"{car_pred:.2%}")
                col3.metric("Beat Probability (Logistic)", f"{lr_prob:.1%}")

                st.divider()

                # ── 5. Gauge + CAR ────────────────────────────────────────
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

                    # ── FIX: clamp y-axis to prediction, not error bars ──
                    # car_pred is often ~0.5% but car_std ~10%,
                    # so error bars blow out the axis and bar disappears.
                    pad  = max(abs(car_pred) * 3, 0.02)   # min ±2%
                    y_lo = min(car_pred - pad, -pad)
                    y_hi = max(car_pred + pad,  pad)

                    fig_car = go.Figure()
                    fig_car.add_trace(go.Bar(
                        x=[ticker],
                        y=[car_pred],
                        marker_color='green' if car_pred > 0 else 'red',
                        width=0.4,
                        name='Predicted CAR',
                        text=f"{car_pred:.2%}",
                        textposition='outside',
                    ))
                    # ±1σ as thin reference lines (don't break axis scale)
                    fig_car.add_hline(y= car_std, line_dash='dot',
                                      line_color='rgba(150,150,150,0.5)',
                                      annotation_text=f'+1σ  {car_std:.1%}',
                                      annotation_position='right',
                                      annotation_font_size=10)
                    fig_car.add_hline(y=-car_std, line_dash='dot',
                                      line_color='rgba(150,150,150,0.5)',
                                      annotation_text=f'-1σ  {-car_std:.1%}',
                                      annotation_position='right',
                                      annotation_font_size=10)
                    fig_car.add_hline(y=0, line_dash='dash', line_color='gray')
                    fig_car.update_layout(
                        yaxis_title='3-Day Abnormal Return',
                        yaxis=dict(range=[y_lo, y_hi], tickformat='.1%'),
                        height=300,
                        showlegend=False,
                    )
                    st.plotly_chart(fig_car, use_container_width=True)

                st.divider()

                # ── 6. Scorecard + Feature Importance ─────────────────────
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
                        scorecard['Beat']      = scorecard['Beat'].map({1: '✅', 0: '❌'})
                        st.dataframe(scorecard, use_container_width=True, hide_index=True)
                    else:
                        st.info("No historical data for this ticker in our dataset — scorecard unavailable. "
                                "Try: AAPL, AMGN, AMD, ADBE, BA, BAX, BMY, AXP")

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

                # ── 7. Scatter plot ───────────────────────────────────────
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
                            'beat':     'Beat'
                        },
                        title=f'{matched_ticker} — Historical Surprise vs CAR',
                        hover_data=['anndats_act', 'actual', 'meanest']
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
                    st.info("Not enough historical data points for scatter plot (need > 5 quarters). "
                            "The gauge, CAR chart, and feature importance above are still valid.")

            except Exception as e:
                st.error(f"❌ Error for {ticker}: {str(e)}")
                st.exception(e)

    else:
        # ── Landing page ──────────────────────────────────────────────────
        st.markdown("""
        ## Welcome to the Earnings Surprise & Stock Return Predictor

        This app predicts whether a company will **beat or miss** analyst EPS estimates
        and forecasts the **3-day abnormal stock return** around the earnings announcement.

        ### How to use:
        1. Enter a stock ticker in the sidebar (e.g. AAPL, AMGN, AMD)
        2. Click **🚀 Predict**
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
        - yfinance — live fundamentals
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
    st.markdown("*A full walkthrough of how this project was built — data collection, cleaning, feature engineering, model training, and the live app.*")
    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — DATA COLLECTION
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("📦 Phase 1 — Data Collection")

    p1a, p1b, p1c = st.columns(3)

    with p1a:
        st.markdown("#### IBES Data (WRDS)")
        st.markdown("""
        **1.67 million rows** of analyst EPS estimates downloaded from WRDS IBES.

        Each row = one analyst consensus estimate for one company for one quarter.

        **Key columns:**
        - `ticker` — stock symbol
        - `meanest` — consensus EPS estimate
        - `actual` — reported EPS
        - `anndats_act` — earnings announcement date
        """)

    with p1b:
        st.markdown("#### CRSP Data (WRDS)")
        st.markdown("""
        **67.5 million rows** of daily stock returns downloaded from WRDS CRSP.

        Filtered to only tickers appearing in IBES to reduce size.

        **Key columns:**
        - `permno` — CRSP stock identifier
        - `date` — trading date
        - `ret` — daily return

        S&P 500 market returns added via **yfinance** (`^GSPC`) as `vwretd`.
        """)

    with p1c:
        st.markdown("#### Compustat Data (WRDS)")
        st.markdown("""
        **443K rows** of annual company fundamentals from Compustat.

        Used as control variables for firm quality and size.

        **Key columns:**
        - `at` — total assets
        - `dltt` — long-term debt
        - `ceq` — common equity
        - `ni` — net income
        """)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2 — CLEANING & FEATURE ENGINEERING
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("🔧 Phase 2 — Cleaning & Feature Engineering")

    p2a, p2b = st.columns(2)

    with p2a:
        st.markdown("#### IBES Cleaning")
        st.markdown("""
        - Removed rows with missing actual EPS or consensus
        - Kept only **quarterly forecasts** (`fpi = 6`)
        - Kept only the **most recent estimate** before each announcement date
        - Computed **EPS Surprise:**
        """)
        st.code("Surprise = (Actual − Consensus) / |Consensus|", language=None)
        st.markdown("""
        - Created **Beat/Miss label:** `1` if actual ≥ consensus, `0` otherwise

        ---

        #### CRSP–IBES Link
        - Downloaded CRSP header file with `permno ↔ ticker` mapping
        - Linked IBES tickers to CRSP permnos via `oftic` (official ticker)
        - **591K rows linked successfully**
        """)

        st.markdown("#### Compustat Merge")
        st.markdown("""
        - Calculated `log_assets`, `leverage` (debt/equity), `roe` (net income / equity)
        - Matched to IBES via `oftic` ticker and **fiscal year**
        - **67.5% match rate** — unmatched rows use dataset averages
        """)

    with p2b:
        st.markdown("#### 3-Day CAR Computation")
        st.markdown("""
        For **each earnings announcement** in the linked dataset:

        1. Got **200 days** of stock returns *before* the announcement (estimation window)
        2. Ran **OLS regression:** stock return = α + β × market return
        3. Computed expected return on each event day using α̂ + β̂
        4. Abnormal return = actual return − expected return
        5. CAR = sum over days **−1, 0, +1**
        """)
        st.code(
            "AR_t  = R_stock_t − (α̂ + β̂ × R_market_t)\n"
            "CAR   = AR₋₁ + AR₀ + AR₊₁",
            language=None
        )
        st.markdown("""
        - Ran on **110K rows**, saved checkpoints every 10K rows to Drive
        - Result: **avg CAR = 0.11%** — financially sensible (small positive drift on announcements)

        ---

        #### Feature Engineering — 10 Features
        """)
        feat_table = pd.DataFrame({
            'Feature':     ['prior_surprise', 'prior_beat', 'avg_surprise_4q', 'dispersion',
                            'num_analysts', 'prior_car', 'beat_streak',
                            'log_assets', 'leverage', 'roe'],
            'Source':      ['IBES','IBES','IBES','IBES','IBES','CRSP','IBES',
                            'Compustat','Compustat','Compustat'],
            'What it measures': [
                "Last quarter's EPS surprise",
                "Did they beat last quarter?",
                "Rolling avg surprise last 4 quarters",
                "Analyst disagreement (std dev of estimates)",
                "Number of analysts covering the stock",
                "Stock reaction last quarter (CAR)",
                "Consecutive quarters of beating",
                "Firm size",
                "Debt ratio",
                "Return on equity",
            ],
        })
        st.dataframe(feat_table, use_container_width=True, hide_index=True)

        st.markdown("**Final dataset: 52,891 rows · 1,894 tickers · 1990–2024**")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 3 — MODEL TRAINING
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("🤖 Phase 3 — Model Training")

    p3a, p3b = st.columns(2)

    with p3a:
        st.markdown("#### Train / Test Split")
        st.markdown("""
        Time-based split to **prevent look-ahead bias** — no future data ever leaks into the training set:

        | Split | Period | Rows |
        |-------|--------|------|
        | **Train** | Pre-2019 | 33,121 |
        | **Test** | 2019–2024 | 19,770 |

        ---

        #### Classification — Predict Beat vs. Miss

        | Model | Accuracy | AUC |
        |-------|----------|-----|
        | Logistic Regression | 68.5% | 0.701 |
        | **XGBoost** | 68.4% | **0.713** |

        **AUC of 0.713** means the model correctly ranks a true beat above a true miss 71.3% of the time.
        Both beat random guessing (AUC = 0.50) by a significant margin.
        """)

    with p3b:
        st.markdown("#### Regression — Predict 3-Day CAR")
        st.markdown("""
        | Model | RMSE |
        |-------|------|
        | OLS | 0.1179 |
        | **Random Forest** | 0.1180 |

        RMSE is intentionally modest — 3-day abnormal returns are inherently noisy.
        The models capture the *systematic* component of the reaction, not idiosyncratic noise.

        ---

        #### Why these models?
        - **Logistic Regression** — interpretable baseline; checks whether non-linear models actually help
        - **XGBoost** — handles non-linear interactions, missing values, and class imbalance natively
        - **OLS** — linear CAR baseline; easy to interpret coefficients
        - **Random Forest** — captures non-linear surprise-to-return mapping and interaction effects
        """)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 4 — STREAMLIT APP
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("🚀 Phase 4 — Streamlit App")

    p4a, p4b = st.columns(2)

    with p4a:
        st.markdown("#### Live Data Flow")
        st.markdown("""
        When a user enters a ticker and clicks **Predict**:

        1. **yfinance API** called → fetches live fundamentals (totalAssets, debtToEquity, returnOnEquity, numberOfAnalystOpinions)
        2. **Historical dataset** queried → looks up prior earnings from WRDS IBES + CRSP
        3. **10-feature vector built** — combining live fundamentals + historical earnings signals
        4. **XGBoost** runs → outputs beat/miss probability
        5. **Random Forest** runs → outputs predicted 3-day CAR
        6. **All 6 visuals rendered** in the Predictor tab
        """)

        st.markdown("#### Deployment")
        st.markdown("""
        - Code hosted on **GitHub:** `maheshjw/BA870-Earnings-Predictor`
        - Live app deployed on **Streamlit Cloud**
        - Works on desktop and mobile
        - Models pre-trained and loaded at startup (`pickle` files)
        - Historical dataset loaded once and cached (`@st.cache_data`)
        """)

    with p4b:
        st.markdown("#### 6 Visual Components")
        st.markdown("""
        | # | Chart | What it shows |
        |---|-------|---------------|
        | 1 | **Beat/Miss Gauge** | Color-coded probability meter (0–100%) |
        | 2 | **CAR Bar Chart** | Predicted 3-day abnormal return with ±1σ reference lines |
        | 3 | **Metrics Row** | Beat prob (XGBoost), CAR, Beat prob (Logistic) |
        | 4 | **Historical Scorecard** | Last 8 quarters of actual EPS, surprise, beat/miss, CAR |
        | 5 | **Feature Importance** | XGBoost gain scores — which features drove the prediction |
        | 6 | **Scatter Plot** | Prior EPS surprise vs realized CAR, with current prediction overlaid as a ★ |
        """)

        st.markdown("#### Course Topics Used")
        st.markdown("""
        | Topic | Where used |
        |-------|-----------|
        | IBES data | Phase 1–2 |
        | CRSP returns | Phase 1–2 |
        | Compustat fundamentals | Phase 1–2 |
        | Market model regression | Phase 2 CAR computation |
        | Logistic regression | Phase 3 classification |
        | Streamlit | Phase 4 app |
        | yfinance API | Phase 1 market returns + live data |
        | Stock market efficiency | CAR interpretation |
        """)

    st.divider()

    # ── 3-Day CAR Deep Dive ───────────────────────────────────────────────────
    st.subheader("📐 3-Day CAR — Deep Dive")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("""
        **Why a 3-day window?**

        | Day | Rationale |
        |-----|-----------|
        | **Day −1** | Earnings can leak via pre-announcements, options activity, or analyst whispers |
        | **Day 0** | Official announcement — the primary price-discovery event |
        | **Day +1** | Market continues processing analyst revisions and earnings call commentary |

        A wider window (±5 days) risks contamination from unrelated news.
        A narrower window (day 0 only) misses pre-leakage and post-call reactions.
        """)
    with d2:
        st.markdown("""
        **Interpreting the predicted value:**
        - **CAR = +5%** → stock returned 5% *more* than market-model predicted — strong positive reaction
        - **CAR = −3%** → underperformed expected return by 3%
        - **CAR ≈ 0%** → announcement was already fully priced in

        **Why predicted CARs are small (often < 1%):**
        The model predicts the *expected* abnormal return given the input features.
        For heavily covered stocks like AAPL that beat consistently, the surprise is partially anticipated
        — so the systematic component is small. The ±1σ lines on the chart show how much
        wider the actual outcomes are around that expected value.
        """)

    st.divider()

    # ── Chart Guide ───────────────────────────────────────────────────────────
    st.subheader("📊 Chart Guide")

    with st.expander("🎯 Beat/Miss Probability Gauge", expanded=True):
        st.markdown("""
        XGBoost probability (0–100%) the company will beat analyst EPS consensus at its next announcement.
        - 🟢 **Green (60–100%)** — confident beat signal
        - 🟡 **Yellow (40–60%)** — uncertain; limited model conviction
        - 🔴 **Red (0–40%)** — leans toward a miss
        - **Delta** shows distance from the 50% decision boundary
        - **Logistic Regression** shown alongside — agreement between both models = stronger signal
        """)

    with st.expander("📊 Predicted 3-Day CAR Bar Chart"):
        st.markdown("""
        Random Forest point estimate for the 3-day cumulative abnormal return.
        - Green = positive expected abnormal return · Red = negative
        - **Value label** printed on the bar so it's always readable regardless of magnitude
        - **±1σ dotted lines** = historical spread of realized CARs (reference band, not the prediction range)
        - Y-axis is scaled to the prediction — not blown out by the ±10% std dev
        """)

    with st.expander("📋 Historical Accuracy Scorecard"):
        st.markdown("""
        Last 8 quarters of actual earnings outcomes from WRDS IBES + CRSP.

        | Column | Meaning |
        |--------|---------|
        | Date | Announcement date |
        | Consensus EPS | Mean analyst estimate |
        | Actual EPS | Reported EPS |
        | Surprise | (Actual − Consensus) / |Consensus| |
        | Beat | ✅ beat · ❌ missed |
        | 3-Day CAR | Realized abnormal return |

        Only available for the **1,894 tickers in our WRDS dataset**.
        TSLA, JPM, GOOGL, WMT are not in the dataset — models still run using dataset averages + live yfinance fundamentals.
        """)

    with st.expander("🔍 Feature Importance (XGBoost)"):
        st.markdown("""
        Each feature's **gain score** — average improvement in classification objective when that feature is used in a split.
        - **beat_streak** — most powerful; consecutive beaters tend to keep beating ("sandbagging" guidance)
        - **prior_surprise** — analysts under-correct; a big beat last quarter predicts another beat
        - **avg_surprise_4q** — sustained beat history reflects structural expectation-management advantages
        - **num_analysts** — heavy coverage = tighter consensus = harder to surprise
        - **dispersion** — high analyst disagreement = genuine pre-announcement uncertainty
        - **Fundamentals** (log_assets, leverage, roe) — firm-quality context
        """)

    with st.expander("📉 EPS Surprise vs 3-Day CAR Scatter"):
        st.markdown("""
        Each dot = one past earnings event · X = EPS surprise · Y = 3-day CAR · 🟢 beat · 🔴 miss

        **Blue star ★** = current prediction overlaid on historical cloud.

        | Quadrant | Surprise | CAR | Interpretation |
        |----------|----------|-----|----------------|
        | Top-right ↗ | Positive | Positive | Beat + rewarded — normal |
        | Bottom-left ↙ | Negative | Negative | Miss + punished — normal |
        | Top-left ↖ | Negative | Positive | Miss but stock rose — bad news priced in |
        | Bottom-right ↘ | Positive | Negative | Beat but fell — "buy the rumor, sell the news" |
        """)

    st.divider()

    # ── Sentiment & External Events ───────────────────────────────────────────
    st.subheader("📰 Sentiment, News & External Event Effects on CAR")
    st.markdown("Four documented external factors that affect both beat probability and CAR magnitude — and are important context for interpreting predictions:")

    se1, se2 = st.columns(2)
    with se1:
        st.markdown("""
        #### A. 📈 The Expectations Treadmill
        Positive pre-earnings media sentiment in the 30 days before an announcement is associated with
        **upward analyst estimate revisions**. When consensus rises, the beat hurdle rises too —
        explaining why stocks sometimes fall on strong earnings.

        > *Gap:* Model uses `dispersion` but not analyst revision *direction*.

        ---

        #### B. 🏛️ Macro Event Contamination
        Fed decisions, CPI prints, and geopolitical shocks within the **3-day CAR window**
        inflate/deflate measured abnormal returns independent of earnings.
        The market-model beta-adjustment removes market-wide moves but not idiosyncratic sector shocks.

        > *Gap:* CARs during macro event windows should be interpreted cautiously.
        """)
    with se2:
        st.markdown("""
        #### C. 🎙️ Management Guidance Tone
        **Negative forward guidance alongside a positive EPS beat** often produces a negative CAR —
        investors price the *outlook*, not the past quarter.
        A miss + strong guidance can produce a positive CAR.

        > *Gap:* Guidance language not yet captured.
        > Adding FinBERT-scored transcript tone is the highest-impact extension.

        ---

        #### D. 📉 Short Interest & IV Crush
        High short interest amplifies positive CARs on beats (short squeeze).
        High pre-earnings implied volatility (IV) often collapses after the announcement
        — the realized CAR is smaller than the implied move (IV crush), especially for mega-caps.

        > *Gap:* `dispersion` proxies uncertainty but not directional positioning.
        """)

    st.divider()
    with st.expander("🔮 Future Extensions"):
        st.markdown("""
        - **FinBERT guidance tone** — score earnings call transcripts for forward guidance sentiment
        - **Pre-announcement news sentiment** — Alpha Vantage News API or RavenPack 30-day signal
        - **Options IV** — term structure as pre-announcement uncertainty signal
        - **FINRA short interest** — directional positioning as CAR amplifier
        - **Fama-French 3-factor CAR** — industry-adjusted benchmark instead of single-factor market model
        - **Analyst revision momentum** — direction of estimate changes in 30 days before announcement
        - **Whisper numbers** — unofficial EPS expectations that often diverge from published consensus
        """)

    st.caption("BA870 / AC820 · Mahesh Wadhokar · Data: WRDS IBES, CRSP, Compustat · Live: yfinance API")
