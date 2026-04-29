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
    page_title="Earnings Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;500;600;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── App Header ── */
.app-header {
    display: flex;
    align-items: baseline;
    gap: 14px;
    margin-bottom: 0.25rem;
}
.app-title {
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: #f0f0f0;
    margin: 0;
}
.app-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #888;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 0 0 1.2rem 0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0d0d0d;
    border-right: 1px solid #1e1e1e;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

.sidebar-logo {
    font-family: 'Syne', sans-serif;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 0.04em;
    color: #f0f0f0;
    padding: 0 0 1.2rem 0;
    border-bottom: 1px solid #1e1e1e;
    margin-bottom: 1.2rem;
}

/* ── Metric Cards ── */
div[data-testid="metric-container"] {
    background: #111;
    border: 1px solid #1e1e1e;
    border-radius: 8px;
    padding: 1rem 1.2rem !important;
}
div[data-testid="metric-container"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #666 !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 26px !important;
    font-weight: 600 !important;
    color: #f0f0f0 !important;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
}

/* ── Section Headers ── */
.section-header {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #555;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e1e1e;
    margin-bottom: 1rem;
    margin-top: 0.5rem;
}

/* ── Scorecard Table ── */
div[data-testid="stDataFrame"] {
    border: 1px solid #1e1e1e !important;
    border-radius: 8px;
    overflow: hidden;
}

/* ── Expander ── */
details {
    border: 1px solid #1e1e1e !important;
    border-radius: 8px !important;
    background: #111 !important;
    margin-bottom: 6px;
}
summary {
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.04em;
    padding: 0.6rem 1rem !important;
    color: #aaa !important;
}

/* ── Ticker Tags ── */
.ticker-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 8px 0 16px 0;
}
.ticker-tag {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    background: #1a1a1a;
    color: #888;
    border: 1px solid #252525;
    border-radius: 4px;
    padding: 3px 8px;
    letter-spacing: 0.04em;
}

/* ── Info Box ── */
.info-box {
    background: #111;
    border: 1px solid #1e1e1e;
    border-left: 3px solid #3a7bd5;
    border-radius: 0 8px 8px 0;
    padding: 0.85rem 1rem;
    font-size: 13px;
    color: #aaa;
    line-height: 1.65;
    margin: 0.5rem 0 1rem 0;
}
.info-box strong { color: #ddd; }

/* ── Formula Box ── */
.formula {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    background: #0d0d0d;
    border: 1px solid #1e1e1e;
    border-radius: 6px;
    padding: 0.65rem 1rem;
    color: #7eb8f7;
    margin: 8px 0;
    letter-spacing: 0.02em;
}

/* ── How-it-works Steps ── */
.step {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    margin-bottom: 10px;
}
.step-num {
    min-width: 22px;
    height: 22px;
    border-radius: 50%;
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    color: #777;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-top: 2px;
}
.step-text { font-size: 13px; color: #888; line-height: 1.6; }

/* ── Result Header ── */
.result-company {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 600;
    color: #f0f0f0;
    margin: 0 0 2px 0;
}
.result-ticker {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: #555;
    letter-spacing: 0.08em;
}

/* ── Divider ── */
hr[data-testid="stDivider"] {
    border-color: #1e1e1e !important;
    margin: 1.25rem 0 !important;
}

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #555 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #f0f0f0 !important;
}
div[data-testid="stTabs"] > div > div {
    border-bottom: 1px solid #1e1e1e !important;
    gap: 0 !important;
}

/* ── Predict Button ── */
div[data-testid="stButton"] > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 0.06em !important;
    background: #f0f0f0 !important;
    color: #0d0d0d !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.2rem !important;
    transition: opacity 0.15s ease;
}
div[data-testid="stButton"] > button:hover {
    opacity: 0.88 !important;
}

/* ── Input ── */
div[data-testid="stTextInput"] input {
    font-family: 'DM Mono', monospace !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    background: #0d0d0d !important;
    border: 1px solid #252525 !important;
    color: #f0f0f0 !important;
    border-radius: 8px !important;
}
div[data-testid="stTextInput"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #555 !important;
}
</style>
""", unsafe_allow_html=True)

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

# ── Plotly Theme ──────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Sans, sans-serif', color='#888', size=12),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor='#1e1e1e', zerolinecolor='#2a2a2a', tickfont=dict(size=11)),
    yaxis=dict(gridcolor='#1e1e1e', zerolinecolor='#2a2a2a', tickfont=dict(size=11)),
)

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
with st.sidebar:
    st.markdown('<div class="sidebar-logo">📈 Earnings Predictor</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;color:#555;margin-bottom:8px;">Ticker Symbol</p>', unsafe_allow_html=True)
    ticker = st.text_input(
        label="Ticker Symbol",
        value="AAPL",
        label_visibility="collapsed",
        help="Try: AAPL, AMGN, AMD, ADBE, BA, BAX, BMY, AXP"
    ).upper()

    st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:10px;letter-spacing:0.1em;color:#444;margin:10px 0 6px 0;text-transform:uppercase;">Full history available</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="ticker-grid">
      <span class="ticker-tag">AAPL</span><span class="ticker-tag">AMGN</span>
      <span class="ticker-tag">AMD</span><span class="ticker-tag">ADBE</span>
      <span class="ticker-tag">BA</span><span class="ticker-tag">BAX</span>
      <span class="ticker-tag">BMY</span><span class="ticker-tag">AXP</span>
      <span class="ticker-tag">AGN</span><span class="ticker-tag">AVT</span>
      <span class="ticker-tag">ADI</span><span class="ticker-tag">BDX</span>
    </div>
    """, unsafe_allow_html=True)

    predict_btn = st.button("→ Run Prediction", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div class="step"><div class="step-num">1</div><div class="step-text">Enter a stock ticker symbol above</div></div>
    <div class="step"><div class="step-num">2</div><div class="step-text">Live fundamentals fetched via yfinance</div></div>
    <div class="step"><div class="step-num">3</div><div class="step-text">XGBoost predicts beat or miss probability</div></div>
    <div class="step"><div class="step-num">4</div><div class="step-text">Random Forest forecasts 3-day CAR</div></div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:9px;color:#333;letter-spacing:0.08em;">BA870 / AC820 · MAHESH WADHOKAR</p>', unsafe_allow_html=True)

# ── App Header ────────────────────────────────────────────────────────────────
st.markdown('<p class="app-title">Earnings Surprise & Stock Return Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">BA870 / AC820 · Mahesh Wadhokar · WRDS IBES + CRSP + Compustat + yfinance</p>', unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_predict, tab_overview, tab_method = st.tabs(["Predictor", "Dataset Overview", "Methodology"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    if predict_btn:
        with st.spinner(f'Fetching data for {ticker}…'):
            try:
                stock = yf.Ticker(ticker)
                for attempt in range(3):
                    try:
                        info = stock.info
                        break
                    except Exception as e:
                        if attempt < 2:
                            st.warning(f'Rate limited — retrying in 5 s…')
                            time.sleep(5)
                        else:
                            raise e

                hist = find_ticker_in_history(ticker, df_history)
                if len(hist) > 0:
                    hist = hist.sort_values('anndats_act')

                # ── Build Features ────────────────────────────────────────
                if len(hist) < 2:
                    st.warning(f"Limited history for {ticker} — using dataset averages for engineered features.")
                    prior_surprise  = df_history['surprise'].mean()
                    prior_beat      = df_history['beat'].mean()
                    avg_surprise_4q = df_history['surprise'].mean()
                    prior_car       = df_history['car_3day'].mean()
                    beat_streak_val = 0
                    dispersion      = 0.05
                else:
                    prior_surprise  = hist['prior_surprise'].iloc[-1] if 'prior_surprise' in hist.columns else hist['surprise'].iloc[-2]
                    prior_beat      = hist['prior_beat'].iloc[-1] if 'prior_beat' in hist.columns else float(hist['beat'].iloc[-2])
                    avg_surprise_4q = hist['surprise'].tail(4).mean()
                    prior_car       = hist['car_3day'].iloc[-1]
                    beat_streak_val = int(hist['beat'].tail(4).sum())
                    dispersion      = hist['dispersion'].iloc[-1] if 'dispersion' in hist.columns else 0.05

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

                # ── Result Header ─────────────────────────────────────────
                st.markdown(f'<p class="result-company">{company_name}</p><p class="result-ticker">{ticker} · {info.get("sector","—")} · {info.get("exchange","—")}</p>', unsafe_allow_html=True)
                st.divider()

                # ── Top KPIs ──────────────────────────────────────────────
                k1, k2, k3, k4 = st.columns(4)
                verdict = "BEAT" if beat_prob >= 0.5 else "MISS"
                k1.metric("Beat Probability", f"{beat_prob:.1%}", delta=verdict)
                k2.metric("Predicted 3-Day CAR", f"{car_pred:.2%}", delta="Abnormal Return")
                k3.metric("Logistic Baseline", f"{lr_prob:.1%}", delta="AUC 0.701")
                beat_vs_miss = "↑ Bullish" if beat_prob >= 0.5 and car_pred > 0 else ("↓ Bearish" if beat_prob < 0.5 else "⟷ Mixed")
                k4.metric("Signal Alignment", beat_vs_miss, delta=f"streak: {beat_streak_val}/4")

                st.divider()

                # ── Row 1: Gauge + CAR ────────────────────────────────────
                st.markdown('<div class="section-header">Model Output</div>', unsafe_allow_html=True)
                c1, c2 = st.columns(2)

                with c1:
                    gauge_color = "#22c55e" if beat_prob >= 0.5 else "#ef4444"
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=beat_prob * 100,
                        title={'text': "Beat Probability (%)", 'font': {'size': 13, 'color': '#888', 'family': 'DM Mono'}},
                        delta={'reference': 50, 'valueformat': '.1f'},
                        number={'font': {'size': 36, 'family': 'Syne', 'color': '#f0f0f0'}, 'suffix': '%'},
                        gauge={
                            'axis': {'range': [0, 100], 'tickcolor': '#333', 'tickwidth': 1, 'tickfont': {'size': 10, 'color': '#555'}},
                            'bar': {'color': gauge_color, 'thickness': 0.25},
                            'bgcolor': '#111',
                            'borderwidth': 0,
                            'steps': [
                                {'range': [0, 40],   'color': '#1a0a0a'},
                                {'range': [40, 60],  'color': '#131308'},
                                {'range': [60, 100], 'color': '#0a1a0a'}
                            ],
                            'threshold': {
                                'line': {'color': '#444', 'width': 2},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig_gauge.update_layout(**PLOT_LAYOUT, height=280)
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with c2:
                    car_std  = df_history['car_3day'].std()
                    bar_col  = "#22c55e" if car_pred > 0 else "#ef4444"
                    fig_car  = go.Figure()
                    fig_car.add_trace(go.Bar(
                        x=[ticker], y=[car_pred],
                        error_y=dict(type='constant', value=car_std, color='#555', thickness=1.5, width=8),
                        marker_color=bar_col,
                        marker_line_width=0,
                        width=0.4,
                        name='Predicted CAR'
                    ))
                    fig_car.add_hline(y=0, line_dash='dot', line_color='#333', line_width=1)
                    fig_car.update_layout(
                        **PLOT_LAYOUT,
                        title=dict(text="Predicted 3-Day CAR", font=dict(size=13, color='#888', family='DM Mono')),
                        yaxis_title=None,
                        height=280,
                        yaxis_tickformat='.1%',
                        showlegend=False,
                    )
                    st.plotly_chart(fig_car, use_container_width=True)

                st.divider()

                # ── Row 2: Scorecard + Feature Importance ─────────────────
                st.markdown('<div class="section-header">Historical Record & Model Internals</div>', unsafe_allow_html=True)
                c3, c4 = st.columns(2)

                with c3:
                    if len(hist) > 0:
                        hit_rate = hist['beat'].mean()
                        avg_sur  = hist['surprise'].mean()
                        avg_car  = hist['car_3day'].mean()
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Hit Rate",     f"{hit_rate:.1%}")
                        m2.metric("Avg Surprise", f"{avg_sur:.2%}")
                        m3.metric("Avg CAR",      f"{avg_car:.2%}")

                        scorecard = hist.tail(8)[['anndats_act','meanest','actual','surprise','beat','car_3day']].copy()
                        scorecard.columns = ['Date','Consensus EPS','Actual EPS','Surprise','Beat','3-Day CAR']
                        scorecard['Date']      = pd.to_datetime(scorecard['Date']).dt.strftime('%b %Y')
                        scorecard['Surprise']  = scorecard['Surprise'].map('{:.1%}'.format)
                        scorecard['3-Day CAR'] = scorecard['3-Day CAR'].map('{:.2%}'.format)
                        scorecard['Beat']      = scorecard['Beat'].map({1: '✓', 0: '✗'})
                        st.dataframe(scorecard, use_container_width=True, hide_index=True)
                    else:
                        st.info("No historical data found for this ticker in the dataset.")

                with c4:
                    importances = xgb_cls.feature_importances_
                    feat_df = pd.DataFrame({
                        'Feature':    FEATURE_COLS,
                        'Importance': importances
                    }).sort_values('Importance', ascending=True)
                    colors = ['#22c55e' if v > feat_df['Importance'].median() else '#2a6e3f' for v in feat_df['Importance']]
                    fig_imp = go.Figure(go.Bar(
                        x=feat_df['Importance'],
                        y=feat_df['Feature'],
                        orientation='h',
                        marker_color=colors,
                        marker_line_width=0,
                    ))
                    fig_imp.update_layout(
                        **PLOT_LAYOUT,
                        title=dict(text="XGBoost Feature Importance", font=dict(size=13, color='#888', family='DM Mono')),
                        height=320,
                        showlegend=False,
                        xaxis_title=None,
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)

                st.divider()

                # ── Row 3: Scatter ────────────────────────────────────────
                st.markdown('<div class="section-header">Prior Surprise vs Realized CAR</div>', unsafe_allow_html=True)
                if len(hist) > 5:
                    color_map = {1: '#22c55e', 0: '#ef4444'}
                    fig_scatter = px.scatter(
                        hist,
                        x='surprise', y='car_3day',
                        color='beat',
                        color_discrete_map=color_map,
                        labels={'surprise': 'EPS Surprise', 'car_3day': '3-Day CAR', 'beat': 'Beat'},
                        hover_data=['anndats_act', 'actual', 'meanest'],
                        opacity=0.7,
                    )
                    fig_scatter.add_trace(go.Scatter(
                        x=[prior_surprise], y=[car_pred],
                        mode='markers',
                        marker=dict(size=16, color='#facc15', symbol='star',
                                    line=dict(width=1.5, color='#0d0d0d')),
                        name='Current Prediction'
                    ))
                    fig_scatter.add_hline(y=0, line_dash='dot', line_color='#2a2a2a', line_width=1)
                    fig_scatter.add_vline(x=0, line_dash='dot', line_color='#2a2a2a', line_width=1)
                    fig_scatter.update_layout(
                        **PLOT_LAYOUT,
                        height=360,
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(size=11)),
                        xaxis_tickformat='.1%',
                        yaxis_tickformat='.1%',
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.info("Not enough historical data points for scatter plot (need > 5).")

                # ── Model Comparison Footer ───────────────────────────────
                st.divider()
                st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)
                comp_df = pd.DataFrame({
                    'Model':  ['XGBoost', 'Logistic Regression', 'Random Forest', 'OLS'],
                    'Task':   ['Beat/Miss', 'Beat/Miss', '3-Day CAR', '3-Day CAR'],
                    'Metric': ['AUC', 'AUC', 'RMSE', 'RMSE'],
                    'Score':  [0.713, 0.701, 0.118, 0.118],
                    'Output': [f'{beat_prob:.1%}', f'{lr_prob:.1%}', f'{car_pred:.2%}', '—'],
                })
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Error fetching data for **{ticker}**: {e}")
                st.exception(e)

    else:
        # ── Landing State ─────────────────────────────────────────────────
        st.markdown("""
        <div style="padding: 2.5rem 0 1rem 0;">
          <p style="font-family:'Syne',sans-serif;font-size:20px;font-weight:600;color:#f0f0f0;margin:0 0 6px 0;">
            Ready to predict.
          </p>
          <p style="font-family:'DM Sans',sans-serif;font-size:14px;color:#666;margin:0 0 2rem 0;max-width:520px;line-height:1.7;">
            Enter a ticker in the sidebar and click <strong style="color:#aaa">Run Prediction</strong> to get a beat/miss forecast,
            predicted 3-day cumulative abnormal return, and a historical accuracy scorecard.
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Models in use</div>', unsafe_allow_html=True)
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("XGBoost",  "AUC 0.713", delta="Beat classifier")
        mc2.metric("Logistic", "AUC 0.701", delta="Baseline")
        mc3.metric("Rand. Forest", "RMSE 0.118", delta="CAR regressor")
        mc4.metric("OLS",      "RMSE 0.118", delta="Baseline")

        st.divider()
        st.markdown('<div class="section-header">Data sources</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
          <strong>WRDS IBES (1990–2024)</strong> — Consensus EPS estimates, actual EPS, analyst count, and dispersion per earnings announcement.<br><br>
          <strong>WRDS CRSP (1990–2024)</strong> — Daily stock and market returns used to compute 3-day cumulative abnormal returns (CAR) via a market-model OLS regression.<br><br>
          <strong>Compustat</strong> — Firm fundamentals (total assets, debt/equity, ROE) lagged one quarter to prevent look-ahead bias.<br><br>
          <strong>yfinance</strong> — Live fundamentals and price data fetched at prediction time.
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATASET OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown('<div class="section-header">Training Dataset Summary</div>', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Total Observations", f"{len(df_history):,}")
    d2.metric("Unique Tickers",     f"{df_history['ticker'].nunique():,}")
    d3.metric("Overall Beat Rate",  f"{df_history['beat'].mean():.1%}")
    d4.metric("Avg 3-Day CAR",      f"{df_history['car_3day'].mean():.2%}")

    st.divider()

    # Beat rate over time
    st.markdown('<div class="section-header">Beat Rate & CAR Over Time</div>', unsafe_allow_html=True)
    time_df = df_history.groupby(df_history['anndats_act'].dt.year).agg(
        beat_rate=('beat', 'mean'),
        avg_car=('car_3day', 'mean'),
        n=('beat', 'count')
    ).reset_index().rename(columns={'anndats_act': 'Year'})

    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(
        x=time_df['Year'], y=time_df['beat_rate'],
        mode='lines', name='Beat Rate',
        line=dict(color='#22c55e', width=2),
        fill='tozeroy', fillcolor='rgba(34,197,94,0.06)'
    ))
    fig_time.add_trace(go.Scatter(
        x=time_df['Year'], y=time_df['avg_car'],
        mode='lines', name='Avg 3-Day CAR',
        line=dict(color='#3a7bd5', width=2),
        yaxis='y2'
    ))
    fig_time.update_layout(
        **PLOT_LAYOUT,
        height=300,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(size=11, color='#888')),
        yaxis=dict(tickformat='.0%', gridcolor='#1a1a1a', color='#555'),
        yaxis2=dict(tickformat='.1%', overlaying='y', side='right', showgrid=False, color='#555'),
    )
    st.plotly_chart(fig_time, use_container_width=True)

    st.divider()

    # CAR Distribution
    st.markdown('<div class="section-header">CAR Distribution</div>', unsafe_allow_html=True)
    fig_hist = go.Figure()
    beats = df_history[df_history['beat'] == 1]['car_3day']
    misses = df_history[df_history['beat'] == 0]['car_3day']
    fig_hist.add_trace(go.Histogram(
        x=beats, nbinsx=80, name='Beat',
        marker_color='rgba(34,197,94,0.6)', marker_line_width=0
    ))
    fig_hist.add_trace(go.Histogram(
        x=misses, nbinsx=80, name='Miss',
        marker_color='rgba(239,68,68,0.5)', marker_line_width=0
    ))
    fig_hist.update_layout(
        **PLOT_LAYOUT,
        barmode='overlay',
        height=260,
        xaxis_tickformat='.0%',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(size=11, color='#888')),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    # Top tickers by count
    st.markdown('<div class="section-header">Top 20 Tickers by Observation Count</div>', unsafe_allow_html=True)
    top_tickers = df_history.groupby('ticker').agg(
        n=('beat', 'count'),
        beat_rate=('beat', 'mean'),
        avg_car=('car_3day', 'mean')
    ).sort_values('n', ascending=False).head(20).reset_index()

    fig_top = go.Figure(go.Bar(
        x=top_tickers['ticker'],
        y=top_tickers['n'],
        marker_color='#1e3a5f',
        marker_line_width=0,
        text=top_tickers['beat_rate'].map('{:.0%}'.format),
        textposition='outside',
        textfont=dict(size=10, color='#555'),
    ))
    fig_top.update_layout(**PLOT_LAYOUT, height=260, yaxis_title=None, xaxis_title=None, showlegend=False)
    st.plotly_chart(fig_top, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — METHODOLOGY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_method:

    st.markdown('<div class="section-header">1. The Core Problem</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
      This app addresses two related prediction problems in empirical finance:<br><br>
      <strong>(1) Classification:</strong> Will the company's next reported EPS beat or miss analyst consensus estimates?<br>
      <strong>(2) Regression:</strong> What 3-day cumulative abnormal return (CAR) will the stock produce around that announcement?<br><br>
      Both are framed as out-of-sample forecasting tasks using only information available <em>before</em> the announcement date,
      evaluated with time-based cross-validation to prevent look-ahead bias.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── 3-Day CAR Explained ───────────────────────────────────────────────
    st.markdown('<div class="section-header">2. What is a 3-Day CAR?</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="info-box">
          <strong>Abnormal Return (AR)</strong> is the portion of a stock's return that cannot be
          explained by broad market movement. It isolates the firm-specific reaction to the earnings news.<br><br>
          A market-model OLS regression is estimated over a 252-trading-day window ending 10 days before the announcement,
          producing firm-specific alpha (α) and market beta (β):
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="formula">AR_t = R_t − (α̂ + β̂ × R_mkt_t)</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
          Where R_t is the stock daily return and R_mkt_t is the S&P 500 daily return.
          The 3-day CAR sums AR over the window surrounding the announcement:
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="formula">CAR(−1, +1) = AR₋₁ + AR₀ + AR₊₁</div>', unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="info-box">
          <strong>Why a 3-day window?</strong><br><br>
          • <strong>Day −1:</strong> Earnings can leak through management pre-announcements, options activity, or analyst whispers.<br><br>
          • <strong>Day 0:</strong> Announcement day — the primary price-discovery event.<br><br>
          • <strong>Day +1:</strong> The market often continues digesting analyst revisions, management commentary, and conference call transcripts the following morning.<br><br>
          A wider window (e.g., ±5 days) risks contamination from subsequent news unrelated to earnings.
          A narrower window (day 0 only) misses pre-leakage and post-processing reactions.
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Feature Engineering ───────────────────────────────────────────────
    st.markdown('<div class="section-header">3. Feature Engineering</div>', unsafe_allow_html=True)
    feat_data = {
        'Feature': ['prior_surprise', 'prior_beat', 'avg_surprise_4q', 'dispersion',
                    'num_analysts', 'prior_car', 'beat_streak', 'log_assets', 'leverage', 'roe'],
        'Source': ['IBES','IBES','IBES','IBES','IBES','CRSP','IBES','Compustat','Compustat','Compustat'],
        'Description': [
            'EPS surprise magnitude in the prior quarter (actual − consensus)',
            'Binary: did the firm beat in the prior quarter?',
            'Average EPS surprise over the prior 4 quarters',
            'Standard deviation of analyst EPS estimates (uncertainty proxy)',
            'Number of analysts covering the stock',
            '3-day CAR from the prior earnings announcement',
            'Number of consecutive beats in the last 4 quarters (0–4)',
            'Log of total assets (firm size control)',
            'Debt-to-equity ratio (leverage control)',
            'Return on equity (profitability control)',
        ],
        'Lag': ['1Q','1Q','4Q avg','0','0','1Q','4Q','1Q','1Q','1Q'],
    }
    st.dataframe(pd.DataFrame(feat_data), use_container_width=True, hide_index=True)

    st.divider()

    # ── Model Architecture ────────────────────────────────────────────────
    st.markdown('<div class="section-header">4. Model Architecture & Validation</div>', unsafe_allow_html=True)
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("""
        <div class="info-box">
          <strong>Beat/Miss Classification</strong><br><br>
          <strong>XGBoost (primary):</strong> Gradient-boosted ensemble tuned via 5-fold time-series CV.
          Handles non-linear feature interactions and missing values. AUC: 0.713.<br><br>
          <strong>Logistic Regression (baseline):</strong> L2-regularised logistic model on scaled features.
          Provides an interpretable probabilistic benchmark. AUC: 0.701.<br><br>
          Both models are evaluated on <em>out-of-time</em> hold-out sets — the test period never overlaps
          the training window.
        </div>
        """, unsafe_allow_html=True)
    with col_d:
        st.markdown("""
        <div class="info-box">
          <strong>CAR Regression</strong><br><br>
          <strong>Random Forest (primary):</strong> 200-tree ensemble with max_depth=6.
          Captures non-linear surprise-to-return mapping and interaction effects. RMSE: 0.118.<br><br>
          <strong>OLS (baseline):</strong> Standard linear regression on the same feature set.
          Interpretable and stable benchmark. RMSE: 0.118 (RF advantage emerges in tail events).<br><br>
          <em>Note:</em> R² for CAR regression is intentionally modest — short-window returns are noisy.
          The model captures systematic patterns, not noise.
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Charts Explained ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">5. Reading the Charts</div>', unsafe_allow_html=True)
    charts_info = {
        "Beat/Miss Gauge": "Displays the XGBoost probability (0–100%) that the upcoming earnings will beat consensus. The color zone encodes conviction: green ≥ 60%, yellow 40–60%, red ≤ 40%. The delta shows distance from the 50% decision boundary. Use in conjunction with the Logistic Regression baseline to check model agreement.",
        "Predicted 3-Day CAR Bar": "Shows the Random Forest point forecast for the 3-day abnormal return around the next announcement. Error bars represent ±1 standard deviation of historical realized CARs for that ticker — a rough confidence band. Green bars indicate a positive expected abnormal return; red indicates negative.",
        "Historical Accuracy Scorecard": "Displays the last 8 quarters of actual earnings outcomes: consensus EPS, reported EPS, surprise percentage, beat/miss status (✓/✗), and realized 3-day CAR. This lets you assess the stock's historical predictability before trusting the model's current forecast.",
        "XGBoost Feature Importance": "Shows each feature's gain score in the XGBoost model — how much each feature improves the classification objective when used in a split. Higher = more predictive. Beat streak and prior surprise typically dominate; firm fundamentals (leverage, ROE) contribute but are secondary.",
        "Surprise vs. CAR Scatter": "Each point is a historical earnings event for the selected ticker. The X-axis shows EPS surprise magnitude; Y-axis shows realized 3-day CAR. Green = beat, red = miss. Quadrant II (positive surprise, negative return) and Quadrant IV (miss, positive return) are cases where the market had already priced the outcome. The yellow star is the current prediction.",
    }
    for name, desc in charts_info.items():
        with st.expander(f"  {name}"):
            st.markdown(f'<p style="font-size:13px;color:#888;line-height:1.7;margin:0.5rem 0;">{desc}</p>', unsafe_allow_html=True)

    st.divider()

    # ── Sentiment & News ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">6. Sentiment, News & External Event Effects</div>', unsafe_allow_html=True)
    col_e, col_f = st.columns(2)
    with col_e:
        st.markdown("""
        <div class="info-box">
          <strong>A — The Expectations Treadmill</strong><br>
          Positive media sentiment in the 30 days before earnings is associated with upward analyst estimate revisions.
          When estimates rise, the hurdle for a beat rises too — making outperformance harder.
          This dynamic partially explains why stocks sometimes fall on strong earnings: the beat was already "priced in."
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
          <strong>B — Macro Event Contamination</strong><br>
          Fed rate decisions, CPI prints, and geopolitical shocks occurring <em>within</em> the 3-day CAR window
          inflate or deflate measured abnormal returns independent of earnings quality.
          The market-model beta-adjustment partially controls for market-wide moves, but idiosyncratic macro
          shocks during the window are not fully removed — this is a known limitation.
        </div>
        """, unsafe_allow_html=True)
    with col_f:
        st.markdown("""
        <div class="info-box">
          <strong>C — Management Guidance Tone</strong><br>
          Academic research (Huang et al. 2014; Bushee et al. 2018) shows that negative forward guidance
          issued alongside a positive EPS beat often produces a <em>negative</em> CAR — investors price
          the future, not the past quarter. The current model does not capture guidance language.
          Adding a FinBERT-scored guidance tone feature is a natural extension.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
          <strong>D — Short Interest & Options Positioning</strong><br>
          Elevated short interest amplifies positive CARs on beats (short squeeze) and can dampen negative CARs
          on misses. The <code>dispersion</code> feature (analyst estimate spread) is a partial proxy for
          pre-announcement uncertainty, but does not capture directional short positioning.
          Incorporating FINRA short interest data would improve tail-event accuracy.
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div class="info-box" style="border-left-color:#555;">
      <strong>Future extensions:</strong>
      NLP sentiment scoring via Alpha Vantage News API or FinBERT ·
      Management guidance tone extraction from earnings call transcripts ·
      FINRA short interest as a feature ·
      Options implied volatility as a pre-announcement uncertainty signal ·
      Sector-adjusted abnormal return benchmarking
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:9px;color:#333;letter-spacing:0.08em;margin-top:1rem;">BA870 / AC820 · MAHESH WADHOKAR · DATA: WRDS IBES, CRSP, COMPUSTAT · LIVE: YFINANCE</p>', unsafe_allow_html=True)