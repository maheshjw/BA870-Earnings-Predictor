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
    """Returns (info_dict, stock_object, warning_message_or_None)."""
    stock = yf.Ticker(ticker)
    for attempt in range(4):
        try:
            info = stock.info
            # yfinance sometimes returns a nearly-empty dict
            if info and len(info) > 5:
                return info, stock, None
        except Exception as e:
            err = str(e)
            if attempt < 3:
                wait = (attempt + 1) * 6   # 6 s, 12 s, 18 s
                time.sleep(wait)
            else:
                return {}, stock, f"Yahoo Finance unavailable ({err[:80]}). Using dataset averages for live features."
    return {}, stock, "Yahoo Finance returned empty data. Using dataset averages for live features."

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
                info, stock, yf_warning = fetch_yfinance_info(ticker)
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

                # ── SENTIMENT ANALYSIS ────────────────────────────────────
                st.divider()
                st.markdown("#### 📰 Sentiment Analysis")
                st.caption("News sentiment via yfinance headlines (TextBlob) · Earnings tone via FinBERT")

                sent_col1, sent_col2 = st.columns(2)

                # ── LEFT: News Sentiment ──────────────────────────────────
                with sent_col1:
                    st.markdown("##### 📡 Recent News Sentiment")
                    try:
                        from textblob import TextBlob

                        news_items = stock.news
                        if not news_items:
                            st.info("No recent news found for this ticker.")
                        else:
                            rows = []
                            polarities = []
                            for item in news_items[:10]:
                                # yfinance news structure varies — handle both old and new
                                content = item.get('content', {})
                                if isinstance(content, dict):
                                    title    = content.get('title', item.get('title', ''))
                                    pub_date = content.get('pubDate', item.get('providerPublishTime', ''))
                                    link     = content.get('canonicalUrl', {}).get('url', '') if isinstance(content.get('canonicalUrl'), dict) else ''
                                else:
                                    title    = item.get('title', '')
                                    pub_date = item.get('providerPublishTime', '')
                                    link     = item.get('link', '')

                                if not title:
                                    continue

                                blob       = TextBlob(title)
                                polarity   = blob.sentiment.polarity       # -1 to +1
                                subjectivity = blob.sentiment.subjectivity # 0 to 1
                                polarities.append(polarity)

                                if polarity > 0.1:
                                    label = '🟢 Positive'
                                elif polarity < -0.1:
                                    label = '🔴 Negative'
                                else:
                                    label = '🟡 Neutral'

                                # format date
                                if isinstance(pub_date, int):
                                    import datetime
                                    pub_date = datetime.datetime.fromtimestamp(pub_date).strftime('%Y-%m-%d')
                                elif isinstance(pub_date, str) and 'T' in pub_date:
                                    pub_date = pub_date[:10]

                                rows.append({
                                    'Date':        pub_date,
                                    'Headline':    title[:80] + ('…' if len(title) > 80 else ''),
                                    'Sentiment':   label,
                                    'Score':       f"{polarity:+.2f}",
                                })

                            if rows:
                                # Summary metrics
                                avg_pol   = sum(polarities) / len(polarities)
                                n_pos     = sum(1 for p in polarities if p > 0.1)
                                n_neg     = sum(1 for p in polarities if p < -0.1)
                                n_neu     = len(polarities) - n_pos - n_neg

                                sm1, sm2, sm3, sm4 = st.columns(4)
                                sm1.metric("Avg Score",  f"{avg_pol:+.2f}",
                                           delta="Bullish" if avg_pol > 0.05 else ("Bearish" if avg_pol < -0.05 else "Neutral"))
                                sm2.metric("🟢 Positive", n_pos)
                                sm3.metric("🟡 Neutral",  n_neu)
                                sm4.metric("🔴 Negative", n_neg)

                                # Sentiment bar chart
                                fig_sent = go.Figure()
                                colors   = ['#22c55e' if p > 0.1 else ('#ef4444' if p < -0.1 else '#eab308')
                                            for p in polarities]
                                fig_sent.add_trace(go.Bar(
                                    x=list(range(1, len(polarities)+1)),
                                    y=polarities,
                                    marker_color=colors,
                                    marker_line_width=0,
                                    name='Polarity'
                                ))
                                fig_sent.add_hline(y=0, line_dash='dot', line_color='gray', line_width=1)
                                fig_sent.update_layout(
                                    height=200,
                                    margin=dict(l=10, r=10, t=10, b=30),
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    xaxis=dict(title='Article #', tickfont=dict(size=10), gridcolor='#1a1a1a'),
                                    yaxis=dict(title='Polarity', tickfont=dict(size=10),
                                               range=[-1, 1], gridcolor='#1a1a1a'),
                                    showlegend=False,
                                )
                                st.plotly_chart(fig_sent, use_container_width=True)

                                # Headlines table
                                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                            else:
                                st.info("Could not parse news headlines.")

                    except ImportError:
                        st.warning("TextBlob not installed. Run: `pip install textblob`")
                    except Exception as e_sent:
                        st.warning(f"News sentiment unavailable: {str(e_sent)[:120]}")

                # ── RIGHT: FinBERT Earnings Tone ──────────────────────────
                with sent_col2:
                    st.markdown("##### 🤖 Earnings Tone (FinBERT)")
                    st.caption("FinBERT is a BERT model fine-tuned on financial text — more accurate than TextBlob for finance language.")
                    try:
                        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

                        @st.cache_resource(show_spinner="Loading FinBERT model...")
                        def load_finbert():
                            tok   = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                            model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
                            return pipeline("text-classification", model=model,
                                            tokenizer=tok, top_k=None)

                        finbert = load_finbert()

                        # Collect sentences to score:
                        # 1. Recent news headlines (same as left panel)
                        # 2. Company description as a proxy for tone
                        sentences = []
                        news_items_fb = stock.news or []
                        for item in news_items_fb[:8]:
                            content = item.get('content', {})
                            title   = (content.get('title') if isinstance(content, dict) else None) or item.get('title', '')
                            if title:
                                sentences.append(title)

                        # Add company summary if available
                        summary = info.get('longBusinessSummary', '')
                        if summary:
                            # Split into sentences, take first 5
                            import re
                            sents = re.split(r'(?<=[.!?])\s+', summary)
                            sentences += sents[:5]

                        if not sentences:
                            st.info("No text available for FinBERT analysis.")
                        else:
                            # Score all sentences
                            results     = finbert(sentences[:15])  # cap at 15 to keep it fast
                            pos_scores  = []
                            neg_scores  = []
                            neu_scores  = []

                            for result in results:
                                scores = {r['label']: r['score'] for r in result}
                                pos_scores.append(scores.get('positive', 0))
                                neg_scores.append(scores.get('negative', 0))
                                neu_scores.append(scores.get('neutral',  0))

                            avg_pos = sum(pos_scores) / len(pos_scores)
                            avg_neg = sum(neg_scores) / len(neg_scores)
                            avg_neu = sum(neu_scores) / len(neu_scores)
                            dominant = max([('Positive', avg_pos), ('Neutral', avg_neu), ('Negative', avg_neg)],
                                           key=lambda x: x[1])

                            # Summary metrics
                            fb1, fb2, fb3, fb4 = st.columns(4)
                            fb1.metric("Dominant Tone", dominant[0],
                                       delta=f"{dominant[1]:.0%} confidence")
                            fb2.metric("🟢 Positive",  f"{avg_pos:.0%}")
                            fb3.metric("🟡 Neutral",   f"{avg_neu:.0%}")
                            fb4.metric("🔴 Negative",  f"{avg_neg:.0%}")

                            # Stacked bar showing avg distribution
                            fig_fb = go.Figure()
                            fig_fb.add_trace(go.Bar(
                                name='Positive', x=['FinBERT Score'], y=[avg_pos],
                                marker_color='#22c55e', marker_line_width=0
                            ))
                            fig_fb.add_trace(go.Bar(
                                name='Neutral', x=['FinBERT Score'], y=[avg_neu],
                                marker_color='#eab308', marker_line_width=0
                            ))
                            fig_fb.add_trace(go.Bar(
                                name='Negative', x=['FinBERT Score'], y=[avg_neg],
                                marker_color='#ef4444', marker_line_width=0
                            ))
                            fig_fb.update_layout(
                                barmode='stack',
                                height=220,
                                margin=dict(l=10, r=10, t=10, b=10),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                legend=dict(orientation='h', yanchor='bottom', y=1.02,
                                            font=dict(size=11)),
                                yaxis=dict(tickformat='.0%', gridcolor='#1a1a1a',
                                           tickfont=dict(size=10)),
                                xaxis=dict(tickfont=dict(size=11)),
                                showlegend=True,
                            )
                            st.plotly_chart(fig_fb, use_container_width=True)

                            # Per-sentence breakdown
                            sent_rows = []
                            for i, (sent, result) in enumerate(zip(sentences[:15], results)):
                                scores  = {r['label']: r['score'] for r in result}
                                top_lbl = max(scores, key=scores.get)
                                emoji   = '🟢' if top_lbl == 'positive' else ('🔴' if top_lbl == 'negative' else '🟡')
                                sent_rows.append({
                                    'Text':      sent[:90] + ('…' if len(sent) > 90 else ''),
                                    'Tone':      f"{emoji} {top_lbl.capitalize()}",
                                    'Confidence': f"{scores[top_lbl]:.0%}",
                                })
                            st.dataframe(pd.DataFrame(sent_rows),
                                         use_container_width=True, hide_index=True)

                    except ImportError:
                        st.warning("transformers not installed. Run: `pip install transformers torch`")
                    except Exception as e_fb:
                        st.warning(f"FinBERT unavailable: {str(e_fb)[:120]}")

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
# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — METHODOLOGY  (numbers taken directly from BA870_AC820_Data_Pipeline.ipynb)
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — METHODOLOGY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_methodology:

    st.header("📖 Methodology")
    st.markdown("*All numbers taken directly from the Colab data pipeline notebook.*")
    st.divider()

    # ── SUMMARY ROW ───────────────────────────────────────────────────────────
    o1, o2, o3, o4, o5, o6 = st.columns(6)
    o1.metric("IBES Rows",       "556,000",    delta="Raw download")
    o2.metric("CRSP Rows",       "67,500,000", delta="Raw download")
    o3.metric("Compustat Rows",  "321,000",    delta="Raw download")
    o4.metric("IBES–CRSP Linked","591,466",    delta="After permno join")
    o5.metric("Final Dataset",   "52,891",     delta="Clean rows")
    o6.metric("XGBoost AUC",     "0.713",      delta="Best model")

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 1
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("📦 Phase 1 — Data Collection")

    p1a, p1b, p1c = st.columns(3)

    with p1a:
        st.markdown("#### IBES — EPS Estimates")
        st.metric("Rows Downloaded", "556,000", delta="WRDS IBES Summary 1990–2024")
        st.markdown("""
        Each row is one consensus EPS estimate for one company for one quarter.

        **Key columns used:**
        - `meanest` — consensus EPS (average of all analyst forecasts)
        - `actual` — actual reported EPS
        - `anndats_act` — earnings announcement date
        - `fpi = 6` — quarterly forecast filter
        - `oftic` — official ticker (used to link to CRSP)
        - `stdev` — analyst estimate std deviation (dispersion)
        - `numest` — number of analysts covering the stock
        """)

    with p1b:
        st.markdown("#### CRSP — Daily Stock Returns")
        st.metric("Rows Downloaded", "67,500,000", delta="WRDS CRSP Daily 1990–2024")
        st.markdown("""
        Downloaded in 500K-row chunks, filtered to IBES tickers only to reduce the raw 3.4GB file to a manageable size.

        **Key columns used:**
        - `permno` — CRSP permanent security ID (unique stock identifier)
        - `date` — trading date
        - `ret` — daily holding period return

        **Market return added separately:**
        S&P 500 daily returns downloaded via yfinance (`^GSPC`) and merged as `vwretd`.
        """)

    with p1c:
        st.markdown("#### Compustat — Firm Fundamentals")
        st.metric("Rows Downloaded", "321,000", delta="WRDS Compustat Annual")
        st.markdown("""
        Filtered to industrial firms, standard format, consolidated statements, and total assets > 0.

        **Key columns used:**
        - `at` — total assets
        - `dltt` — long-term debt
        - `ceq` — common equity
        - `ni` — net income
        - `tic` — ticker symbol

        **Three features derived:**
        - `log_assets` = log(total assets) — firm size
        - `leverage` = long-term debt / total assets
        - `roe` = net income / common equity
        """)

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 2
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("🔧 Phase 2 — Cleaning & Feature Engineering")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CRSP–IBES Linked",  "591,466", delta="via htsymbol → oftic")
    c2.metric("CAR Computed On",   "591,466", delta="parallel processing")
    c3.metric("Compustat Match",   "67.5%",   delta="oftic + fiscal year")
    c4.metric("Final Clean Rows",  "52,891",  delta="after dropna all 10 features")

    st.markdown("")
    ph2a, ph2b = st.columns(2)

    with ph2a:
        st.markdown("#### IBES Cleaning")
        st.markdown("""
        1. Removed rows with missing `actual`, `meanest`, `anndats_act`, or `ticker`
        2. Kept only **quarterly forecasts** — `fpi = 6`
        3. Kept only the **most recent estimate** before each announcement date
        4. Computed **EPS Surprise** = (Actual − Consensus) / |Consensus|, clipped at ±1.0
        5. Created **Beat/Miss label** — `1` if actual ≥ consensus, else `0`
        """)

        st.markdown("#### CRSP–IBES Link")
        st.markdown("""
        IBES uses ticker symbols. CRSP uses `permno` (permanent security number).
        We used the **CRSP Stock Header file** (`htsymbol` column) to map between them,
        joining on `oftic` (IBES official ticker) → `htsymbol` (CRSP header ticker).

        **Result: 591,466 IBES announcements successfully linked to CRSP permnos**
        """)
        la, lb = st.columns(2)
        la.metric("Rows Linked", "591,466")
        lb.metric("Join key", "oftic → htsymbol")

        st.markdown("#### Compustat Merge")
        st.markdown("""
        Matched to IBES via `oftic` ticker and `fiscal year`.
        Unmatched rows (~32.5%) were dropped — these become the `dropna` rows in the final step.
        """)
        ma, mb = st.columns(2)
        ma.metric("Match Rate", "67.5%")
        mb.metric("Unmatched", "→ dropped")

    with ph2b:
        st.markdown("#### 3-Day CAR Computation")
        st.markdown("""
        For **each of the 591,466 earnings announcements**, the following steps were run:

        1. Retrieved **200 trading days** of stock returns *before* the announcement — the estimation window
        2. Ran **OLS regression**: stock return = α + β × market return (S&P 500)
        3. Computed the **expected return** for each event day using the estimated α and β
        4. **Abnormal return** per day = actual return − expected return
        5. Summed abnormal returns over days **−1, 0, and +1** to get the 3-day CAR

        Minimum 30 days of valid data required — otherwise returns `NaN` (dropped later).
        """)

        ca, cb, cc = st.columns(3)
        ca.metric("Estimation Window", "200 days")
        cb.metric("Min History",       "30 days",  delta="else NaN")
        cc.metric("Avg CAR Result",    "0.11%",    delta="= 0.0011 raw")

        st.markdown("""
        **Processing:** Run in parallel using `ThreadPoolExecutor` with 4 workers, batch size 2,000.
        Progress saved to Google Drive every 10,000 rows as a checkpoint.
        """)

        st.markdown("#### Feature Engineering — 10 Features")
        feat_df = pd.DataFrame({
            'Feature':      ['prior_surprise','prior_beat','avg_surprise_4q',
                             'dispersion','num_analysts','prior_car','beat_streak',
                             'log_assets','leverage','roe'],
            'Source':       ['IBES','IBES','IBES','IBES','IBES',
                             'CRSP','IBES','Compustat','Compustat','Compustat'],
            'Lag':          ['1 quarter','1 quarter','4-quarter avg','same quarter',
                             'same quarter','1 quarter','prior 4 quarters',
                             '1 quarter','1 quarter','1 quarter'],
            'Measures':     ["Last Q EPS surprise",
                             "Beat last Q? (0/1)",
                             "Rolling avg surprise last 4Q",
                             "Analyst std dev of estimates",
                             "Number of analysts",
                             "Prior Q 3-day CAR",
                             "Consecutive beats (0–4)",
                             "Log total assets",
                             "Debt / total assets",
                             "Net income / equity"],
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

        fd1, fd2, fd3 = st.columns(3)
        fd1.metric("Final Rows", "52,891")
        fd2.metric("Tickers",    "1,894")
        fd3.metric("Period",     "1990–2024")

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 3
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("🤖 Phase 3 — Model Training")

    st.markdown("#### Train / Test Split — time-based to prevent look-ahead bias")
    sp1, sp2, sp3 = st.columns(3)
    sp1.metric("Train Set",  "33,121 rows", delta="before 2019-01-01")
    sp2.metric("Test Set",   "19,770 rows", delta="2019-01-01 onwards")
    sp3.metric("Split Type", "Temporal",    delta="No future data in train ✓")

    st.markdown("")
    mc1, mc2 = st.columns(2)

    with mc1:
        st.markdown("#### Classification — Predict Beat vs. Miss")
        cl1, cl2, cl3, cl4 = st.columns(4)
        cl1.metric("LR Accuracy",  "68.5%")
        cl2.metric("LR AUC",       "0.701")
        cl3.metric("XGB Accuracy", "68.4%")
        cl4.metric("XGB AUC",      "0.713", delta="Primary ✓")
        st.markdown("""
        **Logistic Regression** — L2-regularised, trained on scaled features. Interpretable baseline.

        **XGBoost** — `n_estimators=100, max_depth=4, learning_rate=0.1`.
        Handles non-linear feature interactions and class imbalance natively.

        **AUC = 0.713** means the model correctly ranks a true beat above a true miss **71.3%** of the time.
        Both models beat random guessing (AUC = 0.50) significantly.
        """)

    with mc2:
        st.markdown("#### Regression — Predict 3-Day CAR")
        rg1, rg2 = st.columns(2)
        rg1.metric("OLS RMSE",           "0.1179")
        rg2.metric("Random Forest RMSE", "0.1180")
        st.markdown("""
        **OLS** — Linear regression on scaled features. Interpretable benchmark.

        **Random Forest** — `n_estimators=100, max_depth=5`.
        Captures non-linear mapping from surprise magnitude to return.

        RMSE is modest — 3-day abnormal returns are inherently noisy.
        Models capture the **systematic** component, not idiosyncratic noise.
        """)

    st.markdown("#### Files saved to Google Drive then uploaded to GitHub:")
    sv1, sv2, sv3, sv4, sv5, sv6 = st.columns(6)
    sv1.metric("model_lr.pkl",       "LR",            delta="beat classifier")
    sv2.metric("model_xgb.pkl",      "XGBoost",       delta="beat classifier")
    sv3.metric("model_ols.pkl",      "OLS",           delta="CAR regressor")
    sv4.metric("model_rf.pkl",       "Random Forest", delta="CAR regressor")
    sv5.metric("scaler.pkl",         "StandardScaler",delta="for LR + OLS")
    sv6.metric("feature_cols.json",  "10 features",   delta="column order lock")

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 4
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("🚀 Phase 4 — Streamlit App")

    ap1, ap2 = st.columns(2)

    with ap1:
        st.markdown("#### Live Prediction Flow")
        st.markdown("""
        1. User enters ticker → **yfinance API** fetches live firm fundamentals
        2. Retrieves: total assets, debt/equity, return on equity, analyst count
        3. Historical WRDS dataset queried for prior earnings signals
        4. Same **10-feature vector** assembled as in training data
        5. **XGBoost** → beat/miss probability (0–100%)
        6. **Random Forest** → predicted 3-day CAR (%)
        7. **TextBlob** → news headline sentiment (polarity −1 to +1)
        8. **FinBERT** → earnings tone (Positive / Neutral / Negative %)
        """)
        d1, d2 = st.columns(2)
        d1.metric("Code", "GitHub", delta="maheshjw/BA870-Earnings-Predictor")
        d2.metric("Host", "Streamlit Cloud", delta="Live · free tier")

    with ap2:
        st.markdown("#### 8 Visual Components")
        comp_df = pd.DataFrame({
            '#': [1,2,3,4,5,6,7,8],
            'Visual': [
                'Beat/Miss Probability Gauge',
                'Predicted 3-Day CAR Bar Chart',
                'Top 3 Metrics Row',
                'Historical Accuracy Scorecard',
                'Feature Importance (XGBoost)',
                'EPS Surprise vs CAR Scatter',
                'News Sentiment (TextBlob)',
                'Earnings Tone (FinBERT)',
            ],
            'What it shows': [
                'XGBoost beat probability 0–100%, color-coded green/yellow/red',
                'Predicted abnormal return + ±1σ reference lines, value labelled',
                'Beat prob (XGB), predicted CAR, beat prob (Logistic Regression)',
                'Last 8 quarters: consensus EPS, actual EPS, surprise, beat/miss, CAR',
                'XGBoost gain scores for all 10 features — beat_streak always #1',
                'Prior surprise vs realized CAR · blue ★ = current prediction',
                'Polarity per headline (−1 to +1), avg score, pos/neu/neg counts',
                'Positive/Neutral/Negative % distribution + per-sentence confidence',
            ],
        })
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # 3-DAY CAR DEEP DIVE
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("📐 3-Day CAR — How It's Computed")

    cd1, cd2 = st.columns(2)
    with cd1:
        st.markdown("""
        **Cumulative Abnormal Return (CAR)** isolates how much a stock moved *because of earnings news*,
        after removing what the broad market would have caused on the same days.

        **Step 1** — Estimate a market model over a **200-day window** ending before the announcement:
        > Stock Return = α + β × Market Return

        **Step 2** — Compute the **abnormal return** on each event day:
        > AR = Actual Return − (α + β × Market Return)

        **Step 3** — Sum over the 3-day window:
        > CAR = AR₋₁ + AR₀ + AR₊₁

        | Day | Why included |
        |-----|-------------|
        | **−1** | Leakage via options activity, analyst whispers, pre-announcements |
        | **0**  | Official announcement — primary price discovery event |
        | **+1** | Analyst revisions + earnings call commentary processed overnight |
        """)

    with cd2:
        st.markdown("""
        **Interpreting the value:**

        | CAR | Meaning |
        |-----|---------|
        | **+5%** | Stock returned 5% more than the market model predicted |
        | **−3%** | Underperformed expected return by 3% |
        | **≈ 0%** | Announcement fully priced in — no information surprise |
        """)
        cv1, cv2, cv3 = st.columns(3)
        cv1.metric("Estimation Window", "200 days")
        cv2.metric("Avg CAR",           "+0.11%", delta="from 591K events")
        cv3.metric("CAR Std Dev",        "~11.8%", delta="wide outcome range")
        st.info(
            "Predicted CARs are small (often < 1%) because the model predicts the "
            "*expected* abnormal return. For consistently-beating stocks like AAPL "
            "(75.7% hit rate), the surprise is partially anticipated — the systematic "
            "component is small. The ±1σ reference lines on the chart show the true "
            "width of actual outcomes."
        )

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # SENTIMENT ANALYSIS
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("📰 Sentiment Analysis — How It Works")
    st.markdown("Two models run in parallel on the same text — recent yfinance news headlines + company business description.")

    sb1, sb2 = st.columns(2)

    with sb1:
        st.markdown("### 📡 TextBlob — News Sentiment")
        st.markdown("""
        **Approach:** Dictionary-based word lookup. No neural network, no training required.

        **How polarity is calculated:**
        1. Tokenise the headline into words
        2. Look up each word in the PatternSentiment lexicon
        3. Average the non-zero scores across all words
        4. Apply negation modifiers — "not good" flips the sign

        **Output range:** −1.0 (very negative) to +1.0 (very positive)
        """)
        tbs1, tbs2, tbs3 = st.columns(3)
        tbs1.metric("🟢 Positive",  "> +0.1")
        tbs2.metric("🟡 Neutral",   "−0.1 to +0.1")
        tbs3.metric("🔴 Negative",  "< −0.1")
        st.markdown("""
        **Finance weakness:** TextBlob has no financial context.
        - "Earnings miss" → scores Neutral ❌
        - "Guidance cut" → scores Neutral ❌
        - "Strong headwinds" → scores Positive ❌

        Good for a quick directional read, not for finance-specific accuracy.
        """)

    with sb2:
        st.markdown("### 🤖 FinBERT — Earnings Tone")
        st.markdown("""
        **Approach:** BERT neural network fine-tuned specifically on financial text.
        Understands finance jargon that confuses general-purpose models.
        """)
        fb1, fb2 = st.columns(2)
        fb1.metric("Financial News Sentences", "4,840",  delta="Reuters + Bloomberg")
        fb2.metric("Analyst Report Sentences", "10,000", delta="Hand-labeled by experts")
        st.markdown("""
        **How confidence is calculated:**
        1. Tokenise text into subword pieces
        2. Add `[CLS]` token — its final vector represents the whole sentence meaning
        3. Pass through **12 transformer layers** — self-attention lets every word look at every other word simultaneously
        4. Final linear layer produces 3 raw scores (logits)
        5. **Softmax** converts logits to probabilities that always sum to 100%

        The highest probability = the **confidence** score.
        """)

        conf_df = pd.DataFrame({
            'Confidence': ['>80%','50–80%','<50%','~33%'],
            'What it means': [
                'Very sure — trust the label',
                'Reasonably confident — likely correct',
                'Borderline — text is genuinely ambiguous',
                'No idea — all 3 labels equally likely',
            ]
        })
        st.dataframe(conf_df, use_container_width=True, hide_index=True)

        st.markdown("""
        **Why FinBERT beats TextBlob for finance:**

        > *"Apple misses revenue despite strong iPhone sales"*
        - TextBlob → **Positive** ❌ (sees "strong", ignores "misses")
        - FinBERT → **Negative 71% confidence** ✅ (reads full sentence in context)
        """)

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # CHART GUIDE
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("📊 Chart Guide")

    with st.expander("🎯 Beat/Miss Probability Gauge", expanded=True):
        g1, g2, g3 = st.columns(3)
        g1.metric("🟢 Green zone",  "60–100%", delta="Confident beat")
        g2.metric("🟡 Yellow zone", "40–60%",  delta="Uncertain")
        g3.metric("🔴 Red zone",    "0–40%",   delta="Leans miss")
        st.markdown("""
        Delta shows distance from the **50% decision boundary**.
        A reading of 73% = 23 percentage points above the toss-up line.
        Both XGBoost and Logistic Regression shown — when both agree, the signal is stronger.
        """)

    with st.expander("📊 Predicted 3-Day CAR Bar Chart"):
        st.markdown("""
        - **Green bar** = positive predicted abnormal return · **Red** = negative
        - Value label always printed on bar regardless of how small it is
        - **±1σ dotted lines** = historical spread of realized CARs — reference band, not prediction interval
        - Y-axis clamped to 3× predicted value so a 0.46% prediction doesn't disappear on a ±10% axis
        """)

    with st.expander("📋 Historical Accuracy Scorecard"):
        st.markdown("""
        Last **8 quarters** of actual outcomes from WRDS IBES + CRSP for the selected ticker.

        | Column | Source | Meaning |
        |--------|--------|---------|
        | Date | IBES `anndats_act` | Earnings announcement date |
        | Consensus EPS | IBES `meanest` | Mean analyst estimate |
        | Actual EPS | IBES `actual` | Reported EPS |
        | Surprise | Computed | (Actual − Consensus) / |Consensus|, clipped ±1 |
        | Beat | Computed | ✅ actual ≥ consensus · ❌ missed |
        | 3-Day CAR | CRSP | Realized abnormal return days −1, 0, +1 |
        """)
        st.info("Only available for the 1,894 tickers in the WRDS dataset. TSLA, JPM, GOOGL, WMT not included — models still run using dataset averages + live yfinance fundamentals.")

    with st.expander("🔍 Feature Importance (XGBoost)"):
        st.markdown("""
        Each feature's **gain score** = average improvement in log-loss when that feature is used in a split, weighted by frequency.

        | Rank | Feature | Why it dominates |
        |------|---------|-----------------|
        | **1** | `beat_streak` | Firms beating 3–4 consecutive quarters keep beating — "sandbagging" guidance |
        | **2** | `prior_surprise` | Analysts under-correct — big beat last quarter predicts another |
        | **3** | `avg_surprise_4q` | Sustained beat history = structural expectation-management advantage |
        | **4** | `num_analysts` | Heavy coverage = tighter consensus = harder to beat |
        | **5** | `dispersion` | High analyst disagreement = genuine pre-announcement uncertainty |
        | 6–10 | Fundamentals | `log_assets`, `leverage`, `roe`, `prior_car`, `prior_beat` |
        """)

    with st.expander("📉 EPS Surprise vs 3-Day CAR Scatter"):
        st.markdown("""
        Each dot = one past earnings event · X = EPS surprise · Y = 3-day CAR · 🟢 beat · 🔴 miss

        **Blue star ★** = current prediction overlaid on the historical cloud.

        | Quadrant | Surprise | CAR | Interpretation |
        |----------|----------|-----|----------------|
        | Top-right ↗ | + | + | Beat + rewarded — the normal case |
        | Bottom-left ↙ | − | − | Miss + punished — the normal case |
        | Top-left ↖ | − | + | Miss but stock rose — bad news already priced in |
        | Bottom-right ↘ | + | − | Beat but fell — "buy the rumor, sell the news" |
        """)

    with st.expander("📡 News Sentiment (TextBlob)"):
        st.markdown("""
        - One bar per headline, height = polarity score (−1 to +1)
        - Green = positive (> +0.1) · Yellow = neutral · Red = negative (< −0.1)
        - 4 summary metrics: average score, count of positive / neutral / negative headlines
        - Average score > +0.05 → Bullish signal · < −0.05 → Bearish signal
        """)

    with st.expander("🤖 Earnings Tone (FinBERT)"):
        st.markdown("""
        - Scores the same news headlines + first 5 sentences of the company business description
        - Stacked bar = full 100% split between Positive / Neutral / Negative
        - **Dominant Tone** = whichever of the 3 has the highest average probability across all sentences
        - **Confidence** = the softmax probability of the dominant label
          - > 80% = trust the label · < 50% = text is genuinely ambiguous
        - Per-sentence table shows individual tone + confidence for every piece of text scored
        """)

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # EXTERNAL EFFECTS
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("📰 External Sentiment Effects on CAR")
    st.markdown("Four documented factors that explain residual variance the model doesn't capture:")

    ef1, ef2 = st.columns(2)
    with ef1:
        st.markdown("""
        #### A. 📈 The Expectations Treadmill
        Positive pre-earnings media sentiment drives upward analyst revisions, raising the consensus.
        A higher consensus = a harder beat threshold.
        This is why stocks sometimes fall on strong earnings — the beat was already priced in.

        > **Model gap:** `dispersion` captures uncertainty but not the direction of analyst revisions.

        ---

        #### B. 🏛️ Macro Event Contamination
        Fed rate decisions, CPI prints, and geopolitical shocks occurring within the 3-day CAR window
        inflate or deflate the measured abnormal return independent of earnings quality.
        The market-model beta removes broad market moves but not idiosyncratic sector shocks.

        > **Model gap:** CARs measured during major macro events should be interpreted cautiously.
        """)

    with ef2:
        st.markdown("""
        #### C. 🎙️ Management Guidance Tone
        Negative forward guidance alongside a positive EPS beat often produces a negative CAR —
        investors price the *outlook*, not the past quarter.
        A miss paired with strong guidance can produce a positive CAR.

        > **Model gap:** FinBERT currently scores news + company description.
        > Scoring actual earnings call transcripts is the highest-impact next step.

        ---

        #### D. 📉 Short Interest & IV Crush
        High short interest amplifies positive CARs on beats via short covering (short squeeze).
        High pre-earnings implied volatility often collapses after the announcement (IV crush) —
        the realized CAR ends up smaller than the implied move, especially for mega-caps.

        > **Model gap:** `dispersion` proxies uncertainty but not directional short positioning.
        """)

    st.divider()
    with st.expander("🔮 Future Extensions"):
        st.markdown("""
        - **FinBERT on earnings call transcripts** — score actual management guidance, not just news headlines
        - **Analyst revision momentum** — direction and magnitude of estimate changes 30 days before announcement
        - **Options implied volatility** — IV term structure as a pre-announcement uncertainty signal
        - **FINRA short interest** — directional positioning as a CAR amplifier feature
        - **Fama-French 3-factor CAR** — industry-adjusted benchmark instead of single-factor market model
        - **Whisper numbers** — unofficial EPS expectations that often diverge from published consensus
        """)

    st.caption("BA870 / AC820 · Mahesh Wadhokar · Data: WRDS IBES, CRSP, Compustat · Live: yfinance · Sentiment: TextBlob + FinBERT")
