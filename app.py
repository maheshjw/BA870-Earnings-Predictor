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
# TAB 2 — METHODOLOGY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_methodology:

    st.header("📖 Methodology")
    st.markdown("*Full walkthrough of every phase — data, cleaning, features, models, and the live app. All numbers are from our actual code output.*")
    st.divider()

    # ── PHASE 1 ───────────────────────────────────────────────────────────────
    st.subheader("📦 Phase 1 — Data Collection")

    r1, r2, r3 = st.columns(3)
    r1.metric("IBES Rows Downloaded",  "1,670,000",  delta="WRDS analyst EPS estimates")
    r2.metric("CRSP Rows Downloaded",  "67,500,000", delta="WRDS daily stock returns")
    r3.metric("Compustat Rows",        "443,000",    delta="Annual firm fundamentals")

    st.markdown("")
    p1a, p1b, p1c = st.columns(3)
    with p1a:
        st.markdown("**IBES — Key columns:**")
        st.code("ticker       — stock symbol\nmeanest      — consensus EPS\nactual       — reported EPS\nanndats_act  — announcement date", language=None)
    with p1b:
        st.markdown("**CRSP — Key columns:**")
        st.code("permno  — CRSP stock ID\ndate    — trading date\nret     — daily return\nvwretd  — S&P500 return (yfinance ^GSPC)", language=None)
    with p1c:
        st.markdown("**Compustat — Key columns:**")
        st.code("at    — total assets\ndltt  — long-term debt\nceq   — common equity\nni    — net income", language=None)

    st.divider()

    # ── PHASE 2 ───────────────────────────────────────────────────────────────
    st.subheader("🔧 Phase 2 — Cleaning & Feature Engineering")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CRSP-IBES Rows Linked", "591,000",  delta="via oftic ticker match")
    c2.metric("CAR Computed On",       "110,000",  delta="announcement events")
    c3.metric("Compustat Match Rate",  "67.5%",    delta="oftic + fiscal year")
    c4.metric("Final Dataset",         "52,891",   delta="rows · 1,894 tickers")

    st.markdown("")
    p2a, p2b = st.columns(2)

    with p2a:
        st.markdown("#### IBES Cleaning Steps")
        st.markdown("""
1. Removed rows with missing `actual` or `meanest`
2. Kept only quarterly forecasts — `fpi = 6`
3. Kept only the **most recent estimate** before each announcement date
4. Computed EPS Surprise:
        """)
        st.code("Surprise = (Actual - Consensus) / |Consensus|", language=None)
        st.markdown("5. Beat/Miss label: `1` if actual >= consensus, else `0`")

        st.markdown("#### CRSP-IBES Link")
        ca, cb = st.columns(2)
        ca.metric("Rows Linked", "591,000")
        cb.metric("Method", "oftic join")

        st.markdown("#### Compustat Merge")
        st.markdown("Calculated `log_assets`, `leverage` (debt/equity), `roe` (NI/equity). Matched via `oftic` + fiscal year.")
        cm1, cm2 = st.columns(2)
        cm1.metric("Match Rate", "67.5%")
        cm2.metric("Unmatched rows", "use dataset avg")

    with p2b:
        st.markdown("#### 3-Day CAR Computation")
        st.markdown("""
For **each earnings announcement:**
1. Get **200 days** of returns before the event (estimation window)
2. Run OLS: `R_stock = alpha + beta x R_market`
3. Compute expected return each day
4. Abnormal return = actual - expected
5. Sum over days -1, 0, +1
        """)
        st.code("AR_t = R_stock_t - (alpha + beta x R_market_t)\nCAR  = AR_-1 + AR_0 + AR_+1", language=None)

        car1, car2, car3 = st.columns(3)
        car1.metric("Rows Processed", "110,000")
        car2.metric("Checkpoint Every", "10,000 rows")
        car3.metric("Avg CAR Result", "0.11%", delta="financially sensible")

        st.markdown("#### Feature Engineering — 10 Features")
        feat_table = pd.DataFrame({
            'Feature':         ['prior_surprise','prior_beat','avg_surprise_4q',
                                'dispersion','num_analysts','prior_car','beat_streak',
                                'log_assets','leverage','roe'],
            'Source':          ['IBES','IBES','IBES','IBES','IBES','CRSP','IBES',
                                'Compustat','Compustat','Compustat'],
            'What it measures':["Last Q EPS surprise",
                                "Beat last quarter? (0/1)",
                                "Avg surprise last 4 quarters",
                                "Analyst disagreement (std dev)",
                                "# analysts covering stock",
                                "Prior quarter 3-day CAR",
                                "Consecutive beats (0-4)",
                                "Firm size (log total assets)",
                                "Debt-to-equity / 100",
                                "Return on equity"],
        })
        st.dataframe(feat_table, use_container_width=True, hide_index=True)

        fd1, fd2, fd3 = st.columns(3)
        fd1.metric("Final Rows", "52,891")
        fd2.metric("Tickers", "1,894")
        fd3.metric("Period", "1990-2024")

    st.divider()

    # ── PHASE 3 ───────────────────────────────────────────────────────────────
    st.subheader("🤖 Phase 3 — Model Training")

    st.markdown("#### Train / Test Split — time-based to prevent look-ahead bias")
    sp1, sp2, sp3 = st.columns(3)
    sp1.metric("Train Set", "33,121 rows", delta="Pre-2019")
    sp2.metric("Test Set",  "19,770 rows", delta="2019-2024")
    sp3.metric("Split Rule","Temporal",    delta="No future data in train")

    st.markdown("")
    m1col, m2col = st.columns(2)

    with m1col:
        st.markdown("#### Classification — Beat vs. Miss")
        cl1, cl2, cl3, cl4 = st.columns(4)
        cl1.metric("Logistic Reg Accuracy", "68.5%")
        cl2.metric("Logistic Reg AUC",      "0.701")
        cl3.metric("XGBoost Accuracy",      "68.4%")
        cl4.metric("XGBoost AUC",           "0.713", delta="Primary model")
        st.markdown("AUC = 0.713 → XGBoost correctly ranks a true beat above a true miss **71.3%** of the time. Both beat random guessing (AUC = 0.50).")

    with m2col:
        st.markdown("#### Regression — Predict 3-Day CAR")
        rg1, rg2 = st.columns(2)
        rg1.metric("OLS RMSE",            "0.1179")
        rg2.metric("Random Forest RMSE",  "0.1180")
        st.markdown("RMSE is intentionally modest — 3-day abnormal returns are noisy. Models capture the **systematic** component, not idiosyncratic noise.")

    st.divider()

    # ── PHASE 4 ───────────────────────────────────────────────────────────────
    st.subheader("🚀 Phase 4 — Streamlit App")

    ap1, ap2 = st.columns(2)

    with ap1:
        st.markdown("#### Live Data Flow")
        st.markdown("""
1. User enters ticker → **yfinance API** called live
2. Fetches: `totalAssets`, `debtToEquity`, `returnOnEquity`, `numberOfAnalystOpinions`
3. Historical WRDS dataset queried for prior earnings signals
4. **10-feature vector** built from live + historical data
5. **XGBoost** → beat/miss probability (0-100%)
6. **Random Forest** → predicted 3-day CAR (%)
        """)
        app1, app2 = st.columns(2)
        app1.metric("Code", "GitHub", delta="maheshjw/BA870-Earnings-Predictor")
        app2.metric("Live App", "Streamlit Cloud", delta="No install needed")

    with ap2:
        st.markdown("#### 6 Visual Components")
        st.markdown("""
| # | Visual | What it shows |
|---|--------|---------------|
| 1 | **Beat/Miss Gauge** | XGBoost probability 0-100%, color-coded |
| 2 | **CAR Bar Chart** | Predicted 3-day abnormal return + 1-sigma lines |
| 3 | **Top Metrics Row** | Beat prob (XGB), CAR, Beat prob (LR) |
| 4 | **Historical Scorecard** | Last 8 quarters: EPS, surprise, beat/miss, CAR |
| 5 | **Feature Importance** | XGBoost gain scores for all 10 features |
| 6 | **Scatter Plot** | Prior surprise vs realized CAR, star = current prediction |
        """)

    st.divider()

    # ── 3-DAY CAR DEEP DIVE ───────────────────────────────────────────────────
    st.subheader("📐 3-Day CAR — Deep Dive")

    cd1, cd2 = st.columns(2)
    with cd1:
        st.markdown("**Market model OLS (200-day estimation window):**")
        st.code("R_stock = alpha + beta x R_market + e", language=None)
        st.markdown("**Daily abnormal return:**")
        st.code("AR_t = R_stock_t - (alpha + beta x R_market_t)", language=None)
        st.markdown("**3-day cumulative sum:**")
        st.code("CAR(-1,+1) = AR_-1 + AR_0 + AR_+1", language=None)
        st.markdown("""
| Day | Why included |
|-----|--------------|
| **-1** | Leakage via pre-announcements, options, whispers |
| **0** | Official announcement — primary price discovery |
| **+1** | Analyst revisions + earnings call commentary |
        """)

    with cd2:
        st.markdown("""
| CAR Value | Meaning |
|-----------|---------|
| **+5%** | Stock returned 5% more than market model predicted |
| **-3%** | Underperformed expected return by 3% |
| **~0%** | Announcement fully priced in — no surprise |
        """)
        cav1, cav2 = st.columns(2)
        cav1.metric("Avg CAR in Dataset", "0.11%",  delta="from 110K events")
        cav2.metric("CAR Std Dev",        "~11.8%", delta="wide outcome distribution")
        st.markdown("""
**Why predicted CARs are small (often < 1%):**
The model predicts the *expected* abnormal return.
For consistently-beating stocks like AAPL (75.7% hit rate),
the surprise is partially anticipated — systematic component is small.
The 1-sigma reference lines on the chart show actual outcomes are much wider.
        """)

    st.divider()

    # ── CHART GUIDE ───────────────────────────────────────────────────────────
    st.subheader("📊 Chart Guide")

    with st.expander("🎯 Beat/Miss Probability Gauge", expanded=True):
        g1, g2, g3 = st.columns(3)
        g1.metric("Green zone", "60-100%", delta="Confident beat")
        g2.metric("Yellow zone", "40-60%", delta="Uncertain")
        g3.metric("Red zone",    "0-40%",  delta="Leans miss")
        st.markdown("Delta = distance from the 50% decision boundary. Both XGBoost and Logistic Regression shown — agreement strengthens the signal.")

    with st.expander("📊 Predicted 3-Day CAR Bar Chart"):
        st.markdown("""
- **Green** = positive predicted abnormal return · **Red** = negative
- Value label always printed on bar regardless of size
- **+/-1 sigma dotted lines** = historical spread of realized CARs (reference band)
- Y-axis clamped to 3x predicted value — not blown out by the 11.8% std dev
        """)

    with st.expander("📋 Historical Accuracy Scorecard"):
        st.markdown("""
Last **8 quarters** of actual IBES + CRSP outcomes.

| Column | Source | Meaning |
|--------|--------|---------|
| Date | IBES `anndats_act` | Earnings announcement date |
| Consensus EPS | IBES `meanest` | Mean analyst estimate |
| Actual EPS | IBES `actual` | Reported EPS |
| Surprise | Computed | (Actual - Consensus) / |Consensus| |
| Beat | Computed | checkmark = beat, x = missed |
| 3-Day CAR | CRSP | Realized abnormal return +/-1 day |
        """)
        st.info("Only available for the 1,894 tickers in our WRDS dataset (1990-2024). TSLA, JPM, GOOGL, WMT are not in the dataset — models still run using dataset averages + live yfinance fundamentals.")

    with st.expander("🔍 Feature Importance (XGBoost)"):
        st.markdown("""
Each feature's **gain score** = avg improvement in log-loss when used in a split.

| Rank | Feature | Why it matters |
|------|---------|----------------|
| 1 | `beat_streak` | Firms beating 3-4 consecutive quarters keep beating (sandbagging guidance) |
| 2 | `prior_surprise` | Analysts under-correct — big beat last Q predicts another beat |
| 3 | `avg_surprise_4q` | Sustained beat history = structural expectation-management advantage |
| 4 | `num_analysts` | Heavy coverage = tighter consensus = harder to beat |
| 5 | `dispersion` | High analyst disagreement = genuine pre-announcement uncertainty |
| 6-10 | Fundamentals | log_assets, leverage, roe, prior_car, prior_beat — firm quality context |
        """)

    with st.expander("📉 EPS Surprise vs 3-Day CAR Scatter"):
        st.markdown("""
Each dot = one past earnings event · X = EPS surprise · Y = 3-day CAR · green = beat · red = miss

**Blue star** = current prediction overlaid on historical cloud.

| Quadrant | Surprise | CAR | Interpretation |
|----------|----------|-----|----------------|
| Top-right | + | + | Beat + rewarded — the normal case |
| Bottom-left | - | - | Miss + punished — the normal case |
| Top-left | - | + | Miss but rose — bad news was already priced in |
| Bottom-right | + | - | Beat but fell — buy the rumor sell the news |
        """)

    st.divider()

    # ── SENTIMENT ─────────────────────────────────────────────────────────────
    st.subheader("📰 Sentiment, News & External Event Effects on CAR")

    se1, se2 = st.columns(2)
    with se1:
        st.markdown("""
#### A. 📈 The Expectations Treadmill
Positive pre-earnings media sentiment (30 days before) drives upward analyst revisions.
Higher consensus = harder beat threshold. Stocks sometimes fall on strong earnings because the beat was already priced in.

> **Model gap:** `dispersion` captures uncertainty but not revision direction.

---

#### B. 🏛️ Macro Event Contamination
Fed decisions, CPI prints, geopolitical shocks within the 3-day window contaminate the CAR.
Market-model beta removes broad market moves but not idiosyncratic sector shocks.

> **Model gap:** CARs on macro event days should be interpreted cautiously.
        """)
    with se2:
        st.markdown("""
#### C. 🎙️ Management Guidance Tone
Negative forward guidance + positive EPS beat often produces a **negative CAR**.
Miss + strong guidance often produces a **positive CAR**.
Investors price the outlook, not the past quarter.

> **Model gap:** Guidance language not captured. FinBERT transcript scoring = highest-impact extension.

---

#### D. 📉 Short Interest & IV Crush
High short interest amplifies positive CARs on beats (short squeeze).
High pre-earnings IV collapses post-announcement — realized CAR smaller than implied move (IV crush), especially for mega-caps.

> **Model gap:** `dispersion` proxies uncertainty but not directional positioning.
        """)

    st.divider()
    with st.expander("🔮 Future Extensions"):
        st.markdown("""
- **FinBERT guidance tone** — score earnings call transcripts for forward guidance sentiment
- **Pre-announcement news sentiment** — Alpha Vantage News API or RavenPack 30-day signal
- **Options implied volatility** — IV term structure as uncertainty and magnitude signal
- **FINRA short interest** — directional positioning as a CAR amplifier
- **Fama-French 3-factor CAR** — industry-adjusted benchmark instead of single-factor market model
- **Analyst revision momentum** — direction of estimate changes in 30 days before announcement
- **Whisper numbers** — unofficial EPS expectations that often diverge from published consensus
        """)

    st.caption("BA870 / AC820 · Mahesh Wadhokar · Data: WRDS IBES, CRSP, Compustat · Live: yfinance API")
