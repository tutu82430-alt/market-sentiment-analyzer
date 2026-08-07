"""
╔══════════════════════════════════════════════════════════╗
║   Market Sentiment & Signal Analyzer  —  Streamlit App  ║
║   Author : Rohit                                        ║
║   Stack  : Streamlit · Plotly · yfinance · VADER NLP   ║
╚══════════════════════════════════════════════════════════╝

Run:
    streamlit run app.py
"""

import random
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Sentiment & Signal Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────
# CUSTOM CSS  — dark premium feel
# ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Base */
    .stApp { background-color: #0d1117; }
    section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #161b22, #21262d);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
    }
    div[data-testid="stMetricLabel"] p { color: #8b949e !important; font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; }
    div[data-testid="stMetricValue"]  { color: #e6edf3 !important; font-size: 1.6rem; font-weight: 700; }
    div[data-testid="stMetricDelta"]  { font-size: 0.85rem; font-weight: 600; }

    /* Title */
    .hero-title { font-size: 2.2rem; font-weight: 800; background: linear-gradient(90deg,#58a6ff,#bc8cff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .hero-sub   { color: #8b949e; font-size: 1rem; margin-top: -8px; margin-bottom: 24px; }

    /* Sidebar labels */
    .stSelectbox label, .stSlider label { color: #c9d1d9 !important; font-weight: 600; }

    /* Signal badge */
    .badge-bull { background:#1a3a2a; color:#3fb950; border:1px solid #2ea043; border-radius:20px; padding:4px 14px; font-weight:700; font-size:0.85rem; }
    .badge-bear { background:#3a1a1a; color:#f85149; border:1px solid #da3633; border-radius:20px; padding:4px 14px; font-weight:700; font-size:0.85rem; }
    .badge-neut { background:#1a2a3a; color:#58a6ff; border:1px solid #1f6feb; border-radius:20px; padding:4px 14px; font-weight:700; font-size:0.85rem; }

    /* Expander */
    .streamlit-expanderHeader { color: #c9d1d9 !important; font-weight: 600; background: #161b22; border-radius: 8px; }
    .streamlit-expanderContent { background: #161b22; border: 1px solid #30363d; border-radius: 0 0 8px 8px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# ASSET REGISTRY
# ──────────────────────────────────────────────────────────
ASSETS = {
    "🇮🇳  Nifty 50":       "^NSEI",
    "🇺🇸  S&P 500":        "^GSPC",
    "🇺🇸  NASDAQ 100":     "^NDX",
    "🇩🇪  DAX (Germany)":  "^GDAXI",
    "🛢️  Crude Oil (WTI)": "CL=F",
    "🥇  Gold":            "GC=F",
    "₿   Bitcoin":         "BTC-USD",
    "Ξ   Ethereum":        "ETH-USD",
}

# ──────────────────────────────────────────────────────────
# HEADLINE POOLS
# ──────────────────────────────────────────────────────────
BULLISH = [
    "Inflation cools; markets rally on strong economic data",
    "Record FII inflows lift benchmark indices to multi-month highs",
    "Central bank holds rates; analysts cheer policy stability",
    "IT sector surges after strong quarterly earnings beat",
    "Manufacturing PMI hits 5-year high, signals economic revival",
    "Global risk-on sentiment lifts equities across all sectors",
    "Rupee strengthens; import-heavy sectors gain broadly",
    "Consumer confidence rises to 14-month high",
    "Banking sector leads rally as credit growth accelerates",
    "Auto sector surges on record monthly sales figures",
    "Pharma stocks rise on positive regulatory approvals",
    "Strong GDP data fuels optimism in domestic markets",
    "Foreign investors turn net buyers for fifth consecutive session",
    "Tech stocks soar amid positive guidance from industry leaders",
    "Commodity prices ease; margin relief boosts broader market",
]

BEARISH = [
    "Crude oil spike fans inflation worries; markets slide sharply",
    "Index drops 1.5% on weak global cues and FII outflows",
    "Institutional sell-off intensifies; mid-cap index tumbles",
    "Core inflation rises unexpectedly; rate hike fears return",
    "US recession fears mount; global equities bleed",
    "Regulatory crackdown on fintech rattles investor sentiment",
    "Geopolitical tensions escalate; safe-haven assets surge",
    "Earnings miss triggers broad sell-off in market",
    "Banking shares slide on rising non-performing asset concerns",
    "Weak monsoon outlook dents rural consumption stocks",
    "Corporate margins squeezed as raw material costs soar",
    "Fed signals higher-for-longer rates; emerging markets suffer",
    "Supply chain disruptions weigh heavily on industrial stocks",
    "FII net sellers for third straight week; market outlook dims",
    "Technical breakdown below key support triggers panic selling",
]

NEUTRAL = [
    "Markets trade range-bound ahead of key policy announcement",
    "Mixed global signals keep domestic indices in consolidation",
    "Analysts divided on near-term direction; volumes subdued",
    "PSU stocks flat; private banks edge marginally higher",
    "Q4 results season begins; stocks await earnings clarity",
    "Sector rotation visible as large-caps outperform mid-caps",
    "Market breadth weak despite index gains; caution advised",
    "Options expiry drives choppy session with no clear trend",
    "Investors await US CPI data before making fresh bets",
]

# ──────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def add_technicals(df: pd.DataFrame) -> pd.DataFrame:
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()

    # RSI (14-period Wilder smoothing)
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_l = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def generate_headlines(dates: pd.DatetimeIndex, seed: int = 42) -> pd.Series:
    random.seed(seed)
    pool    = BULLISH + BEARISH + NEUTRAL
    weights = (
        [0.48 / len(BULLISH)] * len(BULLISH)
        + [0.37 / len(BEARISH)] * len(BEARISH)
        + [0.15 / len(NEUTRAL)] * len(NEUTRAL)
    )
    return pd.Series(
        [random.choices(pool, weights=weights, k=1)[0] for _ in dates],
        index=dates, name="Headline",
    )


def score_sentiment(headlines: pd.Series) -> pd.Series:
    analyser = SentimentIntensityAnalyzer()
    return headlines.apply(
        lambda h: analyser.polarity_scores(h)["compound"]
    ).rename("Sentiment_Score")


def build_signals(df: pd.DataFrame, sent_thresh: float = 0.4) -> pd.DataFrame:
    df["Buy_Signal"] = (
        (df["Close"].squeeze() > df["SMA_20"].squeeze()) &
        (df["Sentiment_Score"] > sent_thresh)
    )
    return df


# ──────────────────────────────────────────────────────────
# CHART BUILDER
# ──────────────────────────────────────────────────────────
DARK = "#0d1117"
PANEL = "#161b22"
BORDER = "#30363d"
TEXT = "#c9d1d9"

def build_chart(df: pd.DataFrame, asset_label: str) -> go.Figure:
    close = df["Close"].squeeze()
    sma20 = df["SMA_20"].squeeze()
    sma50 = df["SMA_50"].squeeze()
    rsi   = df["RSI"].squeeze()
    sent  = df["Sentiment_Score"]
    buys  = df[df["Buy_Signal"]]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.58, 0.22, 0.20],
        subplot_titles=("", "RSI (14)", "Daily Sentiment Score"),
    )

    # ── Candlestick ───────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"].squeeze(), high=df["High"].squeeze(),
        low=df["Low"].squeeze(),   close=close,
        increasing_line_color="#3fb950", decreasing_line_color="#f85149",
        increasing_fillcolor="#1a3a2a",  decreasing_fillcolor="#3a1a1a",
        name="Price", showlegend=True,
        whiskerwidth=0.3,
    ), row=1, col=1)

    # ── SMA 20 ────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=sma20, name="SMA 20",
        line=dict(color="#f0883e", width=1.8, dash="solid"),
        opacity=0.9,
    ), row=1, col=1)

    # ── SMA 50 ────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=sma50, name="SMA 50",
        line=dict(color="#bc8cff", width=1.5, dash="dot"),
        opacity=0.8,
    ), row=1, col=1)

    # ── Buy Signal arrows ─────────────────────────────────
    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=buys.index,
            y=(buys["Low"].squeeze() * 0.993),
            mode="markers+text",
            marker=dict(
                symbol="arrow-up", size=14,
                color="#3fb950", line=dict(color="#ffffff", width=1),
            ),
            text=["BUY"] * len(buys),
            textposition="bottom center",
            textfont=dict(color="#3fb950", size=9, family="monospace"),
            name=f"Buy Signal ({len(buys)})",
            hovertemplate=(
                "<b>Buy Signal</b><br>"
                "Date: %{x|%Y-%m-%d}<br>"
                "Close: %{customdata[0]:,.2f}<br>"
                "Sentiment: %{customdata[1]:+.3f}<extra></extra>"
            ),
            customdata=np.stack([
                buys["Close"].squeeze().values,
                buys["Sentiment_Score"].values,
            ], axis=-1),
        ), row=1, col=1)

    # ── RSI ───────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=rsi, name="RSI 14",
        line=dict(color="#58a6ff", width=1.5),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.07)",
    ), row=2, col=1)
    for level, color, dash in [(70, "#f85149", "dash"), (30, "#3fb950", "dash"), (50, BORDER, "dot")]:
        fig.add_hline(y=level, line_color=color, line_dash=dash,
                      line_width=1, opacity=0.6, row=2, col=1)

    # ── Sentiment bars ────────────────────────────────────
    bar_colors = np.where(sent > 0, "#3fb950", "#f85149").tolist()
    fig.add_trace(go.Bar(
        x=df.index, y=sent,
        marker_color=bar_colors, name="Sentiment",
        opacity=0.75, showlegend=False,
    ), row=3, col=1)
    fig.add_hline(y=0.4, line_color="#f0883e", line_dash="dash",
                  line_width=1.2, row=3, col=1,
                  annotation_text="Buy threshold", annotation_font_color="#f0883e",
                  annotation_position="right")

    # ── Layout ────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f"<b>{asset_label}</b> — Market Sentiment & Signal Analyzer",
            font=dict(color="#e6edf3", size=17),
            x=0.01,
        ),
        paper_bgcolor=DARK,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="Inter, Arial"),
        legend=dict(
            bgcolor="#21262d", bordercolor=BORDER, borderwidth=1,
            font=dict(color=TEXT, size=11),
            orientation="h", x=0, y=1.02, xanchor="left", yanchor="bottom",
        ),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#21262d", bordercolor=BORDER, font_color=TEXT),
        margin=dict(l=10, r=10, t=60, b=10),
        height=720,
    )

    for i in range(1, 4):
        fig.update_xaxes(
            showgrid=True, gridcolor=BORDER, gridwidth=0.5,
            zeroline=False, linecolor=BORDER,
            row=i, col=1,
        )
        fig.update_yaxes(
            showgrid=True, gridcolor=BORDER, gridwidth=0.5,
            zeroline=False, linecolor=BORDER,
            row=i, col=1,
        )

    fig.update_yaxes(title_text="Price", title_font_size=11, row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], title_font_size=11, row=2, col=1)
    fig.update_yaxes(title_text="Score", range=[-1.1, 1.1], title_font_size=11, row=3, col=1)

    return fig


# ──────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("---")

    asset_label = st.selectbox(
        "Asset",
        options=list(ASSETS.keys()),
        index=0,
        help="Select the financial instrument to analyze.",
    )
    ticker = ASSETS[asset_label]

    months = st.slider(
        "Date Range (months)", min_value=1, max_value=12, value=3, step=1,
        help="How many months of historical data to load.",
    )

    sent_thresh = st.slider(
        "Sentiment Threshold (Buy)", min_value=0.1, max_value=0.9,
        value=0.4, step=0.05,
        help="Minimum VADER compound score required to trigger a Buy Signal.",
    )

    st.markdown("---")
    st.markdown("### 📘 Signal Logic")
    st.markdown("""
**Buy Signal** fires when:
```
Close > SMA_20
AND
Sentiment > threshold
```
Both conditions must hold simultaneously.
""")

    st.markdown("---")
    st.caption("Data: Yahoo Finance · NLP: VADER · Charts: Plotly")


# ──────────────────────────────────────────────────────────
# HERO HEADER
# ──────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">📈 Market Sentiment & Signal Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Combining technical price analysis with NLP-driven news sentiment to surface high-conviction trading signals.</p>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# DATA PIPELINE
# ──────────────────────────────────────────────────────────
end_date   = datetime.today()
start_date = end_date - timedelta(days=months * 30)

with st.spinner(f"Fetching {asset_label} data from Yahoo Finance…"):
    df = fetch_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

if df.empty:
    st.error("⚠️ Could not fetch data. Check your internet connection or try a different asset.")
    st.stop()

df = add_technicals(df)

headlines = generate_headlines(df.index)
df["Headline"]       = headlines
df["Sentiment_Score"] = score_sentiment(headlines)
df = build_signals(df, sent_thresh)

# ──────────────────────────────────────────────────────────
# METRICS ROW
# ──────────────────────────────────────────────────────────
close_series  = df["Close"].squeeze()
latest_close  = float(close_series.iloc[-1])
prev_close    = float(close_series.iloc[-2]) if len(df) > 1 else latest_close
price_delta   = latest_close - prev_close
sma20_latest  = float(df["SMA_20"].squeeze().dropna().iloc[-1])
rsi_latest    = float(df["RSI"].squeeze().dropna().iloc[-1])
avg_sentiment = float(df["Sentiment_Score"].mean())
n_buy_signals = int(df["Buy_Signal"].sum())
trend         = "🟢 Bullish" if latest_close > sma20_latest else "🔴 Bearish"

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Current Price", f"{latest_close:,.2f}", f"{price_delta:+.2f}")
with col2:
    st.metric("SMA-20", f"{sma20_latest:,.2f}")
with col3:
    rsi_label = "Overbought" if rsi_latest > 70 else ("Oversold" if rsi_latest < 30 else "Neutral")
    st.metric("RSI (14)", f"{rsi_latest:.1f}", rsi_label)
with col4:
    sent_label = "😊 Positive" if avg_sentiment > 0.1 else ("😟 Negative" if avg_sentiment < -0.1 else "😐 Neutral")
    st.metric("Avg Sentiment", f"{avg_sentiment:+.3f}", sent_label)
with col5:
    st.metric("Buy Signals", f"{n_buy_signals}", f"{100*n_buy_signals/len(df):.1f}% of days")

st.markdown("<br>", unsafe_allow_html=True)

# Trend badge
if "Bullish" in trend:
    st.markdown(f'Trend: <span class="badge-bull">{trend}</span>', unsafe_allow_html=True)
elif "Bearish" in trend:
    st.markdown(f'Trend: <span class="badge-bear">{trend}</span>', unsafe_allow_html=True)
else:
    st.markdown(f'Trend: <span class="badge-neut">{trend}</span>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# MAIN CHART
# ──────────────────────────────────────────────────────────
fig = build_chart(df, asset_label)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

# ──────────────────────────────────────────────────────────
# BUY SIGNAL TABLE
# ──────────────────────────────────────────────────────────
st.markdown("### 🟢 Buy Signal Days")
buy_days = df[df["Buy_Signal"]].copy()

if buy_days.empty:
    st.info("No Buy Signals in this period. Try lowering the sentiment threshold or increasing the date range.")
else:
    display = buy_days[["Close", "SMA_20", "RSI", "Sentiment_Score", "Headline"]].copy()
    display.index = display.index.strftime("%Y-%m-%d")
    display.columns = ["Close", "SMA-20", "RSI", "Sentiment Score", "Mock Headline"]

    for col in ["Close", "SMA-20"]:
        display[col] = display[col].apply(lambda x: f"{float(x):,.2f}")
    display["RSI"] = display["RSI"].apply(lambda x: f"{float(x):.1f}")
    display["Sentiment Score"] = display["Sentiment Score"].apply(lambda x: f"{float(x):+.3f}")

    st.dataframe(
        display,
        use_container_width=True,
        height=min(38 * (len(display) + 1) + 10, 420),
    )

# ──────────────────────────────────────────────────────────
# METHODOLOGY EXPANDER
# ──────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📖 Methodology & Limitations — Read Before Trading"):
    st.markdown("""
### How This Signal Works

This analyzer fuses **two independent information layers** to surface potential buying opportunities:

| Layer | Source | Method |
|---|---|---|
| **Price Momentum** | Yahoo Finance OHLCV | 20-day & 50-day Simple Moving Average |
| **Market Sentiment** | Mock financial headlines | VADER compound score (rule-based NLP) |

A **Buy Signal** is generated when *both* conditions agree:
```
Buy Signal = (Close > SMA_20) AND (Sentiment Score > threshold)
```

### Technical Indicators Explained

- **SMA-20**: 20-day Simple Moving Average. When price is above it, the short-term trend is upward. A *lagging* indicator — it confirms trends, not predicts them.
- **SMA-50**: 50-day SMA for medium-term trend context.
- **RSI (14)**: Relative Strength Index. >70 = overbought, <30 = oversold. Helps filter entries.

### ⚠️ Critical Limitations

> **Look-Ahead Bias**: The mock headlines used here are generated randomly and are NOT the actual news from those dates. In a real system, you would only have access to news published *before* the market open. Using future information to generate signals would severely overstate performance in backtesting.

> **Lagging Indicators**: Both the SMA-20 and VADER sentiment (even with real news) are inherently lagging — they describe what *has happened*, not what *will happen*. Signals may fire after the bulk of a move has already occurred.

> **Mock Sentiment**: VADER is a general-purpose lexicon-based model not trained on financial text. Domain-specific models like **FinBERT** (Huang et al., 2023) perform significantly better on financial news classification.

> **No Risk Management**: No position sizing, stop-loss, or portfolio construction logic is applied. This is a signal screener, NOT a trading system.

### Roadmap for Production Use
- Replace mock headlines with live data from **NewsAPI**, **Bloomberg**, or **Refinitiv**
- Upgrade to **FinBERT** or **ProsusAI/finbert** for financial NLP
- Add a proper **backtesting engine** (e.g., `vectorbt`, `backtrader`) with walk-forward validation
- Implement **position sizing** based on Kelly criterion or volatility targeting
- Add **multi-factor scoring** combining RSI, MACD, volume confirmation
""")

# ──────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "⚠️ **Disclaimer:** This tool is for educational and research purposes only. "
    "It does not constitute financial advice. Past performance is not indicative of future results. "
    "Always do your own due diligence before making any investment decisions."
)
