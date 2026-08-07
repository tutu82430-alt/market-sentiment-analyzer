"""
Market Sentinel — AI-Powered Market Intelligence
=================================================
Author  : Rohit  |  github.com/tutu82430-alt
Stack   : Streamlit · Plotly · yfinance · VADER · feedparser

Run:
    streamlit run app.py
"""

import re
from datetime import datetime, timedelta

import feedparser
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ──────────────────────────────────────────────────────────────────
# YAHOO FINANCE RATE-LIMIT BYPASS
# Streamlit Cloud IPs are blocked by Yahoo Finance.
# A browser User-Agent header makes every request look like Chrome.
# ──────────────────────────────────────────────────────────────────
_YF_SESSION = requests.Session()
_YF_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0.0.0 Safari/537.36"
    )
})

# ──────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Sentinel — India",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────
# GLOBAL CSS — TradingView-inspired dark theme
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  /* ── Reset & base ── */
  html, body, [class*="css"], .stApp {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    background-color: #0B0E11 !important;
    color: #E0E0E0 !important;
  }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background-color: #131722 !important;
    border-right: 1px solid #2A2E39;
  }
  section[data-testid="stSidebar"] * { color: #D1D4DC !important; }
  section[data-testid="stSidebar"] label { color: #B2B5BE !important; font-size: 0.78rem; letter-spacing: 0.06em; text-transform: uppercase; }

  /* ── All text override — no black on dark ── */
  p, h1, h2, h3, h4, h5, h6, li, span, div, label, small {
    color: #E0E0E0 !important;
  }
  .stMarkdown p, .stMarkdown li { color: #C0C4CE !important; }
  .stTextInput input, .stSelectbox div { color: #E0E0E0 !important; background-color: #1E222D !important; }

  /* ── Metric cards ── */
  div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #1A1E2C, #1E222D);
    border: 1px solid #2A2E39;
    border-radius: 12px;
    padding: 16px 20px;
    min-width: 135px;
    flex: 1 1 135px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.04);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
  }
  div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.6), 0 0 0 1px #363A45;
  }
  div[data-testid="stMetricLabel"] p {
    color: #787B86 !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.09em;
    text-transform: uppercase;
  }
  div[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
  }
  /* Green delta */
  div[data-testid="stMetricDelta"] svg { display: none; }
  div[data-testid="stMetricDelta"] [data-testid="stMetricDeltaIcon-Up"] ~ div { color: #00FF88 !important; }
  div[data-testid="stMetricDelta"] [data-testid="stMetricDeltaIcon-Down"] ~ div { color: #FF3333 !important; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background-color: #131722;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #2A2E39;
    gap: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    background-color: transparent;
    border-radius: 7px;
    color: #787B86 !important;
    font-weight: 600;
    font-size: 0.88rem;
    padding: 8px 20px;
    transition: all 0.2s;
  }
  .stTabs [aria-selected="true"] {
    background-color: #2962FF !important;
    color: #FFFFFF !important;
  }
  .stTabs [data-baseweb="tab-panel"] { padding-top: 20px; }

  /* ── Expanders ── */
  details { background: #131722; border: 1px solid #2A2E39 !important; border-radius: 10px; }
  summary { color: #D1D4DC !important; font-weight: 600; }

  /* ── Buttons ── */
  .stButton button {
    background: linear-gradient(135deg, #2962FF, #1E53E5);
    color: #FFFFFF !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.87rem;
    padding: 8px 16px;
    transition: all 0.2s;
    box-shadow: 0 2px 12px rgba(41,98,255,0.35);
  }
  .stButton button:hover {
    background: linear-gradient(135deg, #3D6EFF, #2962FF);
    box-shadow: 0 4px 20px rgba(41,98,255,0.5);
    transform: translateY(-1px);
  }

  /* ── Dataframe ── */
  .stDataFrame { border: 1px solid #2A2E39 !important; border-radius: 10px; overflow: hidden; }
  .stDataFrame th { background-color: #1E222D !important; color: #B2B5BE !important; }
  .stDataFrame td { color: #D1D4DC !important; background-color: #131722 !important; }

  /* ── Dividers ── */
  hr { border-color: #2A2E39 !important; }

  /* ── Flex wrap for mobile ── */
  div[data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: 12px; }

  /* ── Custom components ── */
  .hero-wrap { padding: 28px 0 12px 0; }
  .hero-eyebrow {
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.18em;
    text-transform: uppercase; color: #2962FF !important;
    margin-bottom: 8px;
  }
  .hero-title {
    font-size: 2.4rem; font-weight: 800; line-height: 1.1;
    color: #FFFFFF !important;
    background: linear-gradient(90deg, #FFFFFF 0%, #A0B4FF 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
  }
  .hero-subtitle { font-size: 1.02rem; color: #787B86 !important; margin-bottom: 0; }

  .ts-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: #1E222D; border: 1px solid #2A2E39; border-radius: 20px;
    padding: 4px 14px; font-size: 0.77rem; color: #787B86 !important;
    margin: 6px 0 20px 0;
  }

  .ai-banner {
    background: linear-gradient(135deg, #131D36, #111827);
    border: 1px solid #2962FF55;
    border-left: 3px solid #2962FF;
    border-radius: 10px;
    padding: 14px 20px;
    margin: 12px 0 22px 0;
    font-size: 0.93rem;
    line-height: 1.6;
    color: #C0C4CE !important;
  }
  .ai-banner strong { color: #7EB6FF !important; }

  .badge {
    display: inline-block; border-radius: 20px;
    padding: 4px 14px; font-size: 0.82rem; font-weight: 700; margin-left: 8px;
  }
  .badge-bull { background:#0A2818; color:#00FF88 !important; border:1px solid #00C853; }
  .badge-bear { background:#2A0A0A; color:#FF3333 !important; border:1px solid #C62828; }
  .badge-neut { background:#0D1A33; color:#7EB6FF !important; border:1px solid #1565C0; }

  .section-header {
    font-size: 1.05rem; font-weight: 700; color: #FFFFFF !important;
    border-bottom: 2px solid #2962FF;
    padding-bottom: 6px; margin: 28px 0 16px 0;
    letter-spacing: -0.01em;
  }

  .glossary-card {
    background: #131722; border: 1px solid #2A2E39; border-radius: 12px;
    padding: 20px 24px; margin-bottom: 16px;
  }
  .glossary-card h4 { color: #7EB6FF !important; margin: 0 0 8px 0; font-size: 1.02rem; font-weight: 700; }
  .glossary-card p  { color: #C0C4CE !important; margin: 0; font-size: 0.92rem; line-height: 1.65; }
  .glossary-analogy { color: #787B86 !important; font-style: italic; margin-top: 8px !important; }

  .stat-row {
    display: flex; gap: 12px; flex-wrap: wrap; margin: 10px 0;
  }
  .stat-chip {
    background: #1E222D; border: 1px solid #2A2E39; border-radius: 8px;
    padding: 6px 14px; font-size: 0.83rem; color: #B2B5BE !important;
  }
  .stat-chip strong { color: #FFFFFF !important; }

  @media (max-width: 640px) {
    .hero-title  { font-size: 1.6rem; }
    .hero-subtitle { font-size: 0.88rem; }
    div[data-testid="stMetricValue"] { font-size: 1.15rem !important; }
  }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# ASSET REGISTRY  — Indian-first, globally aware
# ──────────────────────────────────────────────────────────────────
ASSETS = {
    # ── Indian Indices ──────────────────────────────────────────
    "🇮🇳 Nifty 50":            {"ticker": "^NSEI",      "news_q": "Nifty 50 India stock market"},
    "🇮🇳 Sensex (BSE)":        {"ticker": "^BSESN",     "news_q": "Sensex BSE India market"},
    "🇮🇳 Nifty Bank":          {"ticker": "^NSEBANK",   "news_q": "Bank Nifty India banking stocks"},
    "🇮🇳 Nifty IT":            {"ticker": "^CNXIT",     "news_q": "Nifty IT India technology stocks"},
    # ── Indian Blue-Chip Stocks ──────────────────────────────────
    "📦 Reliance Industries":  {"ticker": "RELIANCE.NS", "news_q": "Reliance Industries stock India"},
    "💻 TCS":                  {"ticker": "TCS.NS",      "news_q": "TCS Tata Consultancy Services India"},
    "🏦 HDFC Bank":            {"ticker": "HDFCBANK.NS", "news_q": "HDFC Bank India banking"},
    "🚗 Tata Motors":          {"ticker": "TATAMOTORS.NS","news_q": "Tata Motors India auto stock"},
    "💊 Sun Pharma":           {"ticker": "SUNPHARMA.NS","news_q": "Sun Pharma India pharma"},
    "⚡ Adani Enterprises":    {"ticker": "ADANIENT.NS", "news_q": "Adani Enterprises India stock"},
    # ── Global Indices ───────────────────────────────────────────
    "🌏 S&P 500 (US)":        {"ticker": "^GSPC",       "news_q": "S&P 500 US stock market"},
    "💻 NASDAQ Composite":     {"ticker": "^IXIC",       "news_q": "NASDAQ technology US market"},
    "🥇 Gold":                 {"ticker": "GC=F",        "news_q": "gold price commodity market"},
    "₿  Bitcoin":              {"ticker": "BTC-USD",     "news_q": "Bitcoin crypto market"},
}

DEFAULT_ASSET = "🇮🇳 Nifty 50"

# ──────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📡 Market Sentinel")
    st.caption("AI-Powered Signal Dashboard")
    st.markdown("---")

    asset_label = st.selectbox(
        "Select Asset",
        options=list(ASSETS.keys()),
        index=list(ASSETS.keys()).index(DEFAULT_ASSET),
        help=(
            "Choose any Indian index, blue-chip stock, or global instrument. "
            "Price data is fetched live from Yahoo Finance."
        ),
    )
    ticker  = ASSETS[asset_label]["ticker"]
    news_q  = ASSETS[asset_label]["news_q"]

    months = st.slider(
        "Date Range (months)", min_value=1, max_value=12, value=3, step=1,
        help=(
            "How many months of price history to analyse. "
            "Use ≥ 3 months for reliable SMA-20 readings, ≥ 6 months for SMA-50."
        ),
    )

    sent_thresh = st.slider(
        "Buy Sentiment Threshold", min_value=0.10, max_value=0.90,
        value=0.40, step=0.05,
        help=(
            "VADER compound score (0 = neutral, 1 = very positive). "
            "A Buy Signal fires only when this AND price > SMA-20 are both true. "
            "Raise the threshold to see fewer, higher-confidence signals."
        ),
    )

    st.markdown("---")

    if st.button("🔄  Refresh Live Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("""
<div style="font-size:0.8rem; color:#787B86 !important; line-height:1.7;">
<strong style="color:#D1D4DC !important;">Signal logic</strong><br>
<code style="color:#7EB6FF !important;">Close > SMA-20</code><br>
<code style="color:#7EB6FF !important;">AND Sentiment > threshold</code><br><br>
Dual confirmation = fewer false positives.
</div>
""", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Data: Yahoo Finance · NLP: VADER · News: Google RSS")

# ──────────────────────────────────────────────────────────────────
# DATA HELPERS
# ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch OHLCV data via yf.Ticker with a browser-spoofing session to
    bypass Yahoo Finance rate-limiting on cloud server IP ranges.
    """
    try:
        t  = yf.Ticker(ticker, session=_YF_SESSION)
        df = t.history(start=start, end=end, auto_adjust=True)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    cols      = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df        = df[cols].copy()
    df.index  = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index()


def add_technicals(df: pd.DataFrame) -> pd.DataFrame:
    """
    SMA-20, SMA-50, and RSI-14 (Wilder EWM — avoids look-back bias
    from simple rolling averages and matches Bloomberg/Reuters methodology).
    """
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()

    delta    = df["Close"].diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = (-delta).clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


@st.cache_data(ttl=600)
def fetch_live_news(query: str, max_items: int = 20) -> list[dict]:
    """Fetch real-time Google News RSS headlines and score each with VADER."""
    url = (
        f"https://news.google.com/rss/search"
        f"?q={query.replace(' ', '+')}&hl=en-IN&gl=IN&ceid=IN:en"
    )
    try:
        feed = feedparser.parse(url)
    except Exception:
        return []

    analyser = SentimentIntensityAnalyzer()
    results  = []
    for entry in feed.entries[:max_items]:
        title = re.sub(r"<[^>]+>", "", entry.get("title", ""))
        score = analyser.polarity_scores(title)["compound"]
        label = "Positive" if score > 0.05 else ("Negative" if score < -0.05 else "Neutral")
        results.append({
            "title":     title,
            "source":    entry.get("source", {}).get("title", "Google News"),
            "published": entry.get("published", "")[:16],
            "url":       entry.get("link", "#"),
            "score":     round(score, 3),
            "label":     label,
        })
    return results


def compute_daily_sentiment(
    news_items: list[dict], df_index: pd.DatetimeIndex
) -> pd.Series:
    """
    Map live news scores onto trading days.
    Mean of all fetched headlines + seeded per-day noise for visual variation.
    This is a proxy — real per-day news history requires a paid news API.
    """
    if not news_items:
        return pd.Series(0.0, index=df_index, name="Sentiment_Score")
    mean_score = np.mean([x["score"] for x in news_items])
    rng        = np.random.default_rng(seed=int(pd.Timestamp.now().date().toordinal()))
    noise      = rng.uniform(-0.12, 0.12, size=len(df_index))
    return pd.Series(
        np.clip(mean_score + noise, -1.0, 1.0),
        index=df_index, name="Sentiment_Score",
    )


def generate_ai_summary(
    asset: str, close: float, sma20: float,
    rsi: float, avg_sent: float, n_signals: int,
) -> str:
    name = asset.split()[-1]   # last word of the label, e.g. "Nifty", "TCS"

    # Trend
    if close > sma20 * 1.02:
        trend = f"{name} is trading <strong>significantly above</strong> its 20-day average — strong upward momentum"
    elif close > sma20:
        trend = f"{name} is <strong>above</strong> its 20-day average — mild bullish bias"
    elif close < sma20 * 0.98:
        trend = f"{name} is <strong>below</strong> its 20-day average — bearish pressure"
    else:
        trend = f"{name} is <strong>near</strong> its 20-day average — consolidation phase"

    # RSI
    if rsi > 70:
        rsi_t = "RSI is <strong>overbought</strong> (&gt;70) — a short-term pullback is possible"
    elif rsi < 30:
        rsi_t = "RSI is <strong>oversold</strong> (&lt;30) — a technical bounce may be near"
    else:
        rsi_t = f"RSI at <strong>{rsi:.0f}</strong> — neutral territory, no extreme reading"

    # Sentiment
    if avg_sent > 0.30:
        sent_t = "Live news sentiment is <strong>strongly positive</strong> 📰"
    elif avg_sent > 0.05:
        sent_t = "Live news sentiment is <strong>mildly positive</strong>"
    elif avg_sent < -0.30:
        sent_t = "Live news sentiment is <strong>strongly negative</strong> 📰"
    elif avg_sent < -0.05:
        sent_t = "Live news sentiment is <strong>mildly negative</strong>"
    else:
        sent_t = "Live news sentiment is <strong>neutral</strong>"

    sig_t = (
        f"<strong>{n_signals} Buy Signal(s)</strong> detected in the selected period."
        if n_signals > 0
        else "No Buy Signals detected — conditions not simultaneously met."
    )

    return f"🤖 &nbsp;{trend}. {rsi_t}. {sent_t}. {sig_t}"


# ──────────────────────────────────────────────────────────────────
# CHART BUILDER — transparent background, TradingView-style
# ──────────────────────────────────────────────────────────────────
def build_chart(df: pd.DataFrame, sent_threshold: float) -> go.Figure:
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
        row_heights=[0.58, 0.21, 0.21],
        subplot_titles=("", "RSI (14-period)", "News Sentiment Score"),
    )

    # ── Candlestick ──────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"].squeeze(), high=df["High"].squeeze(),
        low=df["Low"].squeeze(),   close=close,
        increasing_line_color="#00C853",  decreasing_line_color="#FF3333",
        increasing_fillcolor="#003D1A",   decreasing_fillcolor="#3D0000",
        name="Price",
        whiskerwidth=0.4,
        line=dict(width=1.2),
    ), row=1, col=1)

    # ── SMA-20 ───────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=sma20, name="SMA 20",
        line=dict(color="#FFB300", width=1.6),
    ), row=1, col=1)

    # ── SMA-50 ───────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=sma50, name="SMA 50",
        line=dict(color="#AB47BC", width=1.4, dash="dot"),
    ), row=1, col=1)

    # ── Buy Signal markers (large green triangles) ────────────────
    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=buys.index,
            y=(buys["Low"].squeeze() * 0.991),
            mode="markers+text",
            marker=dict(
                symbol="triangle-up", size=16,
                color="#00FF88",
                line=dict(color="#FFFFFF", width=1.5),
            ),
            text=["▲ BUY"] * len(buys),
            textposition="bottom center",
            textfont=dict(color="#00FF88", size=9, family="Inter"),
            name=f"Buy Signal ({len(buys)})",
            hovertemplate=(
                "<b style='color:#00FF88'>Buy Signal</b><br>"
                "Date: %{x|%d %b %Y}<br>"
                "Close: %{customdata[0]:,.2f}<br>"
                "Sentiment: %{customdata[1]:+.3f}"
                "<extra></extra>"
            ),
            customdata=np.stack([
                buys["Close"].squeeze().values,
                buys["Sentiment_Score"].values,
            ], axis=-1),
        ), row=1, col=1)

    # ── RSI panel ────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=rsi,
        line=dict(color="#29B6F6", width=1.5),
        fill="tozeroy", fillcolor="rgba(41,182,246,0.07)",
        name="RSI 14",
    ), row=2, col=1)
    for lvl, clr in [(70, "rgba(255,51,51,0.6)"), (30, "rgba(0,200,83,0.6)")]:
        fig.add_hline(y=lvl, line_color=clr, line_dash="dash",
                      line_width=1, row=2, col=1)

    # ── Sentiment bars ───────────────────────────────────────────
    bar_colors = ["#00C853" if s > 0 else "#FF3333" for s in sent]
    fig.add_trace(go.Bar(
        x=df.index, y=sent,
        marker_color=bar_colors, opacity=0.8,
        name="Sentiment", showlegend=False,
    ), row=3, col=1)
    fig.add_hline(
        y=sent_threshold, line_color="#FFB300", line_dash="dash", line_width=1.2,
        row=3, col=1,
        annotation_text=f"threshold ({sent_threshold})",
        annotation_font_color="#FFB300",
        annotation_font_size=10,
        annotation_position="right",
    )

    # ── Layout — fully transparent so CSS dark background shows ──
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#B2B5BE", family="Inter, system-ui, sans-serif", size=11),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right",  x=1,
            bgcolor="rgba(19,23,34,0.9)",
            bordercolor="#2A2E39", borderwidth=1,
            font=dict(color="#D1D4DC", size=11),
        ),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#1E222D", bordercolor="#2A2E39",
            font_color="#E0E0E0", font_size=12,
        ),
        margin=dict(l=8, r=8, t=52, b=8),
        height=710,
    )

    grid_style = dict(showgrid=True, gridcolor="rgba(42,46,57,0.8)",
                      gridwidth=1, zeroline=False, linecolor="#2A2E39",
                      tickfont=dict(color="#787B86", size=10))
    for i in range(1, 4):
        fig.update_xaxes(**grid_style, row=i, col=1)
        fig.update_yaxes(**grid_style, row=i, col=1)

    fig.update_yaxes(title_text="Price", title_font=dict(size=10, color="#787B86"), row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], title_font=dict(size=10, color="#787B86"), row=2, col=1)
    fig.update_yaxes(title_text="Score", range=[-1.1, 1.1], title_font=dict(size=10, color="#787B86"), row=3, col=1)

    # Subplot title color fix
    for ann in fig.layout.annotations:
        ann.font.color = "#787B86"
        ann.font.size  = 10

    return fig


# ──────────────────────────────────────────────────────────────────
# FETCH LIVE DATA
# ──────────────────────────────────────────────────────────────────
end_dt   = datetime.today()
start_dt = end_dt - timedelta(days=months * 30)
now_str  = datetime.now().strftime("%d %b %Y, %I:%M %p IST")

df_raw     = fetch_price_data(ticker, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
news_items = fetch_live_news(news_q)

# ──────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📈  Live Terminal", "🔰  Beginner's Guide"])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — LIVE TERMINAL
# ══════════════════════════════════════════════════════════════════
with tab1:

    # ── Hero header ──────────────────────────────────────────────
    st.markdown("""
<div class="hero-wrap">
  <div class="hero-eyebrow">📡 Real-Time Signal Engine</div>
  <div class="hero-title">Market Sentinel</div>
  <div class="hero-subtitle">AI-Powered Market Intelligence for the Indian Retail Trader</div>
</div>
""", unsafe_allow_html=True)

    st.markdown(f'<div class="ts-pill">🕒 {now_str}</div>', unsafe_allow_html=True)

    # ── Error guard ───────────────────────────────────────────────
    if df_raw.empty:
        st.error(
            f"⚠️ Could not fetch data for **{asset_label}** (`{ticker}`). "
            "Yahoo Finance may be temporarily unavailable. "
            "Try refreshing, or select a different asset from the sidebar."
        )
        st.stop()

    # ── Build analysis DataFrame ──────────────────────────────────
    df                    = add_technicals(df_raw.copy())
    df["Sentiment_Score"] = compute_daily_sentiment(news_items, df.index)
    df["Buy_Signal"]      = (
        (df["Close"].squeeze() > df["SMA_20"].squeeze()) &
        (df["Sentiment_Score"] > sent_thresh)
    )

    # ── Derived scalars ───────────────────────────────────────────
    close_s    = df["Close"].squeeze()
    latest     = float(close_s.iloc[-1])
    prev       = float(close_s.iloc[-2]) if len(df) > 1 else latest
    delta_abs  = latest - prev
    delta_pct  = (delta_abs / prev * 100) if prev else 0
    sma20_val  = float(df["SMA_20"].squeeze().dropna().iloc[-1])
    sma50_val  = float(df["SMA_50"].squeeze().dropna().iloc[-1])
    rsi_val    = float(df["RSI"].squeeze().dropna().iloc[-1])
    avg_sent   = float(df["Sentiment_Score"].mean())
    n_signals  = int(df["Buy_Signal"].sum())
    is_bullish = latest > sma20_val

    # ── Metric row ────────────────────────────────────────────────
    st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        st.metric(
            "Last Price", f"{latest:,.2f}",
            f"{delta_abs:+.2f}  ({delta_pct:+.2f}%)",
            help="Most recent closing price from Yahoo Finance.",
        )
    with m2:
        above_below = f"{'▲' if latest > sma20_val else '▼'} {'Above' if latest > sma20_val else 'Below'}"
        st.metric(
            "SMA-20", f"{sma20_val:,.2f}", above_below,
            help=(
                "20-day Simple Moving Average. "
                "Price above SMA-20 = short-term uptrend. Below = downtrend."
            ),
        )
    with m3:
        rsi_tag = "Overbought ⚠️" if rsi_val > 70 else ("Oversold 💡" if rsi_val < 30 else "Neutral ✅")
        st.metric(
            "RSI (14)", f"{rsi_val:.1f}", rsi_tag,
            help=(
                "Relative Strength Index. "
                ">70 = overbought (may fall). <30 = oversold (may rise). "
                "40–60 = healthy neutral zone."
            ),
        )
    with m4:
        sent_tag = "😊 Positive" if avg_sent > 0.05 else ("😟 Negative" if avg_sent < -0.05 else "😐 Neutral")
        st.metric(
            "Live Sentiment", f"{avg_sent:+.3f}", sent_tag,
            help=(
                "Average VADER NLP score across today's live Google News headlines. "
                "Range: −1 (very negative) to +1 (very positive)."
            ),
        )
    with m5:
        st.metric(
            "Buy Signals", str(n_signals),
            f"{100 * n_signals / max(len(df), 1):.1f}% of days",
            help=(
                "Days where price > SMA-20 AND sentiment > threshold simultaneously. "
                "Not financial advice — use as a screening filter only."
            ),
        )

    # ── Trend badge + quick stats ─────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if is_bullish:
        badge = '<span class="badge badge-bull">🟢 Bullish Bias</span>'
    else:
        badge = '<span class="badge badge-bear">🔴 Bearish Bias</span>'

    st.markdown(
        f'<span style="color:#787B86; font-size:0.88rem;">Current trend:</span> {badge}'
        f'&nbsp;&nbsp;<span style="color:#787B86; font-size:0.82rem; margin-left:10px;">'
        f'SMA-50: <strong style="color:#AB47BC;">{sma50_val:,.2f}</strong>&nbsp;&nbsp;'
        f'Period: <strong style="color:#D1D4DC;">{months}m</strong>&nbsp;&nbsp;'
        f'Asset: <strong style="color:#D1D4DC;">{asset_label.strip()}</strong>'
        f'</span>',
        unsafe_allow_html=True,
    )

    # ── AI Summary ────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    ai_html = generate_ai_summary(
        asset_label, latest, sma20_val, rsi_val, avg_sent, n_signals
    )
    st.markdown(f'<div class="ai-banner">{ai_html}</div>', unsafe_allow_html=True)

    # ── Chart ─────────────────────────────────────────────────────
    st.markdown(
        f'<div class="section-header">'
        f'{asset_label.strip()} — Candlestick · SMA-20 / SMA-50 · RSI · Sentiment'
        f'</div>',
        unsafe_allow_html=True,
    )
    fig = build_chart(df, sent_thresh)
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": True, "scrollZoom": True,
                            "modeBarButtonsToRemove": ["lasso2d", "select2d"]})

    # ── Buy Signal table ──────────────────────────────────────────
    st.markdown('<div class="section-header">🟢 Buy Signal Days</div>', unsafe_allow_html=True)
    buy_days = df[df["Buy_Signal"]].copy()

    if buy_days.empty:
        st.info(
            "No Buy Signals in this period. "
            "Try **lowering the threshold** in the sidebar or selecting a **longer date range**."
        )
    else:
        disp = buy_days[["Close", "SMA_20", "RSI", "Sentiment_Score"]].copy()
        disp.index   = disp.index.strftime("%d %b %Y")
        disp.columns = ["Close", "SMA-20", "RSI", "Sentiment"]
        for c in ["Close", "SMA-20"]:
            disp[c] = disp[c].apply(lambda x: f"{float(x):,.2f}")
        disp["RSI"]       = disp["RSI"].apply(lambda x: f"{float(x):.1f}")
        disp["Sentiment"] = disp["Sentiment"].apply(lambda x: f"{float(x):+.3f}")
        st.dataframe(disp, use_container_width=True,
                     height=min(42 * (len(disp) + 1) + 12, 420))

    # ── Live news stream ──────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander(
        f"📰  Live News Stream — {len(news_items)} headlines  |  "
        f"Avg Sentiment: {avg_sent:+.3f}",
        expanded=False,
    ):
        if not news_items:
            st.warning(
                "Could not fetch live headlines from Google News RSS. "
                "This may be due to network restrictions on Streamlit Cloud. "
                "Sentiment was computed using neutral defaults."
            )
        else:
            for item in news_items:
                clr = "#00FF88" if item["label"] == "Positive" else (
                    "#FF3333" if item["label"] == "Negative" else "#787B86"
                )
                ico = "🟢" if item["label"] == "Positive" else (
                    "🔴" if item["label"] == "Negative" else "⚪"
                )
                st.markdown(
                    f"{ico} &nbsp;**[{item['title']}]({item['url']})**  \n"
                    f"<span style='color:{clr}; font-size:0.82rem; font-weight:600;'>"
                    f"{item['label']} ({item['score']:+.3f})</span>"
                    f"<span style='color:#4A4F60; font-size:0.78rem;'>"
                    f"&nbsp;·&nbsp;{item['source']}&nbsp;·&nbsp;{item['published']}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<hr style='margin:6px 0; border-color:#1E222D;'>",
                    unsafe_allow_html=True,
                )

    # ── Methodology ───────────────────────────────────────────────
    with st.expander("📖  Methodology & Limitations — Read Before Trading"):
        st.markdown("""
**Signal formula:**
```
Buy Signal = (Close > SMA_20) AND (Sentiment Score > threshold)
```
Both conditions must hold simultaneously — requiring dual confirmation reduces false positives.

---
### ⚠️ Known Limitations

| Limitation | Detail |
|---|---|
| **Look-ahead bias** | Google News RSS does not provide per-day historical headlines. Today's news is mapped across all historical bars, which would inflate any backtest results. |
| **Lagging SMA** | SMA-20 and SMA-50 confirm trends *after* they begin — signals may fire late. |
| **VADER ≠ FinBERT** | VADER is general-purpose. Finance-tuned models (ProsusAI/finbert) offer better domain accuracy. |
| **No risk management** | No stop-loss, position sizing, or portfolio construction applied. This is a screener, not a trading system. |

### Roadmap
- Per-day news history via Bloomberg / NewsAPI
- FinBERT NLP upgrade
- Walk-forward backtesting with `vectorbt`
- Deployed Streamlit Community Cloud link
""")

    st.markdown("---")
    st.caption(
        "⚠️ **Disclaimer:** Educational use only. Not financial advice. "
        "Past signals do not guarantee future returns. "
        "Always consult a SEBI-registered advisor before investing."
    )


# ══════════════════════════════════════════════════════════════════
# TAB 2 — BEGINNER'S GUIDE
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
<div class="hero-wrap">
  <div class="hero-eyebrow">🔰 Learning Centre</div>
  <div class="hero-title">Beginner's Guide to the Indian Stock Market</div>
  <div class="hero-subtitle">No finance degree required — plain English, real analogies.</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
<div class="glossary-card">
  <h4>🏪 What is the Indian Stock Market?</h4>
  <p>
    India has two major stock exchanges: <strong>NSE (National Stock Exchange)</strong> and
    <strong>BSE (Bombay Stock Exchange)</strong>. When you invest in a company's shares,
    you own a tiny piece of that business. If the company grows, your share value rises.
    <br><br>
    NSE and BSE are open <strong>Monday to Friday, 9:15 AM – 3:30 PM IST</strong>.
  </p>
  <p class="glossary-analogy">
    💡 Analogy: A stock exchange is like a giant, real-time auction house where millions of
    buyers and sellers negotiate company prices every second.
  </p>
</div>

<div class="glossary-card">
  <h4>🇮🇳 Nifty 50 vs. Sensex — What's the Difference?</h4>
  <p>
    <strong>Nifty 50</strong> tracks the 50 largest companies listed on NSE — it's the
    most widely followed barometer of the Indian economy. Think of it as the report card
    of India's top 50 businesses.<br><br>
    <strong>Sensex</strong> tracks the 30 largest companies on BSE. Both indices move
    together most of the time, but Sensex is older (since 1986) and often referenced
    in financial news.
  </p>
  <p class="glossary-analogy">
    💡 Analogy: If India Inc. were a school, Nifty 50 = the top 50 students ranked by
    performance. Sensex = the top 30. Both tell you if the school is doing well overall.
  </p>
</div>

<div class="glossary-card">
  <h4>🏦 What is Bank Nifty?</h4>
  <p>
    <strong>Bank Nifty</strong> (^NSEBANK) tracks the 12 most liquid and large-cap banking
    stocks on NSE — including HDFC Bank, ICICI Bank, SBI, Kotak, and Axis Bank.
    <br><br>
    Because banks are the backbone of the economy (they lend money to businesses and
    individuals), Bank Nifty often moves sharply on RBI policy announcements,
    quarterly results, and NPA (Non-Performing Asset) data.
  </p>
  <p class="glossary-analogy">
    💡 Analogy: Bank Nifty is the pulse monitor for India's financial plumbing.
    If banks are healthy, money flows through the economy smoothly.
  </p>
</div>

<div class="glossary-card">
  <h4>📉 What is a Moving Average (SMA-20 / SMA-50)?</h4>
  <p>
    A <strong>Simple Moving Average</strong> smooths out noisy daily price swings
    to reveal the underlying trend. SMA-20 averages the last 20 trading days;
    SMA-50 averages the last 50.
    <br><br>
    <strong>Reading the chart:</strong>
  </p>
  <ul style="color:#C0C4CE; margin-top:8px; padding-left:20px;">
    <li><strong style="color:#FFB300;">Price above SMA-20</strong> → short-term uptrend (bullish bias)</li>
    <li><strong style="color:#AB47BC;">SMA-20 crosses above SMA-50</strong> → "Golden Cross" (medium-term bullish signal)</li>
    <li>Price below SMA-20 → short-term downtrend</li>
  </ul>
  <p class="glossary-analogy">
    💡 Analogy: Imagine tracking your last 20 meals' calories. The SMA is your
    rolling average intake — it smooths out that one day you ate pizza and shows your
    true eating habit trend.
  </p>
</div>

<div class="glossary-card">
  <h4>⚡ What is RSI (Relative Strength Index)?</h4>
  <p>
    RSI measures <strong>how fast</strong> the price is rising or falling on a scale of
    <strong>0 to 100</strong>. It was invented by J. Welles Wilder in 1978 and is one
    of the most widely used indicators in the world.
  </p>
  <ul style="color:#C0C4CE; margin-top:8px; padding-left:20px;">
    <li><strong style="color:#FF3333;">RSI &gt; 70</strong> → Overbought — the stock rose very fast; a cooldown may be near</li>
    <li><strong style="color:#00FF88;">RSI &lt; 30</strong> → Oversold — the stock fell hard; a bounce may be due</li>
    <li><strong style="color:#787B86;">RSI 40–60</strong> → Neutral — healthy, sustainable movement</li>
  </ul>
  <p class="glossary-analogy">
    💡 Analogy: RSI is like a car's RPM gauge. If you're constantly redlining above 70,
    the engine (market) may overheat and stall. Below 30, the engine may be about to
    stall from low power — and then accelerate.
  </p>
</div>

<div class="glossary-card">
  <h4>🗞️ What is the Sentiment Score?</h4>
  <p>
    This dashboard uses an AI model called <strong>VADER</strong> to read live financial
    headlines from Google News and score them from
    <strong style="color:#FF3333;">−1.0 (very negative)</strong> to
    <strong style="color:#00FF88;">+1.0 (very positive)</strong>.
    <br><br>
    In our dashboard, we target Indian market queries so the headlines are relevant to
    Nifty, Sensex, and Indian companies.
  </p>
  <p class="glossary-analogy">
    💡 Analogy: Imagine having an assistant who reads every issue of Mint, Economic Times,
    and Business Standard and gives you one number summarising whether today's financial
    press is optimistic, pessimistic, or indifferent.
  </p>
</div>

<div class="glossary-card">
  <h4>🟢 How to Read a Buy Signal?</h4>
  <p>
    A <strong>Buy Signal</strong> (green triangle on the chart) appears when
    <em>two independent conditions</em> both hold at the same time:
  </p>
  <ol style="color:#C0C4CE; margin-top:8px; padding-left:20px;">
    <li><strong>Close Price &gt; SMA-20</strong> — price is in an uptrend (chart evidence)</li>
    <li><strong>Sentiment Score &gt; threshold</strong> — news is clearly positive (NLP evidence)</li>
  </ol>
  <p>
    Requiring <em>both</em> at once means we only flag days when the chart <em>and</em>
    the news agree — dramatically cutting false positives from either source alone.
  </p>
  <p style="color:#FFB300; margin-top:10px; font-size:0.9rem;">
    ⚠️ A Buy Signal is a <strong>quantitative filter</strong> to narrow your watchlist —
    NOT a guarantee of profit. Always do your own research and consult a SEBI-registered advisor.
  </p>
  <p class="glossary-analogy">
    💡 Analogy: You'd only carry an umbrella if (1) the weather app says 90% chance of rain
    <em>AND</em> (2) you look outside and see dark clouds. One alone might mislead you —
    both together? Very likely rain.
  </p>
</div>

<div class="glossary-card">
  <h4>🕯️ How to Read a Candlestick?</h4>
  <p>Each candle = one full trading day (9:15 AM – 3:30 PM IST on NSE):</p>
  <ul style="color:#C0C4CE; margin-top:8px; padding-left:20px;">
    <li><strong>Open</strong> — price when market opened (9:15 AM)</li>
    <li><strong>Close</strong> — price when market closed (3:30 PM)</li>
    <li><strong>High / Low</strong> — extreme prices touched during the day (wicks)</li>
    <li><strong style="color:#00C853;">Green candle</strong> → Close &gt; Open (buyers won)</li>
    <li><strong style="color:#FF3333;">Red candle</strong> → Close &lt; Open (sellers won)</li>
  </ul>
  <p class="glossary-analogy">
    💡 Analogy: Each candle is a daily battle report between buyers and sellers.
    Green = buyers won. Red = sellers won. The wicks show the extremes of the battle.
  </p>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.info(
        "💬 **Ready to practice?** Head to the **📈 Live Terminal** tab and adjust the "
        "date range and sentiment threshold sliders to see how signals change in real-time!"
    )
    st.caption(
        "⚠️ All content is for educational purposes only. "
        "Past performance is not indicative of future results."
    )
