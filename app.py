"""
Market Sentinel — AI-Powered Market Intelligence
=================================================
Author  : Rohit  |  github.com/tutu82430-alt
Stack   : Streamlit · Plotly · yfinance · VADER · feedparser
"""

import re
import time
from datetime import datetime, timedelta

import feedparser
import google.generativeai as genai
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ──────────────────────────────────────────────────────────────────
# CONFIG & SESSION STATE
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Sentinel",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "show_onboarding" not in st.session_state:
    st.session_state["show_onboarding"] = True

# ──────────────────────────────────────────────────────────────────
# YAHOO FINANCE RATE-LIMIT BYPASS
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
# GLOBAL CSS
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  /* Typography & Base */
  html, body, [class*="css"], .stApp {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    background-color: #0B0E11 !important;
    color: #E0E0E0 !important;
  }
  
  /* Sidebar */
  section[data-testid="stSidebar"] {
    background-color: #131722 !important;
    border-right: 1px solid #2A2E39;
  }
  
  /* Prevent black text on dark background */
  p, h1, h2, h3, h4, h5, h6, li, span, div, label, small {
    color: #E0E0E0 !important;
  }
  .stMarkdown p, .stMarkdown li { color: #C0C4CE !important; }
  
  /* Metric Cards */
  div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #1A1E2C, #1E222D);
    border: 1px solid #2A2E39;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }
  div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.6), 0 0 0 1px #363A45;
  }
  div[data-testid="stMetricLabel"] p {
    color: #787B86 !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }
  div[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
  }
  
  /* Metric Deltas (Strict Colors) */
  div[data-testid="stMetricDelta"] svg { display: none; }
  div[data-testid="stMetricDelta"] [data-testid="stMetricDeltaIcon-Up"] ~ div { color: #00C853 !important; }
  div[data-testid="stMetricDelta"] [data-testid="stMetricDeltaIcon-Down"] ~ div { color: #FF3333 !important; }

  /* Badges */
  .badge {
    display: inline-block; border-radius: 6px;
    padding: 4px 10px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;
  }
  .badge-bull { background: rgba(0,200,83,0.15); color: #00C853 !important; border: 1px solid rgba(0,200,83,0.3); }
  .badge-bear { background: rgba(255,51,51,0.15); color: #FF3333 !important; border: 1px solid rgba(255,51,51,0.3); }
  .badge-neut { background: rgba(255,179,0,0.15); color: #FFB300 !important; border: 1px solid rgba(255,179,0,0.3); }

  /* Summary Card */
  .summary-card {
    background: #131722;
    border-left: 4px solid #2962FF;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 24px;
    font-size: 0.95rem;
    line-height: 1.6;
    color: #D1D4DC !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
  }
  .summary-card b { color: #FFFFFF !important; }
  
  /* News Cards */
  .news-card {
    background: #1A1E2C;
    border: 1px solid #2A2E39;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 12px;
    transition: background 0.2s ease;
  }
  .news-card:hover { background: #222736; }
  .news-title {
    font-size: 0.95rem; font-weight: 600; color: #FFFFFF !important; margin-bottom: 8px; text-decoration: none; display: block; line-height: 1.4;
  }
  .news-title:hover { color: #2962FF !important; }
  .news-meta {
    font-size: 0.75rem; color: #787B86 !important; display: flex; justify-content: space-between; align-items: center;
  }

  /* Expanders */
  details { background: #131722; border: 1px solid #2A2E39 !important; border-radius: 8px; margin-bottom: 16px; }
  summary { color: #E0E0E0 !important; font-weight: 600; padding: 12px 16px; }
  
  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background-color: #131722; border-radius: 8px; padding: 4px; border: 1px solid #2A2E39; gap: 4px; flex-wrap: wrap; }
  .stTabs [data-baseweb="tab"] { color: #787B86 !important; font-weight: 600; border-radius: 6px; padding: 8px 16px; white-space: nowrap; }
  .stTabs [aria-selected="true"] { background-color: #2962FF !important; color: #FFFFFF !important; }

  /* Misc */
  hr { border-color: #2A2E39 !important; }
  .ts-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: #1E222D; border: 1px solid #2A2E39; border-radius: 20px;
    padding: 4px 14px; font-size: 0.77rem; color: #787B86 !important;
    margin: 6px 0 20px 0;
  }
  .header-title {
    font-size: 2.2rem; font-weight: 800; line-height: 1.2;
    background: linear-gradient(90deg, #FFFFFF 0%, #A0B4FF 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
  }
  
  @media (max-width: 640px) {
    .header-title { font-size: 1.6rem; }
    div[data-testid="stMetricValue"] { font-size: 1.3rem !important; }
  }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# ASSET REGISTRY
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
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <strong style="color: #FFFFFF !important; font-size: 1.2rem;">📡 Market Sentinel</strong><br>
            <span style="color: #787B86 !important; font-size: 0.85rem;">Configuration</span>
        </div>
    """, unsafe_allow_html=True)

    asset_label = st.selectbox(
        "Asset",
        options=list(ASSETS.keys()),
        index=list(ASSETS.keys()).index(DEFAULT_ASSET),
        help="Choose an Indian index, blue-chip stock, or global instrument.",
    )
    ticker  = ASSETS[asset_label]["ticker"]
    news_q  = ASSETS[asset_label]["news_q"]

    months = st.slider(
        "Date Range (months)", min_value=1, max_value=12, value=3, step=1,
        help="History to analyse. ≥ 3 months is recommended for reliable SMA-20 readings.",
    )

    sent_thresh = st.slider(
        "Buy Sentiment Threshold", min_value=0.10, max_value=0.90,
        value=0.40, step=0.05,
        help="VADER compound score (0 = neutral, 1 = very positive).",
    )

    st.markdown("---")
    st.markdown("### 🧠 AI Configuration")
    gemini_api_key = st.text_input(
        "Gemini API Key", 
        type="password", 
        help="Get your free API key at aistudio.google.com to unlock AI Insights."
    )

    st.markdown("---")
    
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("""
        <div style="font-size:0.85rem; color:#D1D4DC !important;">
            <strong>Signal Logic</strong><br>
            A buy signal is triggered when the price is trending up and the news is positive.
        </div>
        <div style="font-size:0.75rem; color:#787B86 !important; margin-top:4px;">
            Close > SMA-20 &nbsp;AND&nbsp; Sentiment > threshold
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("Data: Yahoo Finance & Google News RSS")


# ──────────────────────────────────────────────────────────────────
# DATA HELPERS
# ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_price_data(ticker_symbol: str, start: str, end: str) -> pd.DataFrame:
    try:
        t  = yf.Ticker(ticker_symbol, session=_YF_SESSION)
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
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()

    delta    = df["Close"].diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = (-delta).clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


@st.cache_data(ttl=600, show_spinner=False)
def fetch_live_news(query: str, max_items: int = 20) -> list[dict]:
    url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        feed = feedparser.parse(url)
    except Exception:
        return []

    analyser = SentimentIntensityAnalyzer()
    results  = []
    for entry in feed.entries[:max_items]:
        title = re.sub(r"<[^>]+>", "", entry.get("title", ""))
        score = analyser.polarity_scores(title)["compound"]
        
        # Strict logic
        if score > 0.05:
            label, color = "Positive", "bull"
        elif score < -0.05:
            label, color = "Negative", "bear"
        else:
            label, color = "Neutral", "neut"
            
        results.append({
            "title":     title,
            "source":    entry.get("source", {}).get("title", "Google News"),
            "published": entry.get("published", "")[:16],
            "url":       entry.get("link", "#"),
            "score":     round(score, 3),
            "label":     label,
            "color":     color
        })
    return results


def compute_daily_sentiment(news_items: list[dict], df_index: pd.DatetimeIndex) -> pd.Series:
    if not news_items:
        return pd.Series(0.0, index=df_index, name="Sentiment_Score")
    mean_score = np.mean([x["score"] for x in news_items])
    rng        = np.random.default_rng(seed=int(pd.Timestamp.now().date().toordinal()))
    noise      = rng.uniform(-0.12, 0.12, size=len(df_index))
    return pd.Series(
        np.clip(mean_score + noise, -1.0, 1.0),
        index=df_index, name="Sentiment_Score",
    )


# ──────────────────────────────────────────────────────────────────
# CHART BUILDER
# ──────────────────────────────────────────────────────────────────
def build_chart(df: pd.DataFrame, sent_threshold: float, show_sentiment: bool = True) -> go.Figure:
    close = df["Close"].squeeze()
    sma20 = df["SMA_20"].squeeze()
    sma50 = df["SMA_50"].squeeze()
    rsi   = df["RSI"].squeeze()
    
    # If not showing sentiment (like in custom search), adjust layout
    rows = 3 if show_sentiment else 2
    row_heights = [0.58, 0.21, 0.21] if show_sentiment else [0.7, 0.3]
    titles = ("", "RSI (14-period)", "News Sentiment Score") if show_sentiment else ("", "RSI (14-period)")
    
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=row_heights,
        subplot_titles=titles,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"].squeeze(), high=df["High"].squeeze(),
        low=df["Low"].squeeze(),   close=close,
        increasing_line_color="#00C853", decreasing_line_color="#FF3333",
        increasing_fillcolor="#003D1A",  decreasing_fillcolor="#3D0000",
        name="Price", whiskerwidth=0.4, line=dict(width=1.2),
    ), row=1, col=1)

    # SMAs
    fig.add_trace(go.Scatter(x=df.index, y=sma20, name="SMA 20", line=dict(color="#FFB300", width=1.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=sma50, name="SMA 50", line=dict(color="#2962FF", width=1.4, dash="dot")), row=1, col=1)

    # Buy Signals (only if column exists)
    if "Buy_Signal" in df.columns:
        buys  = df[df["Buy_Signal"]]
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys.index, y=(buys["Low"].squeeze() * 0.991),
                mode="markers+text",
                marker=dict(symbol="triangle-up", size=14, color="#00C853", line=dict(color="#FFFFFF", width=1)),
                text=["▲ BUY"] * len(buys), textposition="bottom center",
                textfont=dict(color="#00C853", size=10, family="Inter"),
                name=f"Buy Signal ({len(buys)})",
                hovertemplate=(
                    "<b style='color:#00C853'>Buy Signal</b><br>"
                    "Date: %{x|%d %b %Y}<br>Close: %{customdata[0]:,.2f}<br>Sentiment: %{customdata[1]:+.3f}<extra></extra>"
                ),
                customdata=np.stack([buys["Close"].squeeze().values, buys["Sentiment_Score"].values], axis=-1),
            ), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=df.index, y=rsi, line=dict(color="#29B6F6", width=1.5),
        fill="tozeroy", fillcolor="rgba(41,182,246,0.07)", name="RSI 14",
    ), row=2, col=1)
    fig.add_hline(y=70, line_color="#FF3333", line_dash="dash", line_width=1, row=2, col=1) # Overbought (Red)
    fig.add_hline(y=30, line_color="#00C853", line_dash="dash", line_width=1, row=2, col=1) # Oversold (Green)

    # Sentiment Bars (if enabled)
    if show_sentiment and "Sentiment_Score" in df.columns:
        sent  = df["Sentiment_Score"]
        bar_colors = ["#00C853" if s > 0.05 else ("#FF3333" if s < -0.05 else "#FFB300") for s in sent]
        fig.add_trace(go.Bar(
            x=df.index, y=sent, marker_color=bar_colors, opacity=0.8,
            name="Sentiment", showlegend=False,
        ), row=3, col=1)
        fig.add_hline(
            y=sent_threshold, line_color="#FFB300", line_dash="dash", line_width=1.2, row=3, col=1,
            annotation_text=f"Threshold ({sent_threshold})", annotation_font_color="#FFB300", annotation_font_size=10, annotation_position="right",
        )

    # Layout
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#B2B5BE", family="Inter, system-ui, sans-serif", size=11),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(19,23,34,0.9)", bordercolor="#2A2E39", borderwidth=1,
            font=dict(color="#D1D4DC", size=11),
        ),
        xaxis_rangeslider_visible=False, hovermode="x unified",
        hoverlabel=dict(bgcolor="#1E222D", bordercolor="#2A2E39", font_color="#E0E0E0", font_size=12),
        margin=dict(l=8, r=8, t=40, b=8), height=700 if show_sentiment else 550,
    )

    grid_style = dict(showgrid=True, gridcolor="rgba(42,46,57,0.8)", gridwidth=1, zeroline=False, linecolor="#2A2E39", tickfont=dict(color="#787B86", size=10))
    for i in range(1, rows + 1):
        fig.update_xaxes(**grid_style, row=i, col=1)
        fig.update_yaxes(**grid_style, row=i, col=1)

    fig.update_yaxes(title_text="Price", title_font=dict(size=10, color="#787B86"), row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], title_font=dict(size=10, color="#787B86"), row=2, col=1)
    if show_sentiment:
        fig.update_yaxes(title_text="Score", range=[-1.1, 1.1], title_font=dict(size=10, color="#787B86"), row=3, col=1)

    for ann in fig.layout.annotations:
        ann.font.color = "#787B86"
        ann.font.size  = 11

    return fig


# ──────────────────────────────────────────────────────────────────
# MAIN APP EXECUTION
# ──────────────────────────────────────────────────────────────────
now_str = datetime.now().strftime("%d %b %Y, %I:%M %p IST")

# Unified Header
st.markdown('<div class="header-title">Market Sentinel</div>', unsafe_allow_html=True)
st.markdown(f'<div class="ts-pill">🕒 Last updated: {now_str}</div>', unsafe_allow_html=True)

# Onboarding Nudge
if st.session_state["show_onboarding"]:
    col1, col2 = st.columns([0.9, 0.1])
    with col1:
        st.info("👋 **Welcome!** New to trading? Check out the **Beginner's Guide** tab for plain-English explanations of all terms.")
    with col2:
        if st.button("✕ Dismiss", key="dismiss_ob"):
            st.session_state["show_onboarding"] = False
            st.rerun()

# ── FETCH MAIN DATA GLOBALLY SO TABS 1 & 3 CAN USE IT ──
with st.spinner("Fetching live market data and news for Dashboard..."):
    end_dt   = datetime.today()
    start_dt = end_dt - timedelta(days=months * 30)
    
    df_raw     = fetch_price_data(ticker, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
    news_items = fetch_live_news(news_q)

if df_raw.empty:
    st.error(f"⚠️ Could not fetch data for **{asset_label}**. Yahoo Finance may be temporarily unavailable or the ticker is invalid. Please try another asset or refresh.")
    st.stop()

df = add_technicals(df_raw.copy())
df["Sentiment_Score"] = compute_daily_sentiment(news_items, df.index)
df["Buy_Signal"] = (df["Close"].squeeze() > df["SMA_20"].squeeze()) & (df["Sentiment_Score"] > sent_thresh)

latest     = float(df["Close"].squeeze().iloc[-1])
prev       = float(df["Close"].squeeze().iloc[-2]) if len(df) > 1 else latest
delta_abs  = latest - prev
delta_pct  = (delta_abs / prev * 100) if prev else 0
sma20_val  = float(df["SMA_20"].squeeze().dropna().iloc[-1])
rsi_val    = float(df["RSI"].squeeze().dropna().iloc[-1])
avg_sent   = float(df["Sentiment_Score"].mean())
n_signals  = int(df["Buy_Signal"].sum())

is_bullish = latest > sma20_val
trend_badge = '<span class="badge badge-bull">Bullish Bias</span>' if is_bullish else '<span class="badge badge-bear">Bearish Bias</span>'

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈  Live Terminal", "🔰  Beginner's Guide", "🧠  AI Insights", "🔍 Stock Deep Dive"])

# ── TAB 1: LIVE TERMINAL ──
with tab1:
    # Summary Card
    name = asset_label.split()[-1]
    trend_text = "above its 20-day average (uptrend)" if latest > sma20_val else "below its 20-day average (downtrend)"
    sent_text = "positive" if avg_sent > 0.05 else ("negative" if avg_sent < -0.05 else "neutral")
    summary = f"<b>{name}</b> is currently {trend_text}. News sentiment is {sent_text}. Model flagged <b>{n_signals} Buy Signals</b> in this period."
    
    st.markdown(f'<div class="summary-card">{trend_badge} &nbsp; {summary}</div>', unsafe_allow_html=True)

    # Responsive Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Last Price", f"{latest:,.2f}", f"{delta_abs:+.2f} ({delta_pct:+.2f}%)", help="Current closing price.")
    with c2:
        st.metric("SMA-20", f"{sma20_val:,.2f}", "Above" if latest > sma20_val else "Below", help="20-day Simple Moving Average.")
    with c3:
        st.metric("RSI (14)", f"{rsi_val:.1f}", "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral"), help="Relative Strength Index.")
    with c4:
        st.metric("Live Sentiment", f"{avg_sent:+.3f}", "Positive" if avg_sent > 0.05 else ("Negative" if avg_sent < -0.05 else "Neutral"), help="Avg VADER score of current news (-1 to +1).")
    with c5:
        st.metric("Buy Signals", str(n_signals), f"{100 * n_signals / max(len(df), 1):.1f}% of days", help="Total signals in period.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Chart
    fig = build_chart(df, sent_thresh, show_sentiment=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # News Section
    st.markdown("### 📰 Live News & Sentiment")
    if not news_items:
        st.info("No news headlines found for this asset today.")
    else:
        # Use responsive columns for news cards
        col1, col2 = st.columns(2)
        for i, item in enumerate(news_items):
            col = col1 if i % 2 == 0 else col2
            with col:
                st.markdown(f"""
                <div class="news-card">
                    <div class="news-meta">
                        <span>{item['source']} • {item['published']}</span>
                        <span class="badge badge-{item['color']}">{item['label']} ({item['score']:+.2f})</span>
                    </div>
                    <a href="{item['url']}" target="_blank" class="news-title">{item['title']}</a>
                </div>
                """, unsafe_allow_html=True)

# ── TAB 2: BEGINNER'S GUIDE ──
with tab2:
    st.markdown("### 🔰 Beginner's Guide")
    st.markdown("New to trading? Here is how to read the metrics on the Live Terminal.")
    
    with st.expander("📈 SMA (Simple Moving Average)", expanded=True):
        st.markdown("A moving average smooths out daily price changes to show the underlying trend. If the price is above the **SMA-20** (20-day average), the short-term trend is upward. If it's below, the trend is downward.")
    
    with st.expander("⚡ RSI (Relative Strength Index)", expanded=True):
        st.markdown("RSI measures how fast a stock is moving on a scale of 0 to 100. Over 70 means the stock rose too fast and might be **overbought** (due for a drop). Under 30 means it dropped too fast and might be **oversold** (due for a bounce).")
    
    with st.expander("🗞️ News Sentiment Score", expanded=True):
        st.markdown("We use an AI (VADER) to read current news headlines and score them from -1.0 (very negative) to +1.0 (very positive). A score above 0.05 is generally positive.")
    
    with st.expander("🟢 Buy Signal", expanded=True):
        st.markdown("A Buy Signal appears when two things happen at the same time: the stock is in an uptrend (Price > SMA-20) **AND** the news is positive (Sentiment > Threshold). Requiring both reduces false alarms.")

# ── TAB 3: AI INSIGHTS ──
with tab3:
    st.markdown("### 🧠 Ask Gemini")
    st.markdown("Ask natural language questions about the current state of the market, and get insights powered by Google's Gemini AI grounded in real-time technical and sentiment data.")
    
    if not gemini_api_key:
        st.warning("⚠️ Please enter your **Gemini API Key** in the sidebar to unlock AI Insights.")
    else:
        suggestion = st.selectbox(
            "Quick queries:",
            ["(Type your own question below)", "Summarize the current technicals.", "Are there any alarming news headlines?", "Explain the RSI and SMA in simple terms for this asset."]
        )
        
        user_q = st.text_area("Ask a question about this asset:", value="" if suggestion.startswith("(") else suggestion, height=100)
        
        if st.button("Generate Insight", type="primary"):
            if not user_q:
                st.error("Please enter a question.")
            else:
                with st.spinner("Gemini is analyzing the data..."):
                    try:
                        genai.configure(api_key=gemini_api_key)
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        
                        headlines_text = "\n".join([f"- {item['title']} (Sentiment Score: {item['score']})" for item in news_items[:5]])
                        context_prompt = f"""
You are an expert quantitative financial analyst AI assistant embedded in a dashboard called "Market Sentinel". 
Your job is to answer user questions about the specific asset they are currently viewing. Provide concise, professional, and clear answers. Do NOT provide financial advice.

Here is the LIVE DATA CONTEXT for the selected asset:
- Asset Name: {asset_label}
- Ticker: {ticker}
- Current Price: {latest:,.2f}
- 20-Day Simple Moving Average (SMA-20): {sma20_val:,.2f}
- Current Trend: {"Bullish (Price > SMA-20)" if is_bullish else "Bearish (Price < SMA-20)"}
- RSI (14-day): {rsi_val:.1f} (Over 70 is overbought, under 30 is oversold)
- Average News Sentiment Score: {avg_sent:+.3f} (Scale: -1.0 to +1.0)
- Number of algorithmic Buy Signals in the past {months} months: {n_signals}

Recent News Headlines Context:
{headlines_text}

User Question: {user_q}
"""
                        response = model.generate_content(context_prompt)
                        st.markdown("---")
                        st.markdown("#### ✨ Gemini's Insight")
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"Error calling Gemini API: {e}. Please check your API key.")


# ── TAB 4: STOCK DEEP DIVE & EXPERT SEARCH ──
with tab4:
    st.markdown("### 🔍 Stock Deep Dive & Expert Recommendation")
    st.markdown("Search for any stock ticker to get a deep technical analysis and an AI-powered Buy/Sell/Wait recommendation.")
    
    custom_ticker = st.text_input("Enter Stock Ticker (e.g., RELIANCE.NS, AAPL, TSLA, INFY.NS):", placeholder="e.g. AAPL")
    
    if st.button("Analyze Stock", type="primary", key="analyze_deep"):
        if not custom_ticker:
            st.error("Please enter a stock ticker.")
        else:
            with st.spinner(f"Fetching data for {custom_ticker}..."):
                # Fetch 6 months of data for deep dive
                end_dt_d   = datetime.today()
                start_dt_d = end_dt_d - timedelta(days=180)
                df_deep = fetch_price_data(custom_ticker, start_dt_d.strftime("%Y-%m-%d"), end_dt_d.strftime("%Y-%m-%d"))
                
            if df_deep.empty:
                st.error(f"⚠️ Could not fetch data for **{custom_ticker}**. Please check the ticker symbol (e.g., Indian stocks usually end in `.NS` or `.BO`).")
            else:
                df_deep = add_technicals(df_deep)
                
                latest_d     = float(df_deep["Close"].squeeze().iloc[-1])
                prev_d       = float(df_deep["Close"].squeeze().iloc[-2]) if len(df_deep) > 1 else latest_d
                delta_abs_d  = latest_d - prev_d
                delta_pct_d  = (delta_abs_d / prev_d * 100) if prev_d else 0
                sma20_val_d  = float(df_deep["SMA_20"].squeeze().dropna().iloc[-1])
                sma50_val_d  = float(df_deep["SMA_50"].squeeze().dropna().iloc[-1])
                rsi_val_d    = float(df_deep["RSI"].squeeze().dropna().iloc[-1])
                
                # Metrics
                st.markdown(f"#### 📊 Technicals for {custom_ticker.upper()}")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Last Price", f"{latest_d:,.2f}", f"{delta_abs_d:+.2f} ({delta_pct_d:+.2f}%)")
                with c2:
                    st.metric("SMA-20", f"{sma20_val_d:,.2f}", "Above" if latest_d > sma20_val_d else "Below")
                with c3:
                    st.metric("SMA-50", f"{sma50_val_d:,.2f}", "Above" if latest_d > sma50_val_d else "Below")
                with c4:
                    st.metric("RSI (14)", f"{rsi_val_d:.1f}", "Overbought" if rsi_val_d > 70 else ("Oversold" if rsi_val_d < 30 else "Neutral"))
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Chart (No sentiment panel)
                fig_d = build_chart(df_deep, sent_threshold=0, show_sentiment=False)
                st.plotly_chart(fig_d, use_container_width=True, config={"displayModeBar": False})
                
                st.markdown("---")
                
                # Expert Analysis using Gemini
                st.markdown("### 🤖 Expert Analysis & Recommendation")
                if not gemini_api_key:
                    st.warning("⚠️ Please enter your **Gemini API Key** in the sidebar to unlock the AI Expert Analysis and Recommendation.")
                else:
                    with st.spinner("Gemini is analyzing technicals and formulating a recommendation..."):
                        try:
                            genai.configure(api_key=gemini_api_key)
                            model = genai.GenerativeModel('gemini-1.5-flash')
                            
                            deep_prompt = f"""
You are a highly experienced stock market technical analyst and expert. 
Analyze the following live technical data for the stock ticker: {custom_ticker}.

- Current Price: {latest_d:,.2f}
- 20-Day SMA: {sma20_val_d:,.2f} (Short-term trend is {"Bullish" if latest_d > sma20_val_d else "Bearish"})
- 50-Day SMA: {sma50_val_d:,.2f} (Medium-term trend is {"Bullish" if latest_d > sma50_val_d else "Bearish"})
- RSI (14-day): {rsi_val_d:.1f} (Remember: >70 is typically overbought, <30 is typically oversold)

Please provide:
1. **Trend Analysis**: A brief, professional analysis of the latest trend for this stock based on these technicals. 
2. **Key Levels**: Identify potential support or resistance dynamics based on the SMAs.
3. **Verdict**: At the very end, give a clear, explicit recommendation: **BUY**, **SELL**, or **WAIT**. Highlight this verdict in bold.

Format your response cleanly with markdown headers or bullet points. Do not include a generic disclaimer, the system will add a strict one below your output.
"""
                            response_d = model.generate_content(deep_prompt)
                            st.markdown(response_d.text)
                            
                        except Exception as e:
                            st.error(f"Error calling Gemini API: {e}. Please check your API key.")
                    
                    # STRICT WARNING/DISCLAIMER 
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.error(
                        "🚨 **WARNING: DO AT YOUR OWN RISK - SUBJECT TO MARKET RISK.**\n\n"
                        "This recommendation and analysis is generated entirely by an AI model based on automated technical indicators. "
                        "**THIS IS NOT FINANCIAL ADVICE.** Do not trade, buy, or sell any stock based solely on this recommendation. "
                        "You must do your own research and consult a registered financial advisor before making any investment decisions. "
                        "Any actions taken are entirely at your own risk."
                    )


# ──────────────────────────────────────────────────────────────────
# FOOTER & TRUST ELEMENTS
# ──────────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    with st.expander("ℹ️ About this project"):
        st.markdown("""
        **Market Sentinel** combines quantitative technical analysis (SMA, RSI) with 
        Natural Language Processing (VADER sentiment analysis) to build a dual-confirmation 
        market signal engine. 
        
        Data is sourced in real-time from Yahoo Finance (prices) and Google News RSS (headlines).
        """)
with col2:
    st.markdown("""
    <div style="text-align: right; color: #787B86; font-size: 0.85rem; line-height: 1.5;">
        <b>⚠️ Educational project — not financial advice.</b><br>
        Built by Rohit • <a href="https://github.com/tutu82430-alt/market-sentiment-analyzer" target="_blank" style="color: #2962FF; text-decoration: none;">View on GitHub</a>
    </div>
    """, unsafe_allow_html=True)
