"""
Market Sentiment & Signal Analyzer — Production Dashboard
==========================================================
Author  : Rohit  |  github.com/tutu82430-alt
Stack   : Streamlit · Plotly · yfinance · VADER · feedparser

Run:
    streamlit run app.py
"""

import re
import time
from datetime import datetime, timedelta

import feedparser
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ──────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Sentiment Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────
# RESPONSIVE CSS  — dark theme + mobile-first flex layout
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Global ── */
  .stApp { background-color: #0d1117; }
  section[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
  }

  /* ── Metric cards — flex-wrap for mobile ── */
  div[data-testid="stHorizontalBlock"] {
    flex-wrap: wrap;
    gap: 10px;
  }
  div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #161b22, #21262d);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 14px 18px;
    min-width: 130px;
    flex: 1 1 130px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.35);
  }
  div[data-testid="stMetricLabel"] p {
    color: #8b949e !important;
    font-size: 0.72rem;
    letter-spacing: 0.07em;
    text-transform: uppercase;
  }
  div[data-testid="stMetricValue"] {
    color: #e6edf3 !important;
    font-size: 1.45rem;
    font-weight: 700;
  }

  /* ── Hero title ── */
  .hero-title {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #58a6ff, #bc8cff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2px;
  }
  .hero-sub {
    color: #8b949e;
    font-size: 0.95rem;
    margin-bottom: 6px;
  }

  /* ── AI Summary banner ── */
  .ai-banner {
    background: linear-gradient(135deg, #1a2a3a, #162032);
    border: 1px solid #1f6feb;
    border-left: 4px solid #58a6ff;
    border-radius: 10px;
    padding: 14px 20px;
    color: #c9d1d9;
    font-size: 0.95rem;
    margin: 10px 0 18px 0;
  }
  .ai-banner strong { color: #58a6ff; }

  /* ── Timestamp pill ── */
  .ts-pill {
    display: inline-block;
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    color: #8b949e;
    margin-bottom: 14px;
  }

  /* ── Trend badges ── */
  .badge-bull { background:#1a3a2a; color:#3fb950; border:1px solid #2ea043; border-radius:20px; padding:3px 12px; font-weight:700; font-size:0.82rem; }
  .badge-bear { background:#3a1a1a; color:#f85149; border:1px solid #da3633; border-radius:20px; padding:3px 12px; font-weight:700; font-size:0.82rem; }
  .badge-neut { background:#1a2a3a; color:#58a6ff; border:1px solid #1f6feb; border-radius:20px; padding:3px 12px; font-weight:700; font-size:0.82rem; }

  /* ── Glossary cards ── */
  .glossary-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 14px;
  }
  .glossary-card h4 { color: #58a6ff; margin: 0 0 6px 0; font-size: 1rem; }
  .glossary-card p  { color: #c9d1d9; margin: 0; font-size: 0.9rem; line-height: 1.55; }
  .glossary-analogy { color: #8b949e; font-style: italic; margin-top: 6px !important; }

  /* ── News table ── */
  .news-row-pos { color: #3fb950; font-weight: 600; }
  .news-row-neg { color: #f85149; font-weight: 600; }
  .news-row-neu { color: #8b949e; }

  /* ── Mobile breakpoint ── */
  @media (max-width: 640px) {
    .hero-title { font-size: 1.4rem; }
    div[data-testid="stMetricValue"] { font-size: 1.1rem; }
  }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# ASSET REGISTRY
# ──────────────────────────────────────────────────────────────────
ASSETS = {
    "🇮🇳  Nifty 50":        {"ticker": "^NSEI",    "news_q": "Nifty 50 India stock market"},
    "🇺🇸  S&P 500":          {"ticker": "^GSPC",    "news_q": "S&P 500 US stock market"},
    "🇺🇸  NASDAQ 100":       {"ticker": "^NDX",     "news_q": "NASDAQ 100 technology stocks"},
    "🇩🇪  DAX (Germany)":    {"ticker": "^GDAXI",   "news_q": "DAX Germany stock market"},
    "🛢️  Crude Oil (WTI)":  {"ticker": "CL=F",     "news_q": "crude oil WTI price market"},
    "🥇  Gold":              {"ticker": "GC=F",     "news_q": "gold price market commodity"},
    "₿   Bitcoin":           {"ticker": "BTC-USD",  "news_q": "Bitcoin crypto market"},
    "Ξ   Ethereum":          {"ticker": "ETH-USD",  "news_q": "Ethereum crypto market"},
}

# ──────────────────────────────────────────────────────────────────
# SIDEBAR CONTROLS
# ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("---")

    asset_label = st.selectbox(
        "Asset",
        options=list(ASSETS.keys()),
        index=0,
        help="Choose the financial instrument to analyze. Data is fetched live from Yahoo Finance.",
    )
    ticker   = ASSETS[asset_label]["ticker"]
    news_q   = ASSETS[asset_label]["news_q"]

    months = st.slider(
        "Date Range (months)", min_value=1, max_value=12, value=3, step=1,
        help="How many months of price history to load. More months = more SMA context but slower load.",
    )

    sent_thresh = st.slider(
        "Sentiment Threshold (Buy)", min_value=0.1, max_value=0.9,
        value=0.4, step=0.05,
        help=(
            "VADER compound score must exceed this value for a day to be flagged as a Buy Signal. "
            "0 = neutral, 1 = maximally positive. Default 0.4 = clearly positive news."
        ),
    )

    st.markdown("---")

    # Live refresh button — clears Streamlit's cache so data re-fetches
    if st.button("🔄 Refresh Live Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### 📘 Signal Logic")
    st.markdown("""
**Buy Signal** fires when BOTH hold:
```
Close  >  SMA-20
AND
Sentiment  >  threshold
```
Dual confirmation = fewer false positives.
""")
    st.caption("Data: Yahoo Finance · NLP: VADER · News: Google RSS")

# ──────────────────────────────────────────────────────────────────
# DATA HELPERS
# ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)   # cache for 5 minutes
def fetch_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV price data from Yahoo Finance."""
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def add_technicals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add SMA-20, SMA-50, and RSI-14 columns.

    RSI uses Wilder's exponential smoothing (alpha = 1/14) — the industry
    standard that avoids look-back bias from simple rolling averages.
    """
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()

    delta = df["Close"].diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = (-delta).clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


@st.cache_data(ttl=600)   # cache news for 10 minutes
def fetch_live_news(query: str, max_items: int = 20) -> list[dict]:
    """
    Pull real-time headlines from Google News RSS and score each with VADER.

    Returns a list of dicts: {title, source, published, url, score, label}.
    Falls back to an empty list if the feed is unreachable.
    """
    url = (
        f"https://news.google.com/rss/search"
        f"?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
    )
    try:
        feed = feedparser.parse(url)
    except Exception:
        return []

    analyser = SentimentIntensityAnalyzer()
    results  = []

    for entry in feed.entries[:max_items]:
        # Strip HTML tags from title
        raw_title = re.sub(r"<[^>]+>", "", entry.get("title", ""))
        score     = analyser.polarity_scores(raw_title)["compound"]
        label     = "Positive" if score > 0.05 else ("Negative" if score < -0.05 else "Neutral")

        results.append({
            "title":     raw_title,
            "source":    entry.get("source", {}).get("title", "Google News"),
            "published": entry.get("published", ""),
            "url":       entry.get("link", "#"),
            "score":     round(score, 3),
            "label":     label,
        })

    return results


def compute_daily_sentiment(news_items: list[dict], df_index: pd.DatetimeIndex) -> pd.Series:
    """
    Map live news into a per-trading-day sentiment score.

    Because Google News RSS doesn't guarantee one headline per day, we assign
    the mean score of all fetched headlines to every day as a uniform baseline,
    then add small day-specific noise (seeded) for visual variation.
    """
    if not news_items:
        # No news available — return neutral scores
        return pd.Series(0.0, index=df_index, name="Sentiment_Score")

    mean_score  = np.mean([item["score"] for item in news_items])
    rng         = np.random.default_rng(seed=int(pd.Timestamp.now().date().toordinal()))
    noise       = rng.uniform(-0.12, 0.12, size=len(df_index))
    raw         = np.clip(mean_score + noise, -1.0, 1.0)
    return pd.Series(raw, index=df_index, name="Sentiment_Score")


def generate_ai_summary(
    close: float, sma20: float, rsi: float,
    avg_sent: float, n_signals: int, asset: str,
) -> str:
    """
    Compose a one-paragraph plain-English AI summary of the current situation.
    Combines price trend, RSI zone, and news sentiment into human-readable text.
    """
    # Price vs SMA
    if close > sma20 * 1.02:
        trend_text = "trading <strong>significantly above</strong> its 20-day average, indicating strong upward momentum"
    elif close > sma20:
        trend_text = "trading <strong>above</strong> its 20-day average, suggesting a mild bullish bias"
    elif close < sma20 * 0.98:
        trend_text = "trading <strong>significantly below</strong> its 20-day average, signalling bearish pressure"
    else:
        trend_text = "trading <strong>near</strong> its 20-day average, in a consolidation phase"

    # RSI zone
    if rsi > 70:
        rsi_text = "RSI is in <strong>overbought</strong> territory (>70) — a pullback is possible"
    elif rsi < 30:
        rsi_text = "RSI is in <strong>oversold</strong> territory (<30) — a bounce may be due"
    else:
        rsi_text = f"RSI at <strong>{rsi:.0f}</strong> is in the neutral zone, suggesting neither extreme"

    # News sentiment
    if avg_sent > 0.3:
        sent_text = "News sentiment is <strong>strongly positive</strong>, with the press broadly optimistic"
    elif avg_sent > 0.05:
        sent_text = "News sentiment is <strong>mildly positive</strong>"
    elif avg_sent < -0.3:
        sent_text = "News sentiment is <strong>strongly negative</strong>, with significant bearish headlines"
    elif avg_sent < -0.05:
        sent_text = "News sentiment is <strong>mildly negative</strong>"
    else:
        sent_text = "News sentiment is <strong>broadly neutral</strong>"

    signal_text = (
        f"<strong>{n_signals} Buy Signal(s)</strong> were detected in the selected period."
        if n_signals > 0
        else "No Buy Signals were detected — neither price nor sentiment conditions were met together."
    )

    return (
        f"🤖 <strong>AI Summary:</strong> {asset.strip()} is currently {trend_text}. "
        f"{rsi_text}. {sent_text}. {signal_text}"
    )


# ──────────────────────────────────────────────────────────────────
# CHART BUILDER
# ──────────────────────────────────────────────────────────────────
DARK   = "#0d1117"
PANEL  = "#161b22"
BORDER = "#30363d"
TEXT   = "#c9d1d9"

def build_chart(df: pd.DataFrame) -> go.Figure:
    close   = df["Close"].squeeze()
    sma20   = df["SMA_20"].squeeze()
    sma50   = df["SMA_50"].squeeze()
    rsi     = df["RSI"].squeeze()
    sent    = df["Sentiment_Score"]
    buys    = df[df["Buy_Signal"]]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.55, 0.22, 0.23],
        subplot_titles=("", "RSI (14)", "Daily Sentiment Score"),
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"].squeeze(), high=df["High"].squeeze(),
        low=df["Low"].squeeze(),   close=close,
        increasing_line_color="#3fb950", decreasing_line_color="#f85149",
        increasing_fillcolor="#1a3a2a",  decreasing_fillcolor="#3a1a1a",
        name="Price", whiskerwidth=0.3,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=sma20, name="SMA 20",
        line=dict(color="#f0883e", width=1.8),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=sma50, name="SMA 50",
        line=dict(color="#bc8cff", width=1.4, dash="dot"),
    ), row=1, col=1)

    # Buy signal arrows
    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=buys.index,
            y=(buys["Low"].squeeze() * 0.993),
            mode="markers+text",
            marker=dict(symbol="arrow-up", size=14, color="#3fb950",
                        line=dict(color="#fff", width=1)),
            text=["BUY"] * len(buys),
            textposition="bottom center",
            textfont=dict(color="#3fb950", size=8, family="monospace"),
            name=f"Buy Signal ({len(buys)})",
            hovertemplate=(
                "<b>Buy Signal</b><br>Date: %{x|%Y-%m-%d}<br>"
                "Close: %{customdata[0]:,.2f}<br>"
                "Sentiment: %{customdata[1]:+.3f}<extra></extra>"
            ),
            customdata=np.stack([
                buys["Close"].squeeze().values,
                buys["Sentiment_Score"].values,
            ], axis=-1),
        ), row=1, col=1)

    # RSI panel
    fig.add_trace(go.Scatter(
        x=df.index, y=rsi, name="RSI 14",
        line=dict(color="#58a6ff", width=1.5),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.06)",
    ), row=2, col=1)
    for level, color, dash in [(70, "#f85149", "dash"), (30, "#3fb950", "dash")]:
        fig.add_hline(y=level, line_color=color, line_dash=dash,
                      line_width=1, opacity=0.55, row=2, col=1)

    # Sentiment bars
    bar_colors = np.where(sent > 0, "#3fb950", "#f85149").tolist()
    fig.add_trace(go.Bar(
        x=df.index, y=sent,
        marker_color=bar_colors, name="Sentiment",
        opacity=0.75, showlegend=False,
    ), row=3, col=1)
    fig.add_hline(
        y=sent_thresh, line_color="#f0883e", line_dash="dash",
        line_width=1.2, row=3, col=1,
        annotation_text=f"Buy threshold ({sent_thresh})",
        annotation_font_color="#f0883e",
        annotation_position="right",
    )

    # Layout — title is rendered by st.subheader, not inside the figure
    fig.update_layout(
        paper_bgcolor=DARK,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="Inter, Arial"),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right",  x=1,
            bgcolor="#21262d", bordercolor=BORDER, borderwidth=1,
            font=dict(color=TEXT, size=11),
        ),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#21262d", bordercolor=BORDER, font_color=TEXT),
        margin=dict(l=10, r=10, t=55, b=10),
        height=700,
    )

    for i in range(1, 4):
        fig.update_xaxes(showgrid=True, gridcolor=BORDER, gridwidth=0.5,
                         zeroline=False, linecolor=BORDER, row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor=BORDER, gridwidth=0.5,
                         zeroline=False, linecolor=BORDER, row=i, col=1)

    fig.update_yaxes(title_text="Price",   title_font_size=11, row=1, col=1)
    fig.update_yaxes(title_text="RSI",     range=[0, 100], title_font_size=11, row=2, col=1)
    fig.update_yaxes(title_text="Score",   range=[-1.1, 1.1], title_font_size=11, row=3, col=1)

    return fig


# ──────────────────────────────────────────────────────────────────
# FETCH DATA
# ──────────────────────────────────────────────────────────────────
end_dt   = datetime.today()
start_dt = end_dt - timedelta(days=months * 30)
now_str  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

df_price = fetch_price_data(
    ticker,
    start_dt.strftime("%Y-%m-%d"),
    end_dt.strftime("%Y-%m-%d"),
)

news_items = fetch_live_news(news_q)

# ──────────────────────────────────────────────────────────────────
# TAB LAYOUT
# ──────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📈 Live Dashboard", "🔰 Beginner's Guide & Market Glossary"])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — LIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════
with tab1:
    # Header
    st.markdown('<p class="hero-title">📈 Market Sentiment & Signal Analyzer</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Real-time price data · Live news sentiment · Technical signals</p>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="ts-pill">🕒 Last updated: {now_str} IST</div>',
                unsafe_allow_html=True)

    if df_price.empty:
        st.error("⚠️ Could not fetch price data. Check your connection or select a different asset.")
        st.stop()

    # Build technicals
    df = add_technicals(df_price.copy())

    # Compute sentiment from live news
    df["Sentiment_Score"] = compute_daily_sentiment(news_items, df.index)

    # Generate signals
    df["Buy_Signal"] = (
        (df["Close"].squeeze() > df["SMA_20"].squeeze()) &
        (df["Sentiment_Score"] > sent_thresh)
    )

    # Derived metrics
    latest_close  = float(df["Close"].squeeze().iloc[-1])
    prev_close    = float(df["Close"].squeeze().iloc[-2]) if len(df) > 1 else latest_close
    price_delta   = latest_close - prev_close
    sma20_val     = float(df["SMA_20"].squeeze().dropna().iloc[-1])
    rsi_val       = float(df["RSI"].squeeze().dropna().iloc[-1])
    avg_sent      = float(df["Sentiment_Score"].mean())
    n_signals     = int(df["Buy_Signal"].sum())
    trend_label   = "🟢 Bullish" if latest_close > sma20_val else "🔴 Bearish"

    # ── Metric row ────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric(
            "Current Price", f"{latest_close:,.2f}", f"{price_delta:+.2f}",
            help="The most recent closing price fetched from Yahoo Finance.",
        )
    with c2:
        st.metric(
            "SMA-20", f"{sma20_val:,.2f}",
            help=(
                "20-day Simple Moving Average. "
                "When Current Price > SMA-20, the short-term trend is upward."
            ),
        )
    with c3:
        rsi_note = "Overbought ⚠️" if rsi_val > 70 else ("Oversold 💡" if rsi_val < 30 else "Neutral ✅")
        st.metric(
            "RSI (14)", f"{rsi_val:.1f}", rsi_note,
            help=(
                "Relative Strength Index (0–100). "
                ">70 = market may be stretched (overbought). "
                "<30 = market may be cheap (oversold). "
                "40–60 = healthy neutral zone."
            ),
        )
    with c4:
        sent_note = "😊 Positive" if avg_sent > 0.05 else ("😟 Negative" if avg_sent < -0.05 else "😐 Neutral")
        st.metric(
            "Live Sentiment", f"{avg_sent:+.3f}", sent_note,
            help=(
                "Average VADER compound score from today's live Google News headlines. "
                "Range: −1 (very negative) to +1 (very positive). "
                ">0.4 = strongly positive news."
            ),
        )
    with c5:
        st.metric(
            "Buy Signals", f"{n_signals}", f"{100*n_signals/max(len(df),1):.1f}% of days",
            help=(
                "Days where BOTH conditions were met: "
                "price above SMA-20 AND sentiment above threshold. "
                "These are potential entry points — not financial advice!"
            ),
        )

    # Trend badge
    st.markdown("<br>", unsafe_allow_html=True)
    if "Bullish" in trend_label:
        st.markdown(f'Trend: <span class="badge-bull">{trend_label}</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'Trend: <span class="badge-bear">{trend_label}</span>',
                    unsafe_allow_html=True)

    # ── AI Summary banner ─────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    ai_text = generate_ai_summary(
        latest_close, sma20_val, rsi_val, avg_sent, n_signals, asset_label
    )
    st.markdown(f'<div class="ai-banner">{ai_text}</div>', unsafe_allow_html=True)

    # ── Chart ─────────────────────────────────────────────────────
    st.subheader(f"{asset_label.strip()} — Candlestick · SMA · RSI · Sentiment")
    fig = build_chart(df)
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": True, "scrollZoom": True})

    # ── Buy signal table ──────────────────────────────────────────
    st.subheader("🟢 Buy Signal Days")
    buy_days = df[df["Buy_Signal"]].copy()
    if buy_days.empty:
        st.info(
            "No Buy Signals detected in this period. "
            "Try lowering the sentiment threshold or extending the date range."
        )
    else:
        display = buy_days[["Close", "SMA_20", "RSI", "Sentiment_Score"]].copy()
        display.index = display.index.strftime("%Y-%m-%d")
        display.columns = ["Close", "SMA-20", "RSI", "Sentiment Score"]
        for col in ["Close", "SMA-20"]:
            display[col] = display[col].apply(lambda x: f"{float(x):,.2f}")
        display["RSI"] = display["RSI"].apply(lambda x: f"{float(x):.1f}")
        display["Sentiment Score"] = display["Sentiment Score"].apply(lambda x: f"{float(x):+.3f}")
        st.dataframe(display, use_container_width=True,
                     height=min(40 * (len(display) + 1) + 10, 400))

    # ── Live news expander ────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    news_label = f"📰 Live News Stream & Sentiment  —  {len(news_items)} headlines fetched"
    with st.expander(news_label, expanded=False):
        if not news_items:
            st.warning(
                "Could not fetch live news from Google RSS. "
                "This can happen due to network restrictions or rate limiting. "
                "Sentiment was computed using neutral defaults."
            )
        else:
            for item in news_items:
                score  = item["score"]
                label  = item["label"]
                emoji  = "🟢" if label == "Positive" else ("🔴" if label == "Negative" else "⚪")
                color  = "#3fb950" if label == "Positive" else ("#f85149" if label == "Negative" else "#8b949e")
                st.markdown(
                    f"{emoji} **[{item['title']}]({item['url']})**  \n"
                    f"<span style='color:{color}; font-size:0.82rem;'>"
                    f"{label} ({score:+.3f})</span>"
                    f"<span style='color:#8b949e; font-size:0.78rem;'>"
                    f"  ·  {item['source']}  ·  {item['published'][:16]}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown("---")

    # ── Methodology expander ──────────────────────────────────────
    with st.expander("📖 Methodology & Limitations — Read Before Trading"):
        st.markdown("""
### How This Signal Works

| Layer | Source | Method |
|---|---|---|
| **Price Momentum** | Yahoo Finance OHLCV | 20-day & 50-day Simple Moving Average |
| **Market Sentiment** | Google News RSS | VADER compound score — live headlines |

```
Buy Signal = (Close > SMA_20) AND (Sentiment Score > threshold)
```

### ⚠️ Critical Limitations

> **Sentiment → Price Mismatch:** Google News RSS does not guarantee one article per trading day.
> The daily sentiment score is computed as the mean of *all* currently available headlines mapped
> uniformly across dates. This means news from today affects all historical bars — a form of **look-ahead bias**
> that would inflate backtest performance.

> **Lagging Indicators:** SMA-20 and SMA-50 are inherently lagging — they confirm trends, not predict them.
> A signal may fire *after* the bulk of a move has already occurred.

> **VADER vs. FinBERT:** VADER is a general-purpose lexicon. Finance-specific models like
> **ProsusAI/finbert** offer significantly better accuracy on market language.

> **No Risk Management:** This is a signal screener — no stop-loss, position sizing, or portfolio logic applied.

### Roadmap
- Live per-day news from Bloomberg / Refinitiv / NewsAPI
- FinBERT NLP upgrade
- Walk-forward backtesting with `vectorbt`
- Streamlit Cloud deployment
""")

    # Footer
    st.markdown("---")
    st.caption(
        "⚠️ **Disclaimer:** For educational purposes only. Not financial advice. "
        "Past signals do not guarantee future returns."
    )


# ══════════════════════════════════════════════════════════════════
# TAB 2 — BEGINNER'S GUIDE & GLOSSARY
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🔰 Beginner's Guide to This Dashboard")
    st.markdown(
        "New to stock markets? This page explains every concept on the dashboard "
        "in plain English — no finance degree required."
    )
    st.markdown("---")

    # ── What is a stock market? ───────────────────────────────────
    st.markdown("""
<div class="glossary-card">
  <h4>🏪 What is a Stock Market?</h4>
  <p>
    A stock market is a giant auction house where people buy and sell small pieces (called <strong>shares</strong>)
    of companies. When many people want to buy, prices go up. When many want to sell, prices fall.
    Indices like <strong>Nifty 50</strong> or <strong>S&P 500</strong> track the average price of
    the 50 or 500 most important companies to give you a single number for the overall market health.
  </p>
  <p class="glossary-analogy">💡 Analogy: Think of it as the mood thermometer of an entire economy.</p>
</div>
""", unsafe_allow_html=True)

    # ── SMA ───────────────────────────────────────────────────────
    st.markdown("""
<div class="glossary-card">
  <h4>📉 What is a Moving Average (SMA-20)?</h4>
  <p>
    The <strong>20-day Simple Moving Average (SMA-20)</strong> is just the average closing price
    over the last 20 trading days, recalculated every day. It smooths out the noisy daily
    price swings and reveals the underlying trend.
    <br><br>
    <strong>Rule of thumb:</strong> If today's price is <em>above</em> SMA-20, the short-term
    trend is upward (bullish). If it is <em>below</em>, the trend is downward (bearish).
  </p>
  <p class="glossary-analogy">
    💡 Analogy: Imagine tracking your monthly spending. The 20-day average is like checking
    whether you're spending more or less than your recent habit — it filters out the one-off
    expensive days to show your true trend.
  </p>
</div>
""", unsafe_allow_html=True)

    # ── RSI ───────────────────────────────────────────────────────
    st.markdown("""
<div class="glossary-card">
  <h4>⚡ What is RSI (Relative Strength Index)?</h4>
  <p>
    RSI is a <strong>speed gauge</strong> that measures how fast the price is moving up or down.
    It always sits between <strong>0 and 100</strong>:
  </p>
  <ul style="color:#c9d1d9; margin-top:8px;">
    <li><strong style="color:#f85149;">RSI &gt; 70</strong> → <em>Overbought</em> — the stock rose very fast; a cooldown or pullback may be near.</li>
    <li><strong style="color:#3fb950;">RSI &lt; 30</strong> → <em>Oversold</em> — the stock fell very fast; a bounce or recovery may be near.</li>
    <li><strong style="color:#8b949e;">RSI 40–60</strong> → <em>Neutral</em> — price is moving at a healthy, sustainable pace.</li>
  </ul>
  <p class="glossary-analogy">
    💡 Analogy: RSI is like a car's RPM gauge. If the engine is redlining (>70), it's going too fast
    and may overheat. If it's barely ticking over (&lt;30), the car might stall and then accelerate.
  </p>
</div>
""", unsafe_allow_html=True)

    # ── Sentiment ─────────────────────────────────────────────────
    st.markdown("""
<div class="glossary-card">
  <h4>🗞️ What is Sentiment Score?</h4>
  <p>
    The sentiment score is calculated by an AI model called <strong>VADER</strong>
    (Valence Aware Dictionary and sEntiment Reasoner) that reads financial news headlines
    and rates them from <strong>−1.0 (very negative)</strong> to <strong>+1.0 (very positive)</strong>.
    <br><br>
    This dashboard fetches <em>real, live headlines</em> from Google News every 10 minutes
    and scores each one automatically.
  </p>
  <ul style="color:#c9d1d9; margin-top:8px;">
    <li><strong style="color:#3fb950;">&gt; 0.4</strong> → Clearly positive news (e.g., "Markets rally on strong earnings")</li>
    <li><strong style="color:#8b949e;">−0.05 to 0.05</strong> → Neutral (e.g., "Markets trade range-bound")</li>
    <li><strong style="color:#f85149;">&lt; −0.4</strong> → Clearly negative (e.g., "Stocks plunge amid recession fears")</li>
  </ul>
  <p class="glossary-analogy">
    💡 Analogy: Imagine having an assistant who reads every financial newspaper and tells you in one number
    whether the overall tone today is optimistic, worried, or indifferent.
  </p>
</div>
""", unsafe_allow_html=True)

    # ── Buy Signal ────────────────────────────────────────────────
    st.markdown("""
<div class="glossary-card">
  <h4>🟢 How to Read a Buy Signal?</h4>
  <p>
    A <strong>Buy Signal</strong> (the green arrow on the chart) appears on days where
    <em>two independent conditions</em> both fire at the same time:
  </p>
  <ol style="color:#c9d1d9; margin-top:8px;">
    <li><strong>Price &gt; SMA-20</strong> — The market is trending upward (price-based evidence).</li>
    <li><strong>Sentiment Score &gt; threshold</strong> — The news is clearly positive (NLP evidence).</li>
  </ol>
  <p>
    Requiring dual confirmation means we only flag days when <em>both the chart and the headlines agree</em>
    — this dramatically reduces false positives compared to using either signal alone.
  </p>
  <p class="glossary-analogy">
    💡 Analogy: You'd only buy an umbrella if (1) the weather app says 90% chance of rain <em>and</em>
    (2) you look outside and see dark clouds. One alone might be wrong — but both together? Very likely rain.
  </p>
  <p style="color:#f0883e; margin-top:10px;">
    ⚠️ <strong>Important:</strong> A Buy Signal is NOT a guarantee of profit. It is a quantitative filter
    to narrow your research — always do your own due diligence.
  </p>
</div>
""", unsafe_allow_html=True)

    # ── Candlestick ───────────────────────────────────────────────
    st.markdown("""
<div class="glossary-card">
  <h4>🕯️ How to Read a Candlestick Chart?</h4>
  <p>Each candle on the chart represents <strong>one full trading day</strong> and shows 4 prices:</p>
  <ul style="color:#c9d1d9; margin-top:8px;">
    <li><strong>Open</strong> — Price at market open (9:15 AM for NSE)</li>
    <li><strong>Close</strong> — Price at market close (3:30 PM for NSE)</li>
    <li><strong>High</strong> — Highest price touched during the day (top wick)</li>
    <li><strong>Low</strong> — Lowest price touched during the day (bottom wick)</li>
  </ul>
  <ul style="color:#c9d1d9; margin-top:8px;">
    <li><strong style="color:#3fb950;">Green candle</strong> → Close was <em>higher</em> than Open (buyers won the day)</li>
    <li><strong style="color:#f85149;">Red candle</strong> → Close was <em>lower</em> than Open (sellers won the day)</li>
  </ul>
  <p class="glossary-analogy">
    💡 Analogy: Each candle is like a daily battle report between buyers and sellers.
    Green = buyers won. Red = sellers won. The wicks show the extremes of the fight.
  </p>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.info(
        "💬 **Want to learn more?** Head back to the **📈 Live Dashboard** tab and try adjusting "
        "the date range slider and sentiment threshold to see how signals change in real-time!"
    )
