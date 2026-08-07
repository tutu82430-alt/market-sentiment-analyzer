"""
Market Sentiment & Signal Analyzer
====================================
Author  : Rohit
Purpose : Combines VADER NLP sentiment analysis with price-based technical
          indicators (SMA-20) to identify 'Bullish Signal' days on Nifty 50.

How to run:
    pip install -r requirements.txt
    python src/main.py
"""

import os
import random
import warnings
from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------
TICKER           = "^NSEI"          # Nifty 50 index
PERIOD_MONTHS    = 3                # How many months of history to fetch
SMA_WINDOW       = 20               # Simple Moving Average window (days)
SENTIMENT_THRESH = 0.3              # Minimum compound score to be "positive"
OUTPUT_DIR       = "output"
CHART_FILE       = os.path.join(OUTPUT_DIR, "market_chart.png")

# -----------------------------------------------------------------
# STEP 1 - FETCH MARKET DATA
# -----------------------------------------------------------------
def fetch_market_data(ticker: str, months: int = 3) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance for the last `months` months."""
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=months * 30)

    print(f"[1/5] Fetching market data for '{ticker}' "
          f"({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})...")

    df = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
    )

    if df.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}'. "
            "Check your internet connection and try again."
        )

    # yfinance sometimes returns MultiIndex columns -- flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    print(f"    OK Downloaded {len(df)} trading days of data.")
    return df


# -----------------------------------------------------------------
# STEP 2 - TECHNICAL INDICATORS
# -----------------------------------------------------------------
def add_technical_indicators(df: pd.DataFrame, sma_window: int = 20) -> pd.DataFrame:
    """Add a Simple Moving Average column to the DataFrame."""
    print(f"[2/5] Computing {sma_window}-day SMA...")
    df[f"SMA_{sma_window}"] = df["Close"].rolling(window=sma_window).mean()
    print(f"    OK SMA_{sma_window} column added.")
    return df


# -----------------------------------------------------------------
# STEP 3 - MOCK NEWS HEADLINES
# -----------------------------------------------------------------
BULLISH_HEADLINES = [
    "Inflation cools down, markets rally strongly",
    "Nifty 50 hits record high on robust corporate earnings",
    "FII inflows surge; Sensex, Nifty close in green",
    "RBI holds rates steady; analysts cheer market stability",
    "IT stocks soar after strong quarterly results",
    "GST collections hit all-time high; economic optimism rises",
    "Rupee strengthens against dollar; import-heavy sectors gain",
    "Budget 2025 boosts capex spending; infrastructure stocks rally",
    "Manufacturing PMI hits 5-year high, signals economic revival",
    "Global risk-on sentiment lifts emerging market equities",
    "Auto sector surges on record monthly sales data",
    "Pharma index rises on positive USFDA approvals",
    "Consumer confidence index rises to 14-month high",
    "Foreign portfolio investors net buyers for fifth consecutive session",
    "Banking sector leads rally as credit growth accelerates",
]

BEARISH_HEADLINES = [
    "Tech stocks dip amid rising regulatory fears",
    "Crude oil spike fans inflation worries; markets slide",
    "Nifty 50 drops 1.2% on weak global cues",
    "FII sell-off intensifies; mid-cap index tumbles",
    "Rupee hits 6-month low; import costs surge",
    "Core inflation rises unexpectedly, rate hike fears return",
    "IT sector under pressure as US recession fears mount",
    "SEBI tightens F&O regulations; derivatives volumes fall",
    "China economic slowdown rattles global equity markets",
    "Banking shares fall on rising NPA concerns",
    "Geopolitical tensions in Middle East drive oil higher",
    "Weak monsoon outlook dents rural consumption stocks",
    "US Fed signals higher-for-longer rates; Dalal Street bleeds",
    "Corporate margins squeezed as raw material costs rise",
    "Investor sentiment sours on mixed earnings season results",
]

NEUTRAL_HEADLINES = [
    "Markets trade range-bound ahead of key policy decision",
    "Nifty holds 22,000 support; traders await earnings season",
    "Mixed global signals keep Indian markets in consolidation",
    "Analysts divided on near-term market direction",
    "PSU stocks trade flat; private banks edge higher",
    "Volumes below average on account of mid-week holiday",
    "Indian markets largely in line with Asian peers",
    "Q4 GDP data awaited; stocks drift sideways",
    "Sector rotation seen as large-caps outperform mid-caps",
]

def generate_mock_headlines(dates: pd.DatetimeIndex) -> pd.Series:
    """
    Generate a realistic random financial headline for each trading date.
    Seeded for full reproducibility.
    """
    print("[3/5] Generating mock financial headlines...")
    random.seed(42)

    all_headlines = BULLISH_HEADLINES + BEARISH_HEADLINES + NEUTRAL_HEADLINES
    weights = (
        [0.50 / len(BULLISH_HEADLINES)] * len(BULLISH_HEADLINES)
        + [0.35 / len(BEARISH_HEADLINES)] * len(BEARISH_HEADLINES)
        + [0.15 / len(NEUTRAL_HEADLINES)] * len(NEUTRAL_HEADLINES)
    )

    headlines = [random.choices(all_headlines, weights=weights, k=1)[0] for _ in dates]
    series = pd.Series(headlines, index=dates, name="Headline")
    print(f"    OK Generated {len(series)} mock headlines.")
    return series


# -----------------------------------------------------------------
# STEP 4 - SENTIMENT SCORING (VADER)
# -----------------------------------------------------------------
def score_sentiment(headlines: pd.Series) -> pd.Series:
    """
    Apply VADER SentimentIntensityAnalyzer to every headline.
    Returns compound scores in range [-1.0, 1.0].
    """
    print("[4/5] Scoring sentiment with VADER...")
    analyser = SentimentIntensityAnalyzer()
    scores = headlines.apply(lambda h: analyser.polarity_scores(h)["compound"])
    scores.name = "Sentiment_Score"
    print(f"    OK Mean daily sentiment score: {scores.mean():.4f}")
    return scores


# -----------------------------------------------------------------
# STEP 5 - SIGNAL GENERATION
# -----------------------------------------------------------------
def generate_signals(
    df: pd.DataFrame,
    sma_col: str,
    sentiment_col: str,
    sentiment_thresh: float,
) -> pd.DataFrame:
    """
    Bullish Signal = True ONLY when:
        (1) Close > SMA_20   -- price above trend
        (2) Sentiment Score > threshold -- positive news sentiment
    """
    print("[5/5] Generating Bullish Signals...")
    df["Bullish_Signal"] = (
        (df["Close"] > df[sma_col]) & (df[sentiment_col] > sentiment_thresh)
    )
    n_signals = df["Bullish_Signal"].sum()
    pct = 100 * n_signals / len(df)
    print(f"    OK {n_signals} Bullish Signal days ({pct:.1f}% of all trading days).")
    return df


# -----------------------------------------------------------------
# STEP 6 - VISUALISATION
# -----------------------------------------------------------------
def plot_and_save(df: pd.DataFrame, sma_col: str, output_path: str) -> None:
    """Plot close + SMA-20 + Bullish Signal markers; save to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.patch.set_facecolor("#0d1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    close     = df["Close"].squeeze()
    sma       = df[sma_col].squeeze()
    sentiment = df["Sentiment_Score"].squeeze()
    signals   = df["Bullish_Signal"]
    dates     = df.index

    # Upper panel -- Price & SMA
    ax1.plot(dates, close, color="#58a6ff", linewidth=1.5, label="Nifty 50 Close", zorder=2)
    ax1.plot(dates, sma, color="#f0883e", linewidth=1.5, linestyle="--",
             label=f"{sma_col} (trend)", zorder=2)

    # Bullish Signal dots
    signal_dates  = dates[signals]
    signal_prices = close[signals]
    ax1.scatter(
        signal_dates, signal_prices,
        marker="o", color="#3fb950", s=70, zorder=5,
        label="Bullish Signal", edgecolors="#ffffff", linewidths=0.5,
    )

    # Shaded area
    ax1.fill_between(dates, close, sma,
                     where=(close > sma), alpha=0.08, color="#3fb950")
    ax1.fill_between(dates, close, sma,
                     where=(close <= sma), alpha=0.08, color="#f85149")

    ax1.set_ylabel("Price (INR)", color="#c9d1d9", fontsize=10)
    ax1.set_title(
        "Nifty 50 -- Market Sentiment & Signal Analyzer",
        color="#e6edf3", fontsize=14, fontweight="bold", pad=12,
    )
    ax1.legend(facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#c9d1d9", fontsize=9, loc="upper left")

    # Lower panel -- Sentiment
    colors = np.where(sentiment > 0, "#3fb950", "#f85149")
    ax2.bar(dates, sentiment, color=colors, width=1, alpha=0.85, zorder=2)
    ax2.axhline(SENTIMENT_THRESH, color="#f0883e", linewidth=1,
                linestyle="--", label=f"Threshold ({SENTIMENT_THRESH})")
    ax2.axhline(0, color="#c9d1d9", linewidth=0.5, alpha=0.4)
    ax2.set_ylabel("Sentiment\nScore", color="#c9d1d9", fontsize=9)
    ax2.set_ylim(-1.1, 1.1)
    ax2.legend(facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#c9d1d9", fontsize=8, loc="upper left")

    # X-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b '%y"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right",
             color="#c9d1d9", fontsize=8)

    # Annotation
    n_signals = int(signals.sum())
    ax1.annotate(
        f"  {n_signals} Bullish Signal{'s' if n_signals != 1 else ''} detected",
        xy=(0.99, 0.04), xycoords="axes fraction",
        ha="right", va="bottom",
        fontsize=9, color="#3fb950",
        bbox=dict(boxstyle="round,pad=0.3", fc="#21262d", ec="#30363d", alpha=0.8),
    )

    plt.tight_layout(h_pad=0.3)
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n    OK Chart saved -> {os.path.abspath(output_path)}")


# -----------------------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("  Market Sentiment & Signal Analyzer")
    print("  Nifty 50  |  VADER NLP  |  SMA-20")
    print("=" * 60)

    df = fetch_market_data(TICKER, months=PERIOD_MONTHS)

    sma_col = f"SMA_{SMA_WINDOW}"
    df = add_technical_indicators(df, sma_window=SMA_WINDOW)

    headlines = generate_mock_headlines(df.index)
    df["Headline"] = headlines

    df["Sentiment_Score"] = score_sentiment(df["Headline"])

    df = generate_signals(df, sma_col, "Sentiment_Score", SENTIMENT_THRESH)

    # Preview
    print("\n--- Sample Output (last 10 rows) ---")
    preview_cols = ["Close", sma_col, "Sentiment_Score", "Bullish_Signal"]
    with pd.option_context("display.float_format", "{:.2f}".format,
                           "display.max_colwidth", 60,
                           "display.width", 120):
        print(df[preview_cols].tail(10).to_string())

    # Save CSV
    os.makedirs("data", exist_ok=True)
    csv_path = os.path.join("data", "analysis_snapshot.csv")
    df.to_csv(csv_path)
    print(f"\n    OK Full data snapshot saved -> {os.path.abspath(csv_path)}")

    plot_and_save(df, sma_col, CHART_FILE)

    # Print bullish signal days
    bullish_days = df[df["Bullish_Signal"]].copy()
    print("\n--- Bullish Signal Days ---")
    if bullish_days.empty:
        print("  No Bullish Signal days found in this period.")
    else:
        for date, row in bullish_days.iterrows():
            print(
                f"  {date.strftime('%Y-%m-%d')} | "
                f"Close: {float(row['Close']):>9.2f} | "
                f"SMA: {float(row[sma_col]):>9.2f} | "
                f"Sentiment: {float(row['Sentiment_Score']):>+.3f} | "
                f"{row['Headline']}"
            )

    print("\n" + "=" * 60)
    print("  Run complete. Open output/market_chart.png to view the chart.")
    print("=" * 60)


if __name__ == "__main__":
    main()
