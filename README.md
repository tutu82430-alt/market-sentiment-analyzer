# 📈 Market Sentiment & Signal Analyzer

> **An interactive web dashboard combining NLP-driven news sentiment with classical technical analysis to surface high-conviction Buy Signals on global financial instruments.**

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18+-purple?style=flat-square&logo=plotly)](https://plotly.com/)
[![VADER](https://img.shields.io/badge/VADER-NLP%20Sentiment-orange?style=flat-square)](https://pypi.org/project/vaderSentiment/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## 🖥️ Live Dashboard Preview

```
┌─────────────────────────────────────────────────────────────────┐
│  ⚙️ Controls          📈 Market Sentiment & Signal Analyzer      │
│  ─────────────       ──────────────────────────────────────      │
│  Asset: Nifty 50     Current  SMA-20   RSI    Sentiment Signals │
│  Range: 3 months     24,636   24,229   54.1   +0.12    8 days  │
│  Threshold: 0.40                                                  │
│                      [Candlestick + SMA + Buy Arrow Chart]       │
│                      [RSI Panel]                                  │
│                      [Daily Sentiment Bar Chart]                  │
│                      [Buy Signal Days Table]                      │
│                      [Methodology & Limitations Expander]         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧩 Problem Statement

Traditional technical analysis (moving averages, RSI) ignores the qualitative information embedded in daily financial news. Pure sentiment analysis, without price confirmation, generates too many false positives.

**This dashboard bridges that gap** — it fuses two independent information streams:

| Signal Layer | Source | Method |
|---|---|---|
| **Price Momentum** | Yahoo Finance OHLCV | 20-day & 50-day Simple Moving Average |
| **Market Sentiment** | Financial headlines | VADER compound score (NLP) |

A **Buy Signal** fires **only when both conditions agree simultaneously**, reducing false positives from either source alone.

---

## 🛠️ Tech Stack

| Library | Version | Role |
|---|---|---|
| `streamlit` | ≥1.32 | Interactive web dashboard framework |
| `plotly` | ≥5.18 | Professional interactive financial charts |
| `yfinance` | ≥0.2.40 | Real-time & historical market data from Yahoo Finance |
| `pandas` | ≥2.0 | DataFrame manipulation, rolling calculations |
| `numpy` | ≥1.24 | Vectorised array operations |
| `vaderSentiment` | ≥3.3.2 | Rule-based NLP sentiment scoring (no API key needed) |
| `requests` | ≥2.31 | HTTP client (ready for live NewsAPI integration) |

---

## 📁 Project Structure

```
market-sentiment-analyzer/
├── app.py                   # ← Main Streamlit dashboard (run this)
├── src/
│   └── main.py              # CLI version (original script, still works)
├── output/                  # Auto-created charts (git-ignored)
├── data/                    # Auto-created CSVs (git-ignored)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/market-sentiment-analyzer.git
cd market-sentiment-analyzer
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the dashboard
```bash
streamlit run app.py
```

The app will open automatically at **http://localhost:8501** in your browser.

### (Optional) Run the CLI version
```bash
python src/main.py
```

---

## 📊 Signal Logic Deep-Dive

```python
Buy_Signal = (Close > SMA_20) AND (Sentiment_Score > threshold)
```

### Why both conditions?

| Condition | What it Checks | Why It Matters |
|---|---|---|
| `Close > SMA_20` | Price is above its 20-day trend | Buyers are in control; short-term momentum is up |
| `Sentiment > 0.4` | News is clearly positive | Market narrative supports the price move |

Only when price action and news sentiment align does the signal fire — requiring **dual confirmation** before flagging a potential entry.

### Indicators Explained

| Indicator | Period | Interpretation |
|---|---|---|
| **SMA-20** | 20 days | Short-term trend filter. Price above = bullish bias |
| **SMA-50** | 50 days | Medium-term trend context. Classic "golden cross" reference |
| **RSI** | 14 days (Wilder EWM) | >70: overbought, <30: oversold, 40–60: neutral |
| **VADER Sentiment** | Per-day | Compound score in [-1, 1]. >0.4 = clearly positive |

---

## 🖼️ Dashboard Features

- **🎛️ Sidebar Controls** — Select from 8 global assets (Nifty 50, S&P 500, NASDAQ, DAX, Gold, Crude Oil, Bitcoin, Ethereum), adjust date range (1–12 months), and tune the sentiment threshold
- **📊 Interactive Candlestick Chart** — Zoom, pan, hover for OHLCV data; SMA-20 and SMA-50 overlaid
- **🟢 Buy Signal Arrows** — Green upward arrows with `BUY` labels mark high-conviction entry days
- **📉 RSI Panel** — 14-period RSI with overbought/oversold zones
- **🗣️ Sentiment Bar Chart** — Daily VADER score with buy threshold line
- **🎯 Metric Cards** — Current Price, SMA-20, RSI, Avg Sentiment, Signal Count
- **📖 Methodology Expander** — Explains signal logic and flags all limitations (look-ahead bias, lagging indicators)

---

## ⚠️ Limitations & Future Improvements

### Current Limitations

| Limitation | Description |
|---|---|
| **Mock Headlines** | Headlines are randomly generated from a preset pool — NOT actual news for those dates |
| **Look-Ahead Bias** | In real backtesting, using post-market news to generate pre-market signals overstates performance |
| **Lagging SMA** | SMA is a lagging indicator — it confirms trends, not predicts them |
| **VADER Accuracy** | VADER is general-purpose; FinBERT (finance-tuned BERT) performs significantly better on financial text |
| **No Risk Management** | No stop-loss, position sizing, or portfolio construction logic is applied |

### Roadmap

- [ ] **Live NewsAPI** — Real financial news from [NewsAPI.org](https://newsapi.org/) or Bloomberg
- [ ] **FinBERT NLP** — Replace VADER with `ProsusAI/finbert` for domain-specific accuracy
- [ ] **Backtesting Engine** — `vectorbt` / `backtrader` integration with walk-forward validation
- [ ] **Risk Management** — Kelly criterion position sizing, ATR-based stop-loss
- [ ] **Multi-factor Scoring** — Combine RSI, MACD, volume, and sentiment into a composite score
- [ ] **Streamlit Cloud Deploy** — One-click deployment to [share.streamlit.io](https://share.streamlit.io)
- [ ] **Alerting** — Email/Telegram notifications when a Buy Signal fires

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 🙋 Author

**Rohit**  
*Quantitative Finance & Python Developer*

> ⭐ If this project helped you, please star the repository — it means a lot!

> ⚠️ **Disclaimer:** This tool is for educational and research purposes only. It does not constitute financial advice. Always conduct your own due diligence before making investment decisions.
