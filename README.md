# 📈 Market Sentiment & Signal Analyzer

> **Combining NLP-driven news sentiment with classical technical analysis to generate actionable Bullish Signals on Nifty 50.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![yfinance](https://img.shields.io/badge/yfinance-data-orange?style=flat-square)](https://pypi.org/project/yfinance/)
[![VADER](https://img.shields.io/badge/VADER-NLP%20Sentiment-purple?style=flat-square)](https://pypi.org/project/vaderSentiment/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## 🧩 Problem Statement

Traditional technical analysis (moving averages, RSI, MACD) ignores the qualitative information contained in daily financial news. At the same time, pure sentiment analysis without price confirmation can generate noisy, unreliable signals.

**This project bridges the gap** — it fuses two independent information streams:

| Signal Layer | Source | Method |
|---|---|---|
| Price momentum | Yahoo Finance (Nifty 50) | 20-day Simple Moving Average |
| Market sentiment | Financial headlines | VADER compound score |

A **Bullish Signal** is generated only when *both* conditions agree simultaneously, giving a stronger, more confident entry indication.

---

## 🛠️ Tech Stack

| Library | Role |
|---|---|
| `yfinance` | Fetch historical OHLCV market data from Yahoo Finance |
| `pandas` | DataFrame manipulation, rolling SMA calculation |
| `numpy` | Array operations for chart coloring |
| `matplotlib` | Dark-themed multi-panel chart generation |
| `vaderSentiment` | Rule-based NLP sentiment scoring (no API key required) |
| `requests` | (Ready for live NewsAPI integration) |

---

## 📁 Project Structure

```
market-sentiment-analyzer/
├── src/
│   └── main.py              # Main analysis pipeline
├── output/
│   └── market_chart.png     # Generated chart (auto-created)
├── data/
│   └── analysis_snapshot.csv  # Full data export (auto-created, git-ignored)
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
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the analyzer
```bash
python src/main.py
```

The script will:
1. Download the last 3 months of Nifty 50 data from Yahoo Finance
2. Compute the 20-day Simple Moving Average
3. Generate mock financial headlines (reproducible, seeded)
4. Score each headline using VADER sentiment analysis
5. Apply the Bullish Signal logic
6. Save a chart to `output/market_chart.png`
7. Print a summary of all Bullish Signal days to the terminal

---

## 📊 Signal Logic

```
Bullish_Signal = True   IF   (Close > SMA_20)   AND   (Sentiment_Score > 0.3)
                 False   otherwise
```

### Why both conditions?

| Condition | What it Checks | Why It Matters |
|---|---|---|
| `Close > SMA_20` | Price is above its 20-day trend | Momentum is upward; buyers are in control |
| `Sentiment > 0.3` | News is clearly positive | Market narrative supports the move |

A signal fires **only when the news confirms what the chart is already showing** — reducing false positives from either source alone.

---

## 🖼️ Sample Output

After running `python src/main.py`, open `output/market_chart.png`:

- **Blue line** — Nifty 50 daily closing price  
- **Orange dashed line** — 20-day SMA trend line  
- **Green dots** — Bullish Signal days  
- **Bottom panel** — Daily VADER compound sentiment score (green = positive, red = negative)  
- **Orange dashed line** (sentiment panel) — 0.3 threshold

---

## ⚠️ Limitations & Future Improvements

### Current Limitations

| Limitation | Description |
|---|---|
| **Mock headlines** | Headlines are randomly generated from a preset pool — they do not correspond to actual news on those dates |
| **Lagging sentiment** | VADER scores headlines that would only be available *after* the market opens; this introduces **look-ahead bias** into backtesting |
| **Lagging SMA** | A 20-day SMA is a lagging indicator by definition; it confirms trends rather than predicting them |
| **Single ticker** | Only Nifty 50 is analyzed; cross-asset signals are not considered |
| **No risk management** | The signal has no associated stop-loss, position sizing, or portfolio allocation logic |

### Roadmap / Future Improvements

- [ ] **Live NewsAPI integration** — Replace mock headlines with real-time financial news via [NewsAPI.org](https://newsapi.org/)
- [ ] **Transformer-based NLP** — Upgrade from VADER to FinBERT or a finance-tuned BERT model for domain-specific accuracy
- [ ] **Multi-indicator confluence** — Add RSI, MACD, Bollinger Bands for stronger signal filtering
- [ ] **Backtesting engine** — Integrate `backtrader` or `vectorbt` to measure historical P&L, win rate, and Sharpe Ratio
- [ ] **Multi-stock support** — Expand to a watchlist of NSE stocks (RELIANCE, TCS, INFY, HDFC, etc.)
- [ ] **Streamlit dashboard** — Build an interactive web UI to visualize signals in real time
- [ ] **Walk-forward validation** — Proper out-of-sample testing to avoid look-ahead bias

---

## 📜 License

This project is licensed under the **MIT License** — free to use, modify, and distribute.

---

## 🙋 Author

**Rohit**  
*Quantitative Finance & Python Enthusiast*

> ⭐ If you found this project useful, please consider starring the repository!
