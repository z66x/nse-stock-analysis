# NSE Stock Technical Analysis

**DSP252 Data Analytics & Visualization Lab — IIT Bhilai**

A complete data analytics pipeline applied to 15 NSE-listed equities across 5 sectors, covering January 2020 to April 2026. Includes exploratory data analysis, technical indicator engineering, a logistic regression direction classifier, and a live interactive dashboard.

## Live Dashboard

🔗 **[nse-stock-signal.streamlit.app](https://nse-stock-signal.streamlit.app)**

## Project Structure

```
app.py               Streamlit dashboard (3 tabs + live model signal)
nse_collect.py       Data collection + feature engineering (yfinance)
eda_final.ipynb      Exploratory data analysis — 10 plots
model.ipynb          Logistic regression model — 15 stocks
requirements.txt     Python dependencies
combined.csv         All 15 stocks stacked (main app input)
summary.csv          One aggregated row per stock
models/              Trained .pkl files (auto-generated on first run)
processed/           Per-stock CSVs with all indicator columns
raw/                 Per-stock raw OHLCV CSVs
```

## Stocks Covered

| Sector | Stocks |
|--------|--------|
| IT | TCS, Infosys, Wipro |
| Banking | HDFC Bank, Axis Bank, SBI |
| Auto | Maruti, M&M, Hero MotoCorp |
| FMCG | HUL, ITC, Nestle India |
| Energy | NTPC, ONGC, Power Grid |

## Technical Indicators Engineered

All indicators computed from scratch using `pandas` and `numpy` — no external TA library.

| Indicator | Type | Used in model |
|-----------|------|---------------|
| RSI (14) | Momentum | ✓ |
| MACD histogram (12/26/9) | Trend | ✓ |
| Bollinger Band %B (20, 2σ) | Volatility | ✓ |
| ATR% (14) | Volatility | ✓ |
| Rolling volatility (20d) | Risk | ✓ |
| OBV EMA | Volume | ✓ |
| EMA crossover flags (20/50, 50/200) | Trend | ✓ |
| EMA 20 / 50 / 200 | Trend | — |
| Daily return, log return | Returns | — |

## Model

- **Algorithm:** Logistic Regression (one model per stock, 15 total)
- **Train:** 2020–2023 · **Test:** 2024–2026 (time-based split — no lookahead)
- **Average test accuracy:** 50.25% (range: 46.3%–53.7%)
- **Best stock:** HINDUNILVR — 53.72%
- **Most predictive indicator:** RSI
- **Accuracy note:** consistent with weak-form Efficient Market Hypothesis — if technical indicators reliably predicted next-day direction, arbitrage would eliminate the edge immediately

## Dashboard Features

**Stock Analysis tab**
- Candlestick chart with Bollinger Bands + EMA overlays
- RSI panel with overbought/oversold zones
- MACD histogram + signal line
- Live model signal — fetches today's OHLCV via yfinance, recomputes indicators, predicts tomorrow's direction with confidence score and top 3 feature drivers

**Sector Comparison tab**
- Sector-average time series for cumulative return, rolling volatility, RSI, or BB width
- 5-year average total return bar chart by sector

**Risk & Return tab**
- Bubble chart (risk vs return, bubble size = avg daily volume)
- Max drawdown bar chart
- 15×15 pairwise return correlation heatmap

## Run Locally

```bash
git clone https://github.com/z66x/nse-stock-analysis.git
cd nse-stock-analysis
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack

`Python` · `pandas` · `numpy` · `scikit-learn` · `plotly` · `streamlit` · `yfinance` · `joblib`
