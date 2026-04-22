# NSE Stock Technical Analysis

**DSP252 Data Analytics & Visualization Lab — IIT Bhilai**

A complete data analytics pipeline applied to 15 NSE-listed equities across 5 sectors (IT, Banking, Auto, FMCG, Energy), covering January 2020 to April 2026.

## Live Dashboard

🔗 [Open Dashboard](https://your-app.streamlit.app) ← update after deploying

## Project Structure

```
nse_collect.py       Data collection + feature engineering (yfinance)
eda_final.ipynb      Exploratory data analysis — 10 plots
model.ipynb          Logistic regression model — 15 stocks
app.py               Streamlit dashboard (3 tabs + model prediction)
requirements.txt     Python dependencies
combined.csv         All 15 stocks stacked (main input)
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

RSI (14) · MACD (12/26/9) · Bollinger Bands (20, 2σ) · EMA 20/50/200 · ATR (14) · OBV · Daily Return · Rolling Volatility

## Model

- **Algorithm:** Logistic Regression (one model per stock)
- **Train:** 2020–2023 · **Test:** 2024–2026
- **Average accuracy:** 50.25% (range: 46.3%–53.7%)
- **Most predictive indicator:** RSI

## Run Locally

```bash
pip install -r requirements.txt

# Step 1: collect data
python nse_collect.py

# Step 2: run dashboard
streamlit run app.py
```

## Tech Stack

Python · pandas · numpy · scikit-learn · Plotly · Streamlit · yfinance
