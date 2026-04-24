"""
NSE Stock Data Collection + Feature Engineering
================================================
Pulls daily OHLCV data for 15 NSE-listed stocks across 5 sectors
from Yahoo Finance (via yfinance), computes 25+ technical indicators
from scratch, and saves clean CSVs ready for EDA and modelling.

Outputs:
    raw/           one CSV per ticker (raw OHLCV only)
    processed/     one CSV per ticker (OHLCV + all indicators)
    combined.csv   all 15 tickers stacked (main EDA input)
    summary.csv    one summary row per ticker

Note: indicators are computed manually using pandas/numpy.
"""

import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

TICKERS = {
    "IT":      ["TCS.NS", "INFY.NS", "WIPRO.NS"],
    "Banking": ["HDFCBANK.NS", "AXISBANK.NS", "SBIN.NS"],
    "Auto":    ["MARUTI.NS", "M&M.NS", "HEROMOTOCO.NS"],
    "FMCG":   ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS"],
    "Energy":  ["NTPC.NS", "ONGC.NS", "POWERGRID.NS"],
}

BENCHMARK = "^NSEI"   # Nifty 50 index — used as market benchmark
START = "2020-01-01"  # start from Jan 2020 to include COVID crash
END = datetime.today().strftime("%Y-%m-%d")  # pull up to today

RAW_DIR  = "raw"
PROC_DIR = "processed"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame with columns [Open, High, Low, Close, Volume].
    Returns the same DataFrame with all technical indicators appended.
    All formulas implemented manually in pandas/numpy — no external TA library.
    """
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # ── Returns ───────────────────────────────────────────────────────────
    # pct_change gives (today - yesterday) / yesterday
    df["daily_return"]    = close.pct_change()
    df["log_return"]      = np.log(close / close.shift(1))
    df["rolling_vol_20"]  = df["daily_return"].rolling(20).std()   # 20-day realised vol

    # ── EMAs ──────────────────────────────────────────────────────────────
    # ewm = exponential weighted mean, adjust=False uses recursive formula
    df["ema_20"]  = close.ewm(span=20,  adjust=False).mean()
    df["ema_50"]  = close.ewm(span=50,  adjust=False).mean()
    df["ema_200"] = close.ewm(span=200, adjust=False).mean()

    # EMA crossover flags (1 = fast above slow, 0 = fast below slow)
    df["ema_20_50_cross"]  = (df["ema_20"] > df["ema_50"]).astype(int)
    df["ema_50_200_cross"] = (df["ema_50"] > df["ema_200"]).astype(int)

    # ── RSI (14) ──────────────────────────────────────────────────────────
    # Wilder smoothing = ewm with com=13 (equivalent to span=27, alpha=1/14)
    delta     = close.diff()
    gain      = delta.clip(lower=0)
    loss      = -delta.clip(upper=0)
    avg_gain  = gain.ewm(com=13, adjust=False).mean()   # Wilder smoothing
    avg_loss  = loss.ewm(com=13, adjust=False).mean()
    rs        = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # Overbought / oversold flags
    df["rsi_overbought"] = (df["rsi"] > 70).astype(int)
    df["rsi_oversold"]   = (df["rsi"] < 30).astype(int)

    # ── MACD (12, 26, 9) ──────────────────────────────────────────────────
    ema_12          = close.ewm(span=12, adjust=False).mean()
    ema_26          = close.ewm(span=26, adjust=False).mean()
    df["macd"]      = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bullish / bearish crossover flags
    df["macd_bullish"] = (
        (df["macd"] > df["macd_signal"]) &
        (df["macd"].shift(1) <= df["macd_signal"].shift(1))
    ).astype(int)
    df["macd_bearish"] = (
        (df["macd"] < df["macd_signal"]) &
        (df["macd"].shift(1) >= df["macd_signal"].shift(1))
    ).astype(int)

    # ── Bollinger Bands (20, 2σ) ──────────────────────────────────────────
    # std over 20 days gives the band width; 2σ covers ~95% of price action
    sma_20             = close.rolling(20).mean()
    std_20             = close.rolling(20).std()
    df["bb_mid"]       = sma_20
    df["bb_upper"]     = sma_20 + 2 * std_20
    df["bb_lower"]     = sma_20 - 2 * std_20
    df["bb_width"]     = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]  # normalised width
    df["bb_pct_b"]     = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])  # 0-1 position

    # Touch flags
    df["bb_upper_touch"] = (close >= df["bb_upper"]).astype(int)
    df["bb_lower_touch"] = (close <= df["bb_lower"]).astype(int)

    # ── ATR (14) ──────────────────────────────────────────────────────────
    # True Range accounts for overnight gaps (not just intraday high-low)
    prev_close     = close.shift(1)
    tr             = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr"]      = tr.ewm(com=13, adjust=False).mean()   # Wilder smoothing
    df["atr_pct"]  = df["atr"] / close                     # ATR as % of price (normalised)

    # ── OBV ───────────────────────────────────────────────────────────────
    # Adds volume on up days, subtracts on down days — tracks money flow
    direction    = np.sign(close.diff()).fillna(0)
    df["obv"]    = (direction * vol).cumsum()
    df["obv_ema"] = df["obv"].ewm(span=20, adjust=False).mean()   # smoothed OBV trend

    # ── Target variable ───────────────────────────────────────────────────
    # Binary classification target: 1 = price goes UP tomorrow, 0 = DOWN
    # shift(-1) looks one day forward — last row will be NaN (no tomorrow)
    df["target"] = (close.shift(-1) > close).astype(int)

    return df


# Data fetch

def fetch_ticker(ticker: str, sector: str) -> pd.DataFrame | None:
    print(f"  Fetching {ticker}...")
    try:
        raw = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=True)
        if raw.empty:
            print(f"  [WARN] No data for {ticker}")
            return None

        # yfinance returns MultiIndex columns when auto_adjust=True — flatten
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw = raw[["Open", "High", "Low", "Close", "Volume"]].copy()

        # asfreq("B") reindexes to business days, ffill fills market holidays
        # with the previous trading day's closing price
        raw = raw.asfreq("B").ffill()   # business day frequency

        # Save raw CSV
        raw.to_csv(f"{RAW_DIR}/{ticker.replace('.NS', '')}.csv")

        # Compute indicators
        df = compute_indicators(raw.copy())

        # Tag with metadata
        df["ticker"] = ticker
        df["sector"] = sector
        df["name"]   = ticker.replace(".NS", "")

        # EMA-200 needs 200 days of data before producing a valid value
        # dropping these avoids NaN values going into model training
        df = df.dropna(subset=["ema_200", "rsi", "macd", "bb_mid", "atr"])

        # Save processed CSV
        df.to_csv(f"{PROC_DIR}/{ticker.replace('.NS', '')}_processed.csv")

        print(f"    {len(df)} trading days after indicator warmup.")
        return df

    except Exception as e:
        print(f"  [ERROR] {ticker}: {e}")
        return None


# Summary row per ticker 

def build_summary_row(df: pd.DataFrame, ticker: str, sector: str) -> dict:
    close = df["Close"]
    return {
        "ticker":              ticker,
        "sector":              sector,
        "start_date":          str(df.index[0].date()),
        "end_date":            str(df.index[-1].date()),
        "trading_days":        len(df),
        "start_price":         round(close.iloc[0], 2),
        "end_price":           round(close.iloc[-1], 2),
        "total_return_pct":    round((close.iloc[-1] / close.iloc[0] - 1) * 100, 2),
        "avg_daily_return":    round(df["daily_return"].mean() * 100, 4),
        "annualised_vol":      round(df["daily_return"].std() * np.sqrt(252) * 100, 2),
        "max_drawdown_pct":    round(((close / close.cummax()) - 1).min() * 100, 2),
        "avg_rsi":             round(df["rsi"].mean(), 2),
        "pct_days_overbought": round(df["rsi_overbought"].mean() * 100, 2),
        "pct_days_oversold":   round(df["rsi_oversold"].mean() * 100, 2),
        "avg_bb_width":        round(df["bb_width"].mean(), 4),
        "avg_atr_pct":         round(df["atr_pct"].mean() * 100, 4),
        "macd_bullish_signals":df["macd_bullish"].sum(),
        "macd_bearish_signals":df["macd_bearish"].sum(),
        "avg_volume":          int(df["Volume"].mean()),
    }

def main():
    all_dfs    = []
    summaries  = []

    # Fetch benchmark separately
    print("\n── Nifty 50 (benchmark) ──")
    nifty_raw = yf.download(BENCHMARK, start=START, end=END, progress=False, auto_adjust=True)
    if isinstance(nifty_raw.columns, pd.MultiIndex):
        nifty_raw.columns = nifty_raw.columns.get_level_values(0)
    nifty_raw[["Close"]].to_csv(f"{RAW_DIR}/NIFTY50.csv")
    print(f"  {len(nifty_raw)} days saved.")

    # Fetch all stocks
    for sector, tickers in TICKERS.items():
        print(f"\n── {sector} ──")
        for ticker in tickers:
            df = fetch_ticker(ticker, sector)
            if df is not None:
                all_dfs.append(df)
                summaries.append(build_summary_row(df, ticker, sector))
            time.sleep(0.5)   # be gentle with yfinance

    # Combined dataset (all tickers stacked)
    combined = pd.concat(all_dfs, axis=0)
    combined.to_csv("combined.csv")

    # Summary table
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv("summary.csv", index=False)

    print(f"""
Done
  raw/              → {len(os.listdir(RAW_DIR))} files (raw OHLCV per ticker)
  processed/        → {len(os.listdir(PROC_DIR))} files (OHLCV + indicators)
  combined.csv      → {len(combined):,} rows  ({combined['ticker'].nunique()} tickers stacked)
  summary.csv       → {len(summary_df)} rows  (one per ticker)

Columns in processed CSVs:
  OHLCV:       Open, High, Low, Close, Volume
  Returns:     daily_return, log_return, rolling_vol_20
  EMAs:        ema_20, ema_50, ema_200, ema_20_50_cross, ema_50_200_cross
  RSI:         rsi, rsi_overbought, rsi_oversold
  MACD:        macd, macd_signal, macd_hist, macd_bullish, macd_bearish
  BB:          bb_mid, bb_upper, bb_lower, bb_width, bb_pct_b, bb_upper_touch, bb_lower_touch
  ATR:         atr, atr_pct
  OBV:         obv, obv_ema
  Target:      target  (1 = next day up, 0 = next day down)
""")

if __name__ == "__main__":
    main()
