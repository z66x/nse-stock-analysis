"""
NSE Stock Technical Analysis — Streamlit Dashboard
===================================================
"""

import joblib
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NSE Technical Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────────

SECTOR_COLORS = {
    "IT":      "#4C72B0",
    "Banking": "#DD8452",
    "Auto":    "#55A868",
    "FMCG":    "#C44E52",
    "Energy":  "#8172B2",
}

FEATURES = [
    "rsi", "macd_hist", "bb_pct_b", "atr_pct",
    "rolling_vol_20", "obv_ema", "ema_20_50_cross", "ema_50_200_cross"
]

TICKERS_BY_SECTOR = {
    "IT":      ["TCS.NS", "INFY.NS", "WIPRO.NS"],
    "Banking": ["HDFCBANK.NS", "AXISBANK.NS", "SBIN.NS"],
    "Auto":    ["MARUTI.NS", "M&M.NS", "HEROMOTOCO.NS"],
    "FMCG":    ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS"],
    "Energy":  ["NTPC.NS", "ONGC.NS", "POWERGRID.NS"],
}

TRAIN_END  = "2023-12-31"
TEST_START = "2024-01-01"

# ── Load data ──────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    df = pd.read_csv("combined.csv", index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    summary = pd.read_csv("summary.csv")
    return df, summary

df, summary = load_data()
TICKERS = sorted(df["ticker"].unique().tolist())
SECTORS = sorted(df["sector"].unique().tolist())
TICKER_TO_SECTOR = dict(zip(summary["ticker"], summary["sector"]))

# ── Train / load models ────────────────────────────────────────────────────────

@st.cache_resource
def get_models():
    """
    Train logistic regression models for all stocks.
    Uses @st.cache_resource so models are trained once and reused.
    If .pkl files exist in models/ directory, loads from disk instead.
    """
    models = {}
    os.makedirs("models", exist_ok=True)

    for ticker in TICKERS:
        name = ticker.replace(".NS", "")
        model_path  = f"models/{name}_model.pkl"
        scaler_path = f"models/{name}_scaler.pkl"

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model  = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
        else:
            stock = df[df["ticker"] == ticker].sort_index().dropna(subset=FEATURES + ["target"])
            train = stock.loc[:TRAIN_END]
            if len(train) < 100:
                continue
            X_train = train[FEATURES]
            y_train = train["target"]
            scaler  = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)
            model   = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_scaled, y_train)
            joblib.dump(model,  model_path)
            joblib.dump(scaler, scaler_path)

        models[ticker] = (model, scaler)

    return models

models = get_models()

# ── Prediction helper ──────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_live_row(ticker: str):
    """
    Fetches today's OHLCV from yfinance, appends to historical data,
    recomputes indicators, returns the last row with fresh values.
    TTL = 1 hour so it doesn't hammer yfinance on every interaction.
    """
    import yfinance as yf
    try:
        hist = df[df["ticker"] == ticker].sort_index()[["Open","High","Low","Close","Volume"]].copy()
        today = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
        if isinstance(today.columns, pd.MultiIndex):
            today.columns = today.columns.get_level_values(0)
        today = today[["Open","High","Low","Close","Volume"]]
        # Remove overlap with historical data then append
        today = today[today.index > hist.index.max()]
        if today.empty:
            return hist, hist.index.max(), False   # no new data, use last historical
        combined = pd.concat([hist, today])
        return combined, today.index.max(), True   # True = live data fetched
    except Exception:
        hist = df[df["ticker"] == ticker].sort_index()[["Open","High","Low","Close","Volume"]]
        return hist, hist.index.max(), False


def compute_indicators_live(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Recomputes all 8 model features on the provided OHLCV dataframe."""
    close, high, low, vol = ohlcv["Close"], ohlcv["High"], ohlcv["Low"], ohlcv["Volume"]

    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs       = avg_gain / avg_loss
    ohlcv["rsi"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    ohlcv["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()

    sma20  = close.rolling(20).mean()
    std20  = close.rolling(20).std()
    bb_lo  = sma20 - 2 * std20
    bb_hi  = sma20 + 2 * std20
    ohlcv["bb_pct_b"] = (close - bb_lo) / (bb_hi - bb_lo)

    prev_close = close.shift(1)
    tr = pd.concat([high-low, (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    ohlcv["atr_pct"] = tr.ewm(com=13, adjust=False).mean() / close

    ohlcv["rolling_vol_20"] = close.pct_change().rolling(20).std()

    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * vol).cumsum()
    ohlcv["obv_ema"] = obv.ewm(span=20, adjust=False).mean()

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    ohlcv["ema_20_50_cross"]  = (ema20 > ema50).astype(int)
    ohlcv["ema_50_200_cross"] = (ema50 > ema200).astype(int)

    return ohlcv


def predict_stock(ticker: str):
    """Returns (direction, confidence, top3, signal_date, prediction_date, is_live)."""
    if ticker not in models:
        return None, None, None, None, None, False

    ohlcv, data_date, is_live = fetch_live_row(ticker)
    ohlcv = compute_indicators_live(ohlcv.copy())
    ohlcv = ohlcv.dropna(subset=FEATURES)
    if ohlcv.empty:
        return None, None, None, None, None, False

    latest    = ohlcv[FEATURES].iloc[[-1]]
    model, scaler = models[ticker]
    X_scaled  = scaler.transform(latest)
    proba     = model.predict_proba(X_scaled)[0]
    direction = "UP ↑" if proba[1] >= 0.5 else "DOWN ↓"
    confidence = max(proba) * 100

    contribs = dict(zip(FEATURES, model.coef_[0] * X_scaled[0]))
    top3     = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

    signal_date     = data_date.strftime("%d %b %Y")
    next_trading    = data_date + pd.offsets.BDay(1)
    prediction_date = next_trading.strftime("%d %b %Y")

    return direction, confidence, top3, signal_date, prediction_date, is_live

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📈 NSE Dashboard")
    st.markdown("15 stocks · 5 sectors · 2020–2026")
    st.divider()

    selected_ticker = st.selectbox(
        "Select stock",
        options=TICKERS,
        format_func=lambda t: t.replace(".NS", ""),
        index=0,
    )
    selected_sector_str = TICKER_TO_SECTOR.get(selected_ticker, "")
    st.caption(f"Sector: {selected_sector_str}")

    st.divider()

    date_range = st.date_input(
        "Date range",
        value=[pd.Timestamp("2022-01-01"), df.index.max()],
        min_value=pd.Timestamp("2020-01-01"),
        max_value=df.index.max(),
    )

    show_bb  = st.toggle("Bollinger Bands", value=True)
    ema_opts = st.multiselect(
        "Show EMAs",
        options=["EMA 20", "EMA 50", "EMA 200"],
        default=["EMA 20", "EMA 50"],
    )

    st.divider()
    st.caption("DSP252 · Data Analytics & Visualization Lab · IIT Bhilai")

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📊 Stock Analysis", "🏭 Sector Comparison", "⚖️ Risk & Return"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Stock Analysis
# ══════════════════════════════════════════════════════════════════════════════

with tab1:

    # Date filter
    start_date = pd.Timestamp(date_range[0]) if len(date_range) == 2 else pd.Timestamp("2022-01-01")
    end_date   = pd.Timestamp(date_range[1]) if len(date_range) == 2 else df.index.max()

    stock = df[df["ticker"] == selected_ticker].sort_index().loc[start_date:end_date]
    s_row = summary[summary["ticker"] == selected_ticker].iloc[0]

    # ── Metric cards ──────────────────────────────────────────────────────────
    period_ret   = round((stock["Close"].iloc[-1] / stock["Close"].iloc[0] - 1) * 100, 2) if len(stock) > 1 else 0
    latest_rsi   = round(stock["rsi"].iloc[-1], 1) if not stock.empty else 0
    latest_atr   = round(stock["atr_pct"].iloc[-1] * 100, 2) if not stock.empty else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sector",          selected_sector_str)
    c2.metric("Period Return",   f"{period_ret:+.1f}%")
    c3.metric("Current RSI",     f"{latest_rsi}")
    c4.metric("Max Drawdown",    f"{s_row['max_drawdown_pct']:.1f}%")
    c5.metric("Ann. Volatility", f"{s_row['annualised_vol']:.1f}%")

    st.divider()

    # ── Model prediction card ─────────────────────────────────────────────────
    direction, confidence, top3, signal_date, pred_date, is_live = predict_stock(selected_ticker)

    pred_col, chart_col = st.columns([1, 3])

    with pred_col:
        st.markdown("#### Model Signal")
        if direction:
            color = "green" if "UP" in direction else "red"
            st.markdown(
                f"<h2 style='color:{color};margin:0'>{direction}</h2>",
                unsafe_allow_html=True
            )
            st.metric("Confidence", f"{confidence:.1f}%")

            if is_live:
                st.success(f"🟢 Live · Signal: {signal_date}")
            else:
                st.warning(f"🟡 Cached · Signal: {signal_date}")
            st.caption(f"Prediction for: **{pred_date}**")
            st.caption("Logistic Regression · Train: 2020–2023")

            st.markdown("**Top 3 drivers:**")
            for feat, val in top3:
                arrow = "▲" if val > 0 else "▼"
                st.markdown(f"`{feat}` {arrow} `{val:+.3f}`")
        else:
            st.warning("Model not available for this stock.")

    # ── Main chart ────────────────────────────────────────────────────────────
    with chart_col:
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.55, 0.23, 0.22],
            vertical_spacing=0.03,
            subplot_titles=[
                f"{selected_ticker.replace('.NS','')} — Price",
                "RSI (14)",
                "MACD (12, 26, 9)"
            ]
        )

        fig.add_trace(go.Candlestick(
            x=stock.index, open=stock["Open"], high=stock["High"],
            low=stock["Low"], close=stock["Close"],
            name="OHLC",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            showlegend=False,
        ), row=1, col=1)

        if show_bb:
            fig.add_trace(go.Scatter(x=stock.index, y=stock["bb_upper"],
                name="BB Upper", line=dict(color="rgba(130,130,130,0.5)", width=1, dash="dot")), row=1, col=1)
            fig.add_trace(go.Scatter(x=stock.index, y=stock["bb_lower"],
                name="BB Lower", fill="tonexty", fillcolor="rgba(130,130,130,0.07)",
                line=dict(color="rgba(130,130,130,0.5)", width=1, dash="dot")), row=1, col=1)

        ema_map = {"EMA 20": ("ema_20", "#f39c12"), "EMA 50": ("ema_50", "#3498db"), "EMA 200": ("ema_200", "#e74c3c")}
        for label in ema_opts:
            col_name, color = ema_map[label]
            fig.add_trace(go.Scatter(x=stock.index, y=stock[col_name],
                name=label, line=dict(color=color, width=1.2)), row=1, col=1)

        fig.add_trace(go.Scatter(x=stock.index, y=stock["rsi"],
            line=dict(color="#9b59b6", width=1.3), showlegend=False), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red",   line_width=1, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=2, col=1)
        fig.add_hrect(y0=30, y1=70, fillcolor="gray", opacity=0.04, row=2, col=1)

        hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in stock["macd_hist"]]
        fig.add_trace(go.Bar(x=stock.index, y=stock["macd_hist"],
            marker_color=hist_colors, opacity=0.65, showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=stock.index, y=stock["macd"],
            line=dict(color="#3498db", width=1.2), showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=stock.index, y=stock["macd_signal"],
            line=dict(color="#e74c3c", width=1.2), showlegend=False), row=3, col=1)

        fig.update_layout(
            height=640, template="plotly_white",
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.01),
            margin=dict(l=40, r=120, t=40, b=20),
            hovermode="x unified",
        )
        fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
        fig.update_yaxes(title_text="RSI",       row=2, col=1, range=[0, 100])
        fig.update_yaxes(title_text="MACD",      row=3, col=1)

        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Sector Comparison
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    col_left, col_right = st.columns([1, 3])

    with col_left:
        selected_sectors = st.multiselect(
            "Sectors", options=SECTORS, default=SECTORS
        )
        metric = st.radio(
            "Metric", options=[
                "Cumulative Return",
                "Rolling Volatility",
                "RSI",
                "BB Width",
            ]
        )

    metric_col_map = {
        "Cumulative Return": None,
        "Rolling Volatility": "rolling_vol_20",
        "RSI": "rsi",
        "BB Width": "bb_width",
    }

    with col_right:
        if selected_sectors:
            filtered = df[df["sector"].isin(selected_sectors)]
            fig_ts   = go.Figure()

            for sector in selected_sectors:
                s_df = filtered[filtered["sector"] == sector]
                col_key = metric_col_map[metric]

                if metric == "Cumulative Return":
                    cum_list = []
                    for ticker in s_df["ticker"].unique():
                        t_df = s_df[s_df["ticker"] == ticker].sort_index()
                        cum_list.append((1 + t_df["daily_return"]).cumprod() - 1)
                    avg_series = pd.concat(cum_list, axis=1).mean(axis=1) * 100
                else:
                    avg_series = s_df.groupby(s_df.index)[col_key].mean()

                fig_ts.add_trace(go.Scatter(
                    x=avg_series.index, y=avg_series,
                    name=sector, line=dict(color=SECTOR_COLORS[sector], width=1.8),
                ))

            if metric == "RSI":
                fig_ts.add_hline(y=70, line_dash="dash", line_color="red",   annotation_text="Overbought")
                fig_ts.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")

            fig_ts.update_layout(
                title=f"Sector Comparison — {metric}",
                template="plotly_white", hovermode="x unified",
                height=380,
                margin=dict(l=50, r=120, t=50, b=20),
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.01),
            )
            st.plotly_chart(fig_ts, use_container_width=True)

        # Sector avg return bar
        sector_avg = (
            summary[summary["sector"].isin(selected_sectors)]
            .groupby("sector")["total_return_pct"]
            .mean().reset_index()
            .sort_values("total_return_pct", ascending=True)
        )
        fig_bar = go.Figure(go.Bar(
            x=sector_avg["total_return_pct"],
            y=sector_avg["sector"],
            orientation="h",
            marker_color=[SECTOR_COLORS[s] for s in sector_avg["sector"]],
            text=[f"{v:.1f}%" for v in sector_avg["total_return_pct"]],
            textposition="outside",
        ))
        fig_bar.update_layout(
            title="Average 5-Year Total Return by Sector",
            template="plotly_white", height=300,
            margin=dict(l=80, r=80, t=50, b=20),
            xaxis_title="Total Return (%)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Risk & Return
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    col_a, col_b = st.columns(2)

    # Bubble chart
    with col_a:
        summary_plot = summary.copy()
        summary_plot["label"] = summary_plot["ticker"].str.replace(".NS", "")

        fig_bubble = go.Figure()
        for sector in SECTORS:
            s = summary_plot[summary_plot["sector"] == sector]
            fig_bubble.add_trace(go.Scatter(
                x=s["annualised_vol"], y=s["total_return_pct"],
                mode="markers+text",
                name=sector,
                marker=dict(
                    size=np.log1p(s["avg_volume"]) * 2.2,
                    color=SECTOR_COLORS[sector], opacity=0.75,
                    line=dict(width=1, color="white"),
                ),
                text=s["label"], textposition="top center", textfont=dict(size=10),
                customdata=np.stack([s["max_drawdown_pct"], s["avg_rsi"]], axis=-1),
                hovertemplate=(
                    "<b>%{text}</b><br>Return: %{y:.1f}%<br>"
                    "Vol: %{x:.1f}%<br>Max DD: %{customdata[0]:.1f}%<extra></extra>"
                )
            ))
        fig_bubble.update_layout(
            title="Risk vs Return  (bubble size ∝ avg daily volume)",
            xaxis_title="Annualised Volatility (%)",
            yaxis_title="Total Return (%)",
            template="plotly_white", height=460,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.01),
            margin=dict(l=50, r=120, t=60, b=40),
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

    # Drawdown bar
    with col_b:
        dd = summary.sort_values("max_drawdown_pct").copy()
        dd["label"] = dd["ticker"].str.replace(".NS", "")
        dd["color"] = dd["max_drawdown_pct"].apply(
            lambda x: "#ef5350" if x < -50 else "#ff9800" if x < -35 else "#66bb6a"
        )
        fig_dd = go.Figure(go.Bar(
            x=dd["max_drawdown_pct"], y=dd["label"],
            orientation="h", marker_color=dd["color"],
            text=[f"{v:.1f}%" for v in dd["max_drawdown_pct"]],
            textposition="outside",
        ))
        fig_dd.update_layout(
            title="Max Drawdown by Stock",
            xaxis_title="Max Drawdown (%)",
            template="plotly_white", height=460,
            margin=dict(l=90, r=60, t=50, b=20),
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    # Correlation heatmap
    st.divider()
    pivot = df.pivot_table(index=df.index, columns="ticker", values="daily_return")
    pivot.columns = pivot.columns.str.replace(".NS", "")
    order = [t.replace(".NS","") for s in ["IT","Banking","Auto","FMCG","Energy"]
             for t in df[df["sector"]==s]["ticker"].unique()]
    order = [o for o in order if o in pivot.columns]
    corr  = pivot[order].corr()

    fig_corr = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdYlGn", zmin=-0.2, zmax=1,
        text=corr.round(2).values,
        texttemplate="%{text}", textfont=dict(size=9),
    ))
    fig_corr.update_layout(
        title="Pairwise Return Correlation — All 15 Stocks (ordered by sector)",
        template="plotly_white", height=480,
        margin=dict(l=80, r=20, t=60, b=80),
    )
    st.plotly_chart(fig_corr, use_container_width=True)
