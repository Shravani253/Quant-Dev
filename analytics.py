import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import timedelta
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from sklearn.linear_model import HuberRegressor, TheilSenRegressor
import matplotlib.pyplot as plt
import seaborn as sns

TICKS_FILE = "ticks_live.csv"
ANALYTICS_FILE = "analytics_output.csv"

st.set_page_config(layout="wide", page_title="Quant Analytics")

st.title("Quant Analytics — Aligned with ticks_live collector")

@st.cache_data(ttl=30)
def load_ticks(path=TICKS_FILE):
    df = pd.read_csv(path)
    cols = set(df.columns)
    if cols == {"T", "s", "p", "q"} or {"T", "s", "p", "q"}.issubset(cols):
        df = df.rename(columns={"T": "ts", "s": "symbol", "p": "price", "q": "size"})
    elif {"ts", "s", "p", "q"}.issubset(cols):
        df = df.rename(columns={"s": "symbol", "p": "price", "q": "size"})
    if "ts" in df.columns and df["ts"].dtype.kind in {"i", "u", "f"}:
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    else:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df = df.dropna(subset=["ts"])
    df = df.sort_values("ts")
    df["price"] = df["price"].astype(float)
    df["size"] = df["size"].astype(float)
    return df

def pivot_prices(df):
    price_df = df.pivot_table(index="ts", columns="symbol", values="price", aggfunc="last").sort_index().ffill()
    price_df = price_df.rename(columns={c: f"{c}_price" for c in price_df.columns})
    return price_df

def kalman_dynamic_hedge(y, x):
    n = len(y)
    betas = np.zeros(n)
    state = np.zeros(2)
    P = np.eye(2) * 1e6
    Q = np.eye(2) * 1e-5
    R = 1e-2
    for t in range(n):
        xt = np.array([1.0, x[t]])
        state_pred = state
        P_pred = P + Q
        y_pred = xt @ state_pred
        S = float(xt @ P_pred @ xt.T + R)
        K = (P_pred @ xt) / S
        resid = y[t] - y_pred
        state = state_pred + K * resid
        P = (np.eye(2) - np.outer(K, xt)) @ P_pred
        betas[t] = state[1]
    return betas

def compute_hedges(price_df):
    price_df = price_df.dropna()
    price_df = price_df.reset_index()
    y = price_df["BTCUSDT_price"].values
    x = price_df["ETHUSDT_price"].values
    X_sm = sm.add_constant(x)
    ols = sm.OLS(y, X_sm).fit()
    hedge_ols = float(ols.params[1])
    huber = HuberRegressor().fit(x.reshape(-1, 1), y)
    hedge_huber = float(huber.coef_[0])
    theil = TheilSenRegressor().fit(x.reshape(-1, 1), y)
    hedge_theil = float(theil.coef_[0])
    kalman_betas = kalman_dynamic_hedge(y, x)
    price_df["hedge_ols"] = hedge_ols
    price_df["hedge_huber"] = hedge_huber
    price_df["hedge_theil"] = hedge_theil
    price_df["hedge_kalman"] = kalman_betas
    price_df["spread_ols"] = price_df["BTCUSDT_price"] - hedge_ols * price_df["ETHUSDT_price"]
    price_df["rolling_mean"] = price_df["spread_ols"].rolling(window=60, min_periods=1).mean()
    price_df["rolling_std"] = price_df["spread_ols"].rolling(window=60, min_periods=1).std().fillna(0.0)
    price_df["zscore"] = (price_df["spread_ols"] - price_df["rolling_mean"]) / price_df["rolling_std"].replace(0, np.nan)
    adf_pvalue = adfuller(price_df["spread_ols"].dropna())[1]
    price_df["adf_pvalue"] = adf_pvalue
    return price_df

def liquidity_metrics(df):
    df_min = df.set_index("ts").resample("1T").agg({"size":"sum","price":"count"})
    df_min = df_min.rename(columns={"size":"vol_sum","price":"trades"})
    df_min["avg_size"] = df_min["vol_sum"] / df_min["trades"].replace(0, np.nan)
    return df_min

def cross_corr_heatmap(price_df):
    returns = price_df[["BTCUSDT_price","ETHUSDT_price"]].pct_change().dropna()
    corr = returns.corr()
    fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(corr, annot=True, ax=ax, vmin=-1, vmax=1)
    return fig, corr

def backtest_mean_reversion(price_df, entry_z=2.0, exit_z=0.0):
    price_df = price_df.copy().reset_index(drop=True)
    position = 0
    entry_idx = None
    pnl = []
    pos_series = []
    for i in range(len(price_df)-1):
        z = price_df.loc[i, "zscore"]
        spread_now = price_df.loc[i, "spread_ols"]
        spread_next = price_df.loc[i+1, "spread_ols"]
        if position == 0 and z > entry_z:
            position = -1
            entry_idx = i
        elif position == 0 and z < -entry_z:
            position = 1
            entry_idx = i
        elif position != 0 and abs(z) < exit_z:
            position = 0
            entry_idx = None
        pnl_step = position * (spread_next - spread_now)
        pnl.append(pnl_step)
        pos_series.append(position)
    pnl = np.array(pnl)
    cum_pnl = pnl.cumsum()
    total_pnl = float(cum_pnl[-1]) if len(cum_pnl)>0 else 0.0
    price_df = price_df.iloc[:-1].copy()
    price_df["position"] = pos_series
    price_df["pnl_step"] = np.concatenate([pnl, np.array([])])[:len(price_df)]
    price_df["cum_pnl"] = price_df["pnl_step"].cumsum().fillna(0.0)
    metrics = {
        "total_pnl": total_pnl,
        "trades": int((np.diff(np.concatenate([[0], price_df["position"].values]))!=0).sum()/2) if len(price_df)>0 else 0,
        "max_drawdown": float((price_df["cum_pnl"].cummax()-price_df["cum_pnl"]).max()) if len(price_df)>0 else 0.0
    }
    return price_df, metrics

def time_series_table(price_df):
    table = price_df.set_index("ts").resample("1T").agg({
        "BTCUSDT_price":"last",
        "ETHUSDT_price":"last",
        "spread_ols":"mean",
        "zscore":"mean",
        "hedge_ols":"last",
        "hedge_kalman":"last"
    }).dropna()
    table["vol_sum"] = price_df.set_index("ts").resample("1T")["ETHUSDT_price"].count()
    return table

df_ticks = load_ticks()
price_df = pivot_prices(df_ticks)
price_df = price_df.dropna()
if not {"BTCUSDT_price", "ETHUSDT_price"}.issubset(price_df.columns):
    st.error("Required symbols missing in ticks data. Ensure BTCUSDT and ETHUSDT present.")
else:
    price_df = compute_hedges(price_df)
    df_min_liq = liquidity_metrics(df_ticks)
    fig_corr, corr = cross_corr_heatmap(price_df)
    bt_df, bt_metrics = backtest_mean_reversion(price_df)
    ts_table = time_series_table(price_df)
    st.header("Quick Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(price_df)}")
    col2.metric("ADF p-value", f"{price_df['adf_pvalue'].iloc[0]:.4f}")
    col3.metric("Latest zscore", f"{price_df['zscore'].iloc[-1]:.2f}")
    st.header("Backtest (Mini mean-reversion)")
    st.write(bt_metrics)
    st.line_chart(bt_df.set_index("ts")["cum_pnl"])
    st.header("Cross-correlation Heatmap")
    st.pyplot(fig_corr)
    st.header("Liquidity (per minute)")
    st.dataframe(df_min_liq.tail(200))
    st.header("Time-series table (1-minute aggregation)")
    st.dataframe(ts_table.tail(200))
    csv = ts_table.reset_index().to_csv(index=False).encode()
    st.download_button("Download aggregated CSV", data=csv, file_name="ts_table.csv", mime="text/csv")
    st.header("Alerts & Creative Summaries")
    recent = price_df.tail(30).iloc[-1]
    alerts = []
    if abs(recent["zscore"]) > 2 and recent["adf_pvalue"] < 0.05:
        alerts.append(f"Mean-reversion signal: z={recent['zscore']:.2f}, ADF p={recent['adf_pvalue']:.3f}")
    if df_min_liq["avg_size"].iloc[-1] < df_min_liq["avg_size"].median() * 0.5:
        alerts.append("Low liquidity in last minute")
    for a in alerts:
        st.warning(a)
    st.header("Creative analytics")
    st.subheader("Rolling correlation (30s window)")
    rolling_corr = price_df[["BTCUSDT_price","ETHUSDT_price"]].pct_change().rolling(window=30).corr().dropna()
    if not rolling_corr.empty:
        last_corr = rolling_corr.groupby(level=0).last().iloc[-1,1]
        st.write(f"Latest rolling correlation: {last_corr:.3f}")
    st.subheader("Volatility regime")
    vol = price_df["spread_ols"].rolling(window=60).std()
    regime = pd.cut(vol, bins=[-1, vol.quantile(0.33), vol.quantile(0.66), vol.max()], labels=["low","mid","high"])
    st.write("Current volatility regime:", regime.iloc[-1])
    st.header("Download full analytics CSV")
    analytics_csv = price_df.to_csv(index=False).encode()
    st.download_button("Download analytics_output.csv", data=analytics_csv, file_name="analytics_output.csv", mime="text/csv")
