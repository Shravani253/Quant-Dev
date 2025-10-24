import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns

# --- Streamlit Page Setup ---
st.set_page_config(layout="wide", page_title="Quant Analytics Dashboard")
st.title("📈 Quant Developer Analytics Dashboard")

BACKEND_URL = "http://127.0.0.1:8000"

# --- Load Analytics Data ---
@st.cache_data(ttl=5)
def load_analytics():
    r = requests.get(f"{BACKEND_URL}/analytics_data")
    if r.status_code == 200:
        data = r.json()
        if isinstance(data, list) and len(data) > 0 and "ts" in data[0]:
            return pd.DataFrame(data)
    st.error("No analytics data found. Make sure backend and ticks.py are running.")
    return pd.DataFrame()

df = load_analytics()
if df.empty:
    st.stop()

# --- Data Cleaning ---
df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
df = df.dropna(subset=["ts"]).sort_values("ts")

# --- Metrics Display ---
latest = df.iloc[-1]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest Z-Score", f"{latest['zscore']:.2f}")
col2.metric("ADF p-value", f"{latest['adf_pvalue']:.4f}")
col3.metric("OLS Hedge", f"{latest['hedge_ols']:.4f}")
col4.metric("Kalman Hedge", f"{latest['hedge_kalman']:.4f}")

# --- Price Series Plot ---
fig1, ax1 = plt.subplots()
ax1.plot(df["ts"], df["BTCUSDT_price"], label="BTCUSDT", linewidth=1)
ax1.plot(df["ts"], df["ETHUSDT_price"], label="ETHUSDT", linewidth=1)
ax1.set_title("Price Series")
ax1.legend()
st.pyplot(fig1)

# --- Spread & Rolling Bands ---
fig2, ax2 = plt.subplots()
ax2.plot(df["ts"], df["spread_ols"], label="Spread (OLS)", color="orange")
ax2.axhline(df["rolling_mean"].iloc[-1], color="gray", linestyle="--", label="Rolling Mean")
ax2.fill_between(
    df["ts"],
    df["rolling_mean"] + 2 * df["rolling_std"],
    df["rolling_mean"] - 2 * df["rolling_std"],
    color="gray",
    alpha=0.2
)
ax2.set_title("Spread & Rolling Bands")
ax2.legend()
st.pyplot(fig2)

# --- Z-score Plot ---
fig3, ax3 = plt.subplots()
ax3.plot(df["ts"], df["zscore"], color="purple", label="Z-score")
ax3.axhline(2, color="red", linestyle="--", label="Upper Threshold")
ax3.axhline(-2, color="green", linestyle="--", label="Lower Threshold")
ax3.axhline(0, color="gray", linestyle=":")
ax3.set_title("Z-score Signal")
ax3.legend()
st.pyplot(fig3)

# --- Alerts ---
if abs(latest["zscore"]) > 2 and latest["adf_pvalue"] < 0.05:
    st.warning(f"⚠️ Mean Reversion Alert: z={latest['zscore']:.2f}, ADF p={latest['adf_pvalue']:.3f}")
elif abs(latest["zscore"]) > 2:
    st.info(f"ℹ️ High Deviation Alert: z={latest['zscore']:.2f}")
else:
    st.success("✅ Market Stable: No current alerts")

# --- Correlation Heatmap ---
corr = df[["BTCUSDT_price", "ETHUSDT_price"]].pct_change().corr()
fig4, ax4 = plt.subplots()
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax4)
ax4.set_title("Cross-Correlation Heatmap")
st.pyplot(fig4)

# --- Time-Series Aggregation (1-Minute) ---
df["minute"] = df["ts"].dt.floor("min")
agg = df.groupby("minute").agg({
    "BTCUSDT_price": "last",
    "ETHUSDT_price": "last",
    "spread_ols": "mean",
    "zscore": "mean",
    "rolling_std": "mean",
    "hedge_ols": "last",
    "hedge_kalman": "last",
    "adf_pvalue": "last"
}).reset_index()

st.subheader("📊 Time-Series Feature Summary (1-Minute Aggregation)")
st.dataframe(agg.tail(100))

# --- CSV Download ---
csv = agg.to_csv(index=False).encode()
st.download_button(
    "⬇️ Download Aggregated CSV",
    data=csv,
    file_name="time_series_summary.csv",
    mime="text/csv"
)

# --- Creative Analytics Overview ---
st.subheader("💡 Creative Analytics Overview")
st.write(f"Rolling Volatility (last 1hr): {df['spread_ols'].rolling(60).std().iloc[-1]:.4f}")
st.write(f"Kalman Hedge Drift (recent 30 ticks): {df['hedge_kalman'].tail(30).std():.6f}")
st.write(f"Correlation Stability Index: {corr.iloc[0, 1]:.3f}")
