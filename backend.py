from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from sklearn.linear_model import HuberRegressor, TheilSenRegressor
import os

app = FastAPI(title="Quant Backend - Aligned with ticks_live collector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

TICKS_FILE = "ticks_live.csv"
ANALYTICS_FILE = "analytics_output.csv"


# ---------------- Kalman Filter Hedge Estimation ---------------- #
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


# ---------------- Core Analytics Computation ---------------- #
def compute_analytics(ticks_path):
    df = pd.read_csv(ticks_path)

    # ✅ Convert numeric columns properly
    df["p"] = pd.to_numeric(df["p"], errors="coerce")
    df["q"] = pd.to_numeric(df["q"], errors="coerce")

    if {"T", "s", "p", "q"}.issubset(df.columns):
        df = df.rename(columns={"T": "ts", "s": "symbol", "p": "price", "q": "size"})
    elif {"ts", "s", "p", "q"}.issubset(df.columns):
        df = df.rename(columns={"s": "symbol", "p": "price", "q": "size"})

    if "ts" in df.columns and df["ts"].dtype != "datetime64[ns]":
        if df["ts"].dtype.kind in {"i", "u", "f"}:
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        else:
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)

    df = df.dropna(subset=["ts"]).sort_values("ts")

    if {"symbol", "price"}.issubset(df.columns):
        price_df = df.pivot_table(index="ts", columns="symbol", values="price", aggfunc="last").sort_index().ffill()
        if {"BTCUSDT", "ETHUSDT"}.issubset(price_df.columns):
            price_df = price_df[["BTCUSDT", "ETHUSDT"]].rename(
                columns={"BTCUSDT": "BTCUSDT_price", "ETHUSDT": "ETHUSDT_price"}
            )
        else:
            return None
    elif {"BTCUSDT_price", "ETHUSDT_price"}.issubset(df.columns):
        price_df = df.set_index("ts").resample("1s").last().ffill()[["BTCUSDT_price", "ETHUSDT_price"]]
    else:
        return None

    price_df = price_df.dropna()
    if price_df.shape[0] < 3:
        return None

    price_df = price_df.reset_index()
    y = price_df["BTCUSDT_price"].values
    x = price_df["ETHUSDT_price"].values

    # Hedge estimations
    hedge_ols = float(sm.OLS(y, sm.add_constant(x)).fit().params[1])
    hedge_huber = float(HuberRegressor().fit(x.reshape(-1, 1), y).coef_[0])
    hedge_theil = float(TheilSenRegressor().fit(x.reshape(-1, 1), y).coef_[0])
    kalman_betas = kalman_dynamic_hedge(y, x)
    hedge_kalman_latest = float(kalman_betas[-1]) if len(kalman_betas) else np.nan

    price_df["hedge_ols"] = hedge_ols
    price_df["hedge_huber"] = hedge_huber
    price_df["hedge_theil"] = hedge_theil
    price_df["hedge_kalman"] = list(kalman_betas)

    # Spread & z-score
    price_df["spread_ols"] = price_df["BTCUSDT_price"] - hedge_ols * price_df["ETHUSDT_price"]
    price_df["rolling_mean"] = price_df["spread_ols"].rolling(window=60, min_periods=1).mean()
    price_df["rolling_std"] = price_df["spread_ols"].rolling(window=60, min_periods=1).std().fillna(0.0)
    price_df["zscore"] = (price_df["spread_ols"] - price_df["rolling_mean"]) / price_df["rolling_std"].replace(0, np.nan)
    price_df["adf_pvalue"] = adfuller(price_df["spread_ols"].dropna())[1]

    price_df.to_csv(ANALYTICS_FILE, index=False)
    return price_df


# ---------------- API Routes ---------------- #
@app.get("/")
def home():
    return {
        "message": "✅ Quant Backend is running successfully!",
        "endpoints": {
            "/docs": "Interactive API documentation",
            "/ingest": "POST tick data here",
            "/analytics_data": "Get computed analytics data",
            "/latest": "Get latest analytics record"
        }
    }

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    # ✅ Ensure correct file reading
    content = await file.read()
    with open(TICKS_FILE, "wb") as f:
        f.write(content)

    df = compute_analytics(TICKS_FILE)
    if df is None:
        return JSONResponse({"error": "Insufficient or malformed tick data"}, status_code=400)

    print(f"✅ File processed successfully | Rows: {len(df)}")
    return {"status": "success", "rows": len(df)}


@app.get("/latest")
def latest():
    if not os.path.exists(ANALYTICS_FILE):
        return JSONResponse({"error": "Analytics file missing"}, status_code=404)
    df = pd.read_csv(ANALYTICS_FILE)
    if df.empty:
        return JSONResponse({"error": "Analytics file empty"}, status_code=404)
    return df.tail(1).to_dict(orient="records")[0]


@app.get("/analytics_data")
def analytics_data():
    import os

    # Check file existence
    if not os.path.exists(ANALYTICS_FILE):
        print("❌ Analytics file not found.")
        return {"error": "Analytics file missing."}

    # Load data
    df = pd.read_csv(ANALYTICS_FILE)
    if df.empty:
        print("❌ Analytics file is empty.")
        return {"error": "No analytics data available."}

    # Clean problematic values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)  # convert NaNs to 0s so JSON can encode them
    df = df.astype(object)

    print(f"✅ Analytics data ready. Rows: {len(df)}, Columns: {list(df.columns)}")

    # Return safely encoded JSON
    try:
        result = df.to_dict(orient="records")
        print("✅ JSON conversion successful.")
        return result
    except Exception as e:
        print("❌ JSON conversion failed:", e)
        return {"error": "Data conversion issue. Check backend logs."}



if __name__ == "__main__":
    import uvicorn
    print("🚀 Registered Routes:", [r.path for r in app.routes])
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)
