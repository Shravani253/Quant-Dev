# Quant-Dev
Quant Developer Assignment – Real-time Binance analytics with FastAPI &amp; Streamlit
Quant Developer Assignment — Real-Time Analytics Dashboard
Author: Shravani Vanalkar

Tech Stack: FastAPI · Streamlit · Python · Pandas · WebSockets · Kalman Filter · Statsmodels

🚀 Project Overview

This project implements a real-time quantitative analytics dashboard that streams live Binance market data (BTCUSDT and ETHUSDT), processes it through a FastAPI backend, and visualizes insights using a Streamlit frontend.

It performs dynamic hedge estimation, z-score analysis, mean reversion backtests, and cross-correlation heatmaps, with automatic alerts when statistical thresholds are triggered.

Everything runs locally using a single command:

python app.py

⚙️ Architecture
Binance API → ticks.py → backend.py → analytics_output.csv → frontend.py (Streamlit)


Data Flow

ticks.py connects to Binance WebSocket and fetches BTCUSDT & ETHUSDT trade data every 10s.

Data is saved to ticks_live.csv and sent to the backend via /ingest.

backend.py computes analytics (hedge ratios, spreads, z-scores, etc.) and stores them in analytics_output.csv.

frontend.py displays charts, alerts, and tables in real time through /analytics_data.

🧩 Features

Real-time WebSocket tick data ingestion

OLS, Huber, Theil–Sen, and Kalman hedge ratio estimation

Mean reversion & high deviation alerts

ADF stationarity test

Price and spread visualization with rolling ±2σ bands

Z-score chart with signal thresholds

Cross-correlation heatmap

Aggregated time-series feature summary (1-min resample)

CSV export of analytics data

Single-command local run

📊 Analytics Implemented
Category	Description
Hedge Estimation	OLS, Huber, Theil–Sen, Kalman Filter (dynamic β)
Stationarity Test	ADF p-value of spread
Mean Reversion Alerts	Trigger when z > 2 (entry) or z < 0 (exit)
Aggregation	Resampled 1-min time-series
Correlation	BTC-ETH cross-correlation heatmap
Volatility	Rolling 1hr spread volatility
Export	Downloadable CSV summary
⚡️ Setup & Usage
Prerequisites

Install Python 3.9+

Step 1 — Clone the Repo
git clone https://github.com/Shravani253/Quant-Dev.git
cd Quant-Dev

Step 2 — Create Virtual Environment
python -m venv .venv
.venv\Scripts\activate

Step 3 — Install Dependencies
pip install -r requirements.txt

Step 4 — Run the Application
python app.py


Backend: http://127.0.0.1:8000

Frontend: http://localhost:8501

🧪 Testing

Wait 10–15 seconds for live tick data.

Visit:

http://127.0.0.1:8000/docs for backend docs

http://127.0.0.1:8000/analytics_data for JSON output

http://localhost:8501 for Streamlit dashboard

You’ll see live BTC/ETH charts, spreads, and alerts.

🧱 Directory Structure
quant-dev-assignment/
├─ app.py
├─ backend.py
├─ ticks.py
├─ frontend.py
├─ analytics_output.csv
├─ ticks_live.csv
├─ requirements.txt
├─ README.md
├─ CHATGPT_USAGE.md
├─ architecture.drawio
└─ architecture.png

🧮 Dependencies
fastapi
uvicorn
pandas
numpy
requests
websocket-client
streamlit
statsmodels
scikit-learn
matplotlib
seaborn
python-multipart

💬 ChatGPT Usage Transparency

This project used ChatGPT to assist with:

Aligning backend and frontend APIs

Debugging ingestion and analytics flow

Writing documentation and README structure

All code was tested and verified locally by Shravani Vanalkar.
Prompts used are listed in CHATGPT_USAGE.md.

🧭 Architecture Diagram

Include architecture.drawio and export it as architecture.png using draw.io
.

🧾 License

This project is part of the Quant Developer Assignment and is for evaluation purposes only.

✅ Run everything with:

python app.py


Then open:

Backend Docs → http://127.0.0.1:8000/docs

Dashboard → http://localhost:8501

You’ll see live BTC/ETH analytics, alerts, and downloadable summarie
