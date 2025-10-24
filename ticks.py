import websocket
import json
import pandas as pd
import time
import requests
import threading

BINANCE_URL = "wss://stream.binance.com:9443/stream?streams=btcusdt@trade/ethusdt@trade"
BACKEND_URL = "http://127.0.0.1:8000/ingest"
CSV_FILE = "ticks_live.csv"

buffer = []
lock = threading.Lock()

def on_message(ws, message):
    msg = json.loads(message)
    data = msg["data"]
    tick = {
        "T": data["T"],  # timestamp
        "s": data["s"],  # symbol
        "p": data["p"],  # price
        "q": data["q"]   # quantity
    }
    with lock:
        buffer.append(tick)

def save_loop():
    while True:
        time.sleep(10)
        with lock:
            if len(buffer) == 0:
                continue
            df = pd.DataFrame(buffer)
            buffer.clear()
        df.to_csv(CSV_FILE, index=False)
        requests.post(BACKEND_URL, files={"file": open(CSV_FILE, "rb")})
        print("📤 Sent to backend | Rows:", len(df))

def run_socket():
    ws = websocket.WebSocketApp(BINANCE_URL, on_message=on_message)
    ws.run_forever()

if __name__ == "__main__":
    threading.Thread(target=save_loop, daemon=True).start()
    run_socket()
