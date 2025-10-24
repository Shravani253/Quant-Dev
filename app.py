import subprocess
import sys
import time

def run_backend():
    subprocess.Popen([sys.executable, "backend.py"])
    print("✅ Backend started at http://127.0.0.1:8000")

def run_frontend():
    subprocess.Popen(["streamlit", "run", "frontend.py"])
    print("✅ Frontend started at http://localhost:8501")

if __name__ == "__main__":
    print("🚀 Launching Quant Analytics App...")
    run_backend()
    time.sleep(2)
    run_frontend()
    print("\n✅ Both backend and frontend are running.\n")
