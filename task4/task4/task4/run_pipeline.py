import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

def run_pipeline():
    print("="*60)
    print("🚀 STARTING TASK 4 FULL PIPELINE 🚀")
    print("="*60)

    try:
        # Step 1: Run Model Selection (18 Models Benchmark)
        print("\n[1/4] Running Model Selection & Hyperparameter Tuning...")
        subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "best_model_selection.py")],
            check=True,
            cwd=SCRIPT_DIR,
        )

        # Step 2: Run Error Analysis
        print("\n[2/4] Running Error Analysis & Confusion Matrix Generation...")
        subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "error_analysis.py")],
            check=True,
            cwd=SCRIPT_DIR,
        )

        # Step 3: Start the Flask API in the background
        print("\n[3/4] Starting Flask Backend API (Background Process)...")
        api_process = subprocess.Popen(
            [sys.executable, str(SCRIPT_DIR / "api.py")],
            cwd=SCRIPT_DIR,
        )
        
        # Give the Flask API a few seconds to boot up cleanly
        print("      Waiting 3 seconds for API to initialize...")
        time.sleep(3)

        # Step 4: Run Streamlit Frontend
        print("\n[4/4] Launching Streamlit Dashboard Frontend...")
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(SCRIPT_DIR / "frontend.py")],
            cwd=SCRIPT_DIR,
        )

    except subprocess.CalledProcessError as e:
        print(f"\n[!] Pipeline failed during execution. Error: {e}")
    except KeyboardInterrupt:
        print("\n[!] Pipeline manually interrupted by user.")
    finally:
        # Clean up the background API process when the Streamlit app is closed
        if 'api_process' in locals():
            print("\n[CLEANUP] Shutting down Flask API...")
            api_process.terminate()
            api_process.wait()
            print("Pipeline successfully closed.")

if __name__ == "__main__":
    run_pipeline()