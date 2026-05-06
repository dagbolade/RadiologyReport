"""
Local development runner for the Radiology Report Generator demo.

Usage:
    cd demo
    python run_local.py

This starts the FastAPI server on http://localhost:8000
The frontend is served automatically at the root URL.
Model files are loaded from ../Radiography/
"""

import os
import sys

# Set model base directory to parent of demo/
os.environ["MODEL_BASE_DIR"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  Radiology Report Generator — Local Development")
    print("  Open: http://localhost:8000")
    print("=" * 60)
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
