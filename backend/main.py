"""
PINNFlow Backend Entrypoint.
Ensures repository root is in sys.path and exposes FastAPI app.
"""
import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.api.main import app, get_health, predict_point, predict_field

__all__ = ["app", "get_health", "predict_point", "predict_field"]

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("backend.main:app", host=host, port=port, reload=True)
