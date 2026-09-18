"""
PINNFlow Backend Entrypoint Proxy.
Re-exports the FastAPI application from `src.api.main` for `backend.main:app` entrypoints.
"""
from src.api.main import app, get_health, predict_point, predict_field

__all__ = ["app", "get_health", "predict_point", "predict_field"]
