"""
PINNFlow Root Entrypoint Proxy.
Re-exports the FastAPI application from `src.api.main` for standard `main:app` entrypoints.
"""
from src.api.main import app, get_health, predict_point, predict_field

__all__ = ["app", "get_health", "predict_point", "predict_field"]

if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port, reload=True)
