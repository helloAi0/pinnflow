"""
PINNFlow Main Entrypoint Proxy.
Re-exports the FastAPI application and core handlers from `src.api.main`
to ensure standard `src.main:app` entrypoints boot seamlessly without import errors.
"""
from src.api.main import app, get_health, predict_point, predict_field

__all__ = ["app", "get_health", "predict_point", "predict_field"]

if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("src.main:app", host=host, port=port, reload=True)
