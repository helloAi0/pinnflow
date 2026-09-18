"""
PINNFlow Backend App Alias.
"""
from backend.main import app, get_health, predict_point, predict_field

__all__ = ["app", "get_health", "predict_point", "predict_field"]
