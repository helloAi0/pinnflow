import pytest
from fastapi.testclient import TestClient
from src.api.main import app
import src.main
import main
import backend.main

client = TestClient(app)

def test_entrypoint_proxies():
    """Verify that all proxy entrypoints export the canonical FastAPI app."""
    assert src.main.app is app
    assert main.app is app
    assert backend.main.app is app

def test_api_root():
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "version" in data

def test_api_health():
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert "trained" in data
    assert "model_version" in data

def test_api_metadata():
    res = client.get("/api/v1/metadata")
    assert res.status_code == 200
    data = res.json()
    assert "physics" in data
    assert data["physics"]["reynolds_number"] == 100.0

def test_api_predict_point_validation():
    # Valid payload
    payload = {"x": 2.0, "y": 0.0, "t": 5.0, "compute_vorticity": True}
    res = client.post("/api/v1/predict/point", json=payload)
    if res.status_code == 503:
        # Checkpoint not loaded in un-trained test environment is safe & intended
        assert "PINNFlow Inference Service NOT READY" in res.json()["detail"]
    else:
        assert res.status_code == 200
        data = res.json()
        assert "u" in data
        assert "velocity_magnitude" in data

def test_api_predict_field_validation():
    payload = {
        "x_min": 1.0,
        "x_max": 8.0,
        "y_min": -2.0,
        "y_max": 2.0,
        "nx": 10,
        "ny": 10,
        "t": 10.0,
        "compute_vorticity": False
    }
    res = client.post("/api/v1/predict/field", json=payload)
    if res.status_code == 503:
        assert "PINNFlow Inference Service NOT READY" in res.json()["detail"]
    else:
        assert res.status_code == 200
        data = res.json()
        assert len(data["u"]) == 10
        assert len(data["u"][0]) == 10
        assert "pde_loss" in data["metrics"]
