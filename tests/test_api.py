import pytest
from starlette.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_predict_point_endpoint():
    payload = {
        "x": 2.0, 
        "y": 0.0, 
        "t": 5.0,
        "compute_vorticity": False
    }
    response = client.post("/predict/point", json=payload)
    if response.status_code == 503:
        assert "NOT READY" in response.json()["detail"]
    else:
        assert response.status_code == 200
        data = response.json()
        assert "u" in data and isinstance(data["u"], (int, float))
        assert "v" in data and isinstance(data["v"], (int, float))
        assert "p" in data and isinstance(data["p"], (int, float))

def test_predict_field_endpoint():
    payload = {
        "x_min": 1.0, 
        "x_max": 8.0,
        "y_min": -2.0, 
        "y_max": 2.0,
        "nx": 10, 
        "ny": 10, 
        "t": 5.0,
        "compute_vorticity": False
    }
    response = client.post("/predict/field", json=payload)
    if response.status_code == 503:
        assert "NOT READY" in response.json()["detail"]
    else:
        assert response.status_code == 200
        data = response.json()
        assert "u" in data and isinstance(data["u"], list)
        assert len(data["u"]) == 10
        assert len(data["u"][0]) == 10
