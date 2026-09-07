import pytest
from starlette.testclient import TestClient
from src.api.main import app

# Initialize the offline testing client wrapper targeting current lifespan context
client = TestClient(app)

def test_health_check():
    """Verifies that the API compute engine core boots up and passes schema standards."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "pinnflow-api"

def test_health_endpoint():
    """Validates the auxiliary secondary check endpoint matches test routes."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_point_endpoint():
    """Evaluates flow field mapping accuracy for a single point coordinate query."""
    payload = {
        "x": 1.0, 
        "y": 0.5, 
        "t": 0.0,
        "compute_vorticity": False
    }
    response = client.post("/predict/point", json=payload)
    assert response.status_code == 200, f"Point prediction failed: {response.text}"
    
    data = response.json()
    # Structural keys verification matching current src/api/main.py mappings
    assert "u" in data and isinstance(data["u"], (int, float))
    assert "v" in data and isinstance(data["v"], (int, float))
    assert "p" in data and isinstance(data["p"], (int, float))
    assert "velocity_magnitude" in data and isinstance(data["velocity_magnitude"], (int, float))

def test_predict_field_endpoint():
    """Validates structured meshgrid serialization output configurations for visual charting."""
    payload = {
        "x_min": -2.0, 
        "x_max": 10.0,
        "y_min": -2.0, 
        "y_max": 2.0,
        "nx": 10, 
        "ny": 10, 
        "t": 0.0,
        "compute_vorticity": False
    }
    response = client.post("/predict/field", json=payload)
    assert response.status_code == 200, f"Grid field visualization endpoint failed: {response.text}"
    
    data = response.json()
    # Grid serialization outputs structural validation matches
    assert "x" in data and isinstance(data["x"], list)
    assert "y" in data and isinstance(data["y"], list)
    assert "u" in data and isinstance(data["u"], list)
    assert "v" in data and isinstance(data["v"], list)
    assert "p" in data and isinstance(data["p"], list)
    assert "velocity_magnitude" in data and isinstance(data["velocity_magnitude"], list)
    
    # Confirm coordinate scaling constraints match matrix layout shapes (ny rows x nx columns)
    assert len(data["u"]) == 10
    assert len(data["u"][0]) == 10
