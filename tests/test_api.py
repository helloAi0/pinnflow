import pytest
from starlette.testclient import TestClient
from src.api.main import app

# Initialize the offline testing client wrapper
client = TestClient(app)

def test_health_check():
    """Verifies that the API compute engine core boots up correctly."""
    # Adjust route string to match your endpoint path (e.g., "/" or "/health")
    response = client.get("/")  
    assert response.status_code == 200


def test_predict_coordinates():
    """Evaluates the model inference pipeline against physical points behind the cylinder."""
    payload = {
        "points": [
            {"x": 1.0, "y": 0.0, "t": 5.0},
            {"x": 1.5, "y": 0.2, "t": 5.0},
            {"x": 2.0, "y": -0.2, "t": 5.0}
        ]
    }
    
    # FIX: Use internal client instead of live 'requests.post' to prevent CI failures
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200, f"Prediction endpoint failed with: {response.text}"
    
    data = response.json()
    assert "predictions" in data, "Response JSON schema missing 'predictions' root key"
    assert len(data["predictions"]) == 3, f"Expected 3 outputs, received {len(data['predictions'])}"
    
    # Verify the structure of the mathematical field arrays
    for pred in data["predictions"]:
        assert "u" in pred and isinstance(pred["u"], (int, float))
        assert "v" in pred and isinstance(pred["v"], (int, float))
        assert "p" in pred and isinstance(pred["p"], (int, float))
