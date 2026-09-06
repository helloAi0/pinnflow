import pytest
import torch
import numpy as np
from fastapi.testclient import TestClient

from src.models.mlp import NavierStokesMLP
from src.data.observation_dataset import DataPipelineManager
from src.deployment.api import app

# 1. Model Architecture Tests
def test_model_forward_pass():
    """Verify that the PINN model accepts [N, 3] inputs and returns [N, 3] outputs."""
    model = NavierStokesMLP()
    model.eval()
    
    batch_size = 128
    dummy_input = torch.randn(batch_size, 3)  # [x, y, t]
    
    with torch.no_grad():
        output = model(dummy_input)
        
    assert output.shape == (batch_size, 3), f"Expected shape ({batch_size}, 3), got {output.shape}"
    assert not torch.isnan(output).any(), "Model produced NaN values in forward pass."

# 2. Data Pipeline Tests
def test_data_pipeline_shapes():
    """Verify that DataPipelineManager creates valid PyTorch dataset splits."""
    manager = DataPipelineManager(seed=42)
    datasets = manager.prepare_experiment_data(train_budget=500, noise_level=0.0)
    
    assert "train" in datasets and "test" in datasets, "Dataset splits missing from manager output."
    
    # Inspect a single sample
    x_sample, y_sample = datasets["train"][0]
    assert x_sample.shape == (3,), f"Input sample shape mismatch: expected (3,), got {x_sample.shape}"
    assert y_sample.shape == (3,), f"Target sample shape mismatch: expected (3,), got {y_sample.shape}"

# 3. Automatic Differentiation Engine Tests
def test_autograd_gradient_computation():
    """Verify that autograd computes non-zero spatial derivatives through the model."""
    model = NavierStokesMLP()
    
    coords = torch.randn(10, 3, requires_grad=True)
    preds = model(coords)
    u = preds[:, 0:1]
    
    # Compute first-order derivative du/dx
    grads = torch.autograd.grad(u, coords, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:, 0:1]
    
    assert u_x.shape == (10, 1), "Derivative tensor shape mismatch."
    assert torch.abs(u_x).sum().item() > 0.0, "Autograd produced zero gradients for model parameters."

# 4. Deployment API Tests
def test_fastapi_predict_endpoint():
    """Verify end-to-end HTTP payload processing and response structure."""
    with TestClient(app) as client:
        payload = {
            "points": [
                {"x": 1.0, "y": 0.0, "t": 5.0},
                {"x": 2.0, "y": 0.5, "t": 2.5}
            ]
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 200, f"API returned error code {response.status_code}: {response.text}"
        
        data = response.json()
        assert "predictions" in data, "Response missing 'predictions' key."
        assert len(data["predictions"]) == 2, f"Expected 2 predictions, got {len(data['predictions'])}"
        
        first_pred = data["predictions"][0]
        assert all(k in first_pred for k in ("u", "v", "p")), "Prediction payload missing required physical keys (u, v, p)."