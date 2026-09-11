import pytest
import torch
from starlette.testclient import TestClient

from src.api.main import app
from src.models.mlp import NavierStokesMLP
from src.data.observation_dataset import DataPipelineManager
from src.config.physics_config import DEFAULT_PHYSICS_CONFIG

client = TestClient(app)

def test_model_forward_pass():
    model = NavierStokesMLP(in_dim=3, out_dim=3, hidden_layers=4, hidden_dim=64)
    x = torch.randn(128, 3)
    out = model(x)
    assert out.shape == (128, 3)
    assert not torch.isnan(out).any()

def test_data_pipeline_dimensions():
    data_mgr = DataPipelineManager(seed=42)
    assert data_mgr.full_coords.shape[0] == 1000000
    assert data_mgr.full_coords.shape[1] == 3
    assert data_mgr.full_fields.shape[1] == 3

def test_physics_domain_bounds():
    cfg = DEFAULT_PHYSICS_CONFIG
    assert cfg.x_min == 1.0
    assert cfg.x_max == 8.0
    assert cfg.y_min == -2.0
    assert cfg.y_max == 2.0
    assert cfg.cylinder_radius == 0.5

def test_api_root_endpoint():
    res = client.get("/")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"
