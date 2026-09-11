import torch
import pytest
from src.losses.data_loss import DataLoss
from src.losses.pde_loss import PDELoss
from src.losses.boundary_condition_loss import BoundaryConditionLoss
from src.losses.pressure_gauge_loss import PressureGaugeLoss
from src.losses.weighting import HomoscedasticAdaptiveWeighting, DynamicGradientWeighting, UnifiedPINNLoss
from src.config.physics_config import DEFAULT_PHYSICS_CONFIG
from src.models.mlp import NavierStokesMLP

def test_data_loss_zero_when_identical():
    fn = DataLoss()
    preds = torch.tensor([[1.0, 2.0, 0.5], [3.0, 4.0, -0.5]], dtype=torch.float32)
    targets = preds.clone()
    loss, components = fn(preds, targets)
    assert float(loss.item()) == pytest.approx(0.0, abs=1e-7)

def test_pde_loss_computation():
    pde_fn = PDELoss(physics_config=DEFAULT_PHYSICS_CONFIG)
    model = NavierStokesMLP(in_dim=3, out_dim=3, hidden_layers=2, hidden_dim=16)
    coords = torch.randn(20, 3, requires_grad=True)
    outputs = model(coords)
    
    loss, loss_dict, residuals = pde_fn(coords, outputs)
    assert "loss_continuity" in loss_dict
    assert "loss_momentum_x" in loss_dict
    assert "loss_momentum_y" in loss_dict
    assert residuals["continuity"].shape == (20, 1)

def test_homoscedastic_adaptive_weighting():
    module = HomoscedasticAdaptiveWeighting(num_losses=3)
    losses = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(0.5)]
    total, weights = module(losses)
    assert total.requires_grad
    assert len(weights) == 6  # 3 weights + 3 log_vars

def test_unified_pinn_loss_flow():
    loss_module = UnifiedPINNLoss(physics_config=DEFAULT_PHYSICS_CONFIG, strategy="static")
    model = NavierStokesMLP(in_dim=3, out_dim=3, hidden_layers=2, hidden_dim=16)
    
    data_c = torch.randn(10, 3)
    data_t = torch.randn(10, 3)
    pde_c = torch.randn(20, 3, requires_grad=True)
    
    total_loss, metrics, res = loss_module(model, data_c, data_t, pde_c)
    assert total_loss > 0.0
    assert "loss_data" in metrics
    assert "loss_pde" in metrics
