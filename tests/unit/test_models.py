import torch
import pytest
from src.models.factory import create_model, count_parameters
from src.models.mlp import NavierStokesMLP
from src.models.fourier_network import FourierFeaturePINN
from src.models.adaptive_network import AdaptiveActivationPINN
from src.config.physics_config import PhysicsConfig

def test_canonical_mlp_forward_and_gradient():
    model = NavierStokesMLP(in_dim=3, out_dim=3, hidden_layers=4, hidden_dim=64, activation_type="tanh")
    x = torch.randn(32, 3, requires_grad=True)
    out = model(x)
    assert out.shape == (32, 3)
    assert not torch.isnan(out).any()

    # Verify smooth backward gradient
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()

def test_fourier_feature_pinn_embedding():
    model = FourierFeaturePINN(in_dim=3, out_dim=3, fourier_features=32, fourier_sigma=1.0, hidden_layers=3, hidden_dim=64)
    x = torch.randn(16, 3)
    out = model(x)
    assert out.shape == (16, 3)
    assert not torch.isnan(out).any()

def test_hard_constrained_pinn_no_slip():
    cfg = PhysicsConfig(cylinder_center=(0.0, 0.0), cylinder_radius=0.5)
    model = FourierFeaturePINN(
        in_dim=3, out_dim=3, fourier_features=32,
        apply_hard_constraints=True, physics_config=cfg
    )
    
    # Points exactly on the cylinder perimeter: r = 0.5
    theta = torch.linspace(0, 2 * 3.14159, 20)
    x_cyl = 0.5 * torch.cos(theta)
    y_cyl = 0.5 * torch.sin(theta)
    t_cyl = torch.zeros_like(x_cyl)
    cyl_coords = torch.stack([x_cyl, y_cyl, t_cyl], dim=1)

    out = model(cyl_coords)
    u_cyl = out[:, 0]
    v_cyl = out[:, 1]
    
    # Hard constraint guarantees u=0, v=0 on the boundary to float precision
    assert torch.allclose(u_cyl, torch.zeros_like(u_cyl), atol=1e-6)
    assert torch.allclose(v_cyl, torch.zeros_like(v_cyl), atol=1e-6)

def test_adaptive_activation_pinn():
    model = AdaptiveActivationPINN(in_dim=3, out_dim=3, hidden_layers=3, hidden_dim=32)
    x = torch.randn(10, 3)
    out = model(x)
    assert out.shape == (10, 3)
    assert len(model.a_params) == 3

def test_factory_all_architectures():
    archs = ["standard_mlp", "fourier_pinn", "hard_constrained_pinn", "adaptive_activation_pinn"]
    for arch in archs:
        cfg = {"architecture": arch, "hidden_layers": 2, "hidden_dim": 16}
        m, meta = create_model(cfg)
        assert meta["architecture"] == arch
        assert meta["parameter_count"] > 0
