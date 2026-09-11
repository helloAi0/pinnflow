import torch
import numpy as np
import pytest
from src.physics.derivatives import compute_first_derivatives, compute_second_derivatives
from src.losses.pde_loss import PDELoss
from src.config.physics_config import PhysicsConfig

def test_taylor_green_exact_navier_stokes_residuals():
    """
    Taylor-Green 2D Vortex is an exact analytical solution of incompressible Navier-Stokes:
      u(x, y, t) = sin(x) * cos(y) * exp(-2*nu*t)
      v(x, y, t) = -cos(x) * sin(y) * exp(-2*nu*t)
      p(x, y, t) = 0.25 * (cos(2x) + cos(2y)) * exp(-4*nu*t)
      
    This test verifies that the automatic differentiation engine computes EXACT ZERO residuals:
      R_c = du/dx + dv/dy = 0
      R_u = du/dt + u*du/dx + v*du/dy + dp/dx - nu*(d^2u/dx^2 + d^2u/dy^2) = 0
      R_v = dv/dt + u*dv/dx + v*dv/dy + dp/dy - nu*(d^2v/dx^2 + d^2v/dy^2) = 0
    """
    nu = 0.01
    Re = 1.0 / nu
    cfg = PhysicsConfig(reynolds_number=Re, density=1.0)
    pde_engine = PDELoss(physics_config=cfg)

    # Sample domain [0, 2*pi] x [0, 2*pi] x [0, 1]
    x = torch.linspace(0, 2 * np.pi, 25)
    y = torch.linspace(0, 2 * np.pi, 25)
    t = torch.linspace(0, 1.0, 5)

    X, Y, T = torch.meshgrid(x, y, t, indexing="ij")
    coords = torch.stack([X.flatten(), Y.flatten(), T.flatten()], dim=1).requires_grad_(True)

    x_c = coords[:, 0:1]
    y_c = coords[:, 1:2]
    t_c = coords[:, 2:3]

    decay = torch.exp(-2.0 * nu * t_c)
    u_exact = torch.sin(x_c) * torch.cos(y_c) * decay
    v_exact = -torch.cos(x_c) * torch.sin(y_c) * decay
    p_exact = 0.25 * (torch.cos(2.0 * x_c) + torch.cos(2.0 * y_c)) * (decay ** 2)

    outputs = torch.cat([u_exact, v_exact, p_exact], dim=1)

    residuals = pde_engine.compute_residuals(coords, outputs)

    max_c = float(torch.max(torch.abs(residuals["continuity"])).item())
    max_u = float(torch.max(torch.abs(residuals["momentum_x"])).item())
    max_v = float(torch.max(torch.abs(residuals["momentum_y"])).item())

    # Verify that numerical residuals against exact solution are below floating-point tolerance
    assert max_c < 1e-5, f"Continuity residual exceeded tolerance: {max_c:.8e}"
    assert max_u < 1e-5, f"Momentum-X residual exceeded tolerance: {max_u:.8e}"
    assert max_v < 1e-5, f"Momentum-Y residual exceeded tolerance: {max_v:.8e}"

def test_analytical_laplacian_and_vorticity():
    """
    Validates autograd calculation of Laplacian and Vorticity against exact known forms:
      f(x, y) = sin(pi*x) * cos(pi*y)
      nabla^2 f = -2*pi^2 * sin(pi*x) * cos(pi*y)
    """
    x = torch.linspace(-1, 1, 30)
    y = torch.linspace(-1, 1, 30)
    t = torch.zeros_like(x)
    X, Y, T = torch.meshgrid(x, y, t, indexing="ij")
    coords = torch.stack([X.flatten(), Y.flatten(), T.flatten()], dim=1).requires_grad_(True)

    x_c, y_c = coords[:, 0:1], coords[:, 1:2]
    f = torch.sin(np.pi * x_c) * torch.cos(np.pi * y_c)
    outputs = torch.cat([f, f, f], dim=1)

    u_grads, v_grads, _ = compute_first_derivatives(outputs, coords)
    u_2nd, _ = compute_second_derivatives(u_grads, v_grads, coords)

    laplacian_exact = -2.0 * (np.pi ** 2) * torch.sin(np.pi * x_c) * torch.cos(np.pi * y_c)
    laplacian_err = float(torch.max(torch.abs(u_2nd["laplacian"] - laplacian_exact)).item())

    assert laplacian_err < 1e-5, f"Laplacian autograd error too large: {laplacian_err:.8e}"

def test_divergence_free_analytical_field():
    """
    Streamfunction psi = sin(x) * sin(y) generates velocity:
      u = dpsi/dy = sin(x) * cos(y)
      v = -dpsi/dx = -cos(x) * sin(y)
    Divergence div(u) = du/dx + dv/dy = cos(x)*cos(y) - cos(x)*cos(y) = 0 identically.
    """
    coords = torch.randn(100, 3, requires_grad=True)
    x_c, y_c = coords[:, 0:1], coords[:, 1:2]
    u = torch.sin(x_c) * torch.cos(y_c)
    v = -torch.cos(x_c) * torch.sin(y_c)
    p = torch.zeros_like(u)

    outputs = torch.cat([u, v, p], dim=1)
    u_grads, v_grads, _ = compute_first_derivatives(outputs, coords)
    div_u = u_grads["x"] + v_grads["y"]

    max_div = float(torch.max(torch.abs(div_u)).item())
    assert max_div < 1e-6, f"Divergence-free assertion failed: {max_div:.8e}"
