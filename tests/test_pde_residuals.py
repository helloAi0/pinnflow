import torch
import numpy as np
from src.physics.navier_stokes import NavierStokes2D

def test_taylor_green_vortex_residuals():
    nu = 0.01
    Re = 1.0 / nu
    pde_solver = NavierStokes2D(reynolds_number=Re)

    # Grid domain
    x = torch.linspace(0, 2 * np.pi, 20)
    y = torch.linspace(0, 2 * np.pi, 20)
    t = torch.linspace(0, 1, 5)
    
    X, Y, T = torch.meshgrid(x, y, t, indexing='ij')
    coords = torch.stack([X.flatten(), Y.flatten(), T.flatten()], dim=1).requires_grad_(True)

    x_c, y_c, t_c = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]

    # Taylor-Green Exact Solutions
    decay = torch.exp(-2.0 * nu * t_c)
    u_exact = torch.sin(x_c) * torch.cos(y_c) * decay
    v_exact = -torch.cos(x_c) * torch.sin(y_c) * decay
    p_exact = 0.25 * (torch.cos(2.0 * x_c) + torch.cos(2.0 * y_c)) * (decay ** 2)

    outputs = torch.cat([u_exact, v_exact, p_exact], dim=1)

    residuals = pde_solver.compute_residuals(coords, outputs)

    max_c = torch.max(torch.abs(residuals["continuity"])).item()
    max_u = torch.max(torch.abs(residuals["momentum_x"])).item()
    max_v = torch.max(torch.abs(residuals["momentum_y"])).item()

    print(f"Taylor-Green Manufactured Residual Verification:")
    print(f"  Max Continuity Residual: {max_c:.8e}")
    print(f"  Max Momentum-X Residual: {max_u:.8e}")
    print(f"  Max Momentum-Y Residual: {max_v:.8e}")

    assert max_c < 1e-5, "Continuity residual test failed."
    assert max_u < 1e-5, "Momentum-X residual test failed."
    assert max_v < 1e-5, "Momentum-Y residual test failed."
    print("ALL PDE RESIDUAL TESTS PASSED CLEANLY.")

if __name__ == "__main__":
    test_taylor_green_vortex_residuals()