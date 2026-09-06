import torch
import numpy as np
from src.physics.derivatives import compute_first_derivatives, compute_second_derivatives

def test_autograd_analytical_accuracy():
    # Construct test coordinates
    x = torch.linspace(0, 1, 10, requires_grad=True)
    y = torch.linspace(0, 1, 10, requires_grad=True)
    t = torch.linspace(0, 1, 10, requires_grad=True)
    
    X, Y, T = torch.meshgrid(x, y, t, indexing='ij')
    coords = torch.stack([X.flatten(), Y.flatten(), T.flatten()], dim=1).requires_grad_(True)

    # Exact function: f = sin(pi*x) * cos(pi*y) * exp(-t)
    x_c, y_c, t_c = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]
    f = torch.sin(np.pi * x_c) * torch.cos(np.pi * y_c) * torch.exp(-t_c)
    
    # Pass dummy outputs matching [u, v, p] signature
    outputs = torch.cat([f, f, f], dim=1)

    u_grads, v_grads, _ = compute_first_derivatives(outputs, coords)
    u_2nd, _ = compute_second_derivatives(u_grads, v_grads, coords)

    # Exact analytical derivatives
    df_dx_exact = np.pi * torch.cos(np.pi * x_c) * torch.cos(np.pi * y_c) * torch.exp(-t_c)
    laplacian_exact = -2.0 * (np.pi ** 2) * torch.sin(np.pi * x_c) * torch.cos(np.pi * y_c) * torch.exp(-t_c)

    # Maximum absolute discrepancies
    err_df_dx = torch.max(torch.abs(u_grads["x"] - df_dx_exact)).item()
    err_laplacian = torch.max(torch.abs(u_2nd["laplacian"] - laplacian_exact)).item()

    print(f"Analytical Test Verification Results:")
    print(f"  First derivative max error (df/dx):     {err_df_dx:.8e}")
    print(f"  Second derivative max error (Laplacian): {err_laplacian:.8e}")

    assert err_df_dx < 1e-6, "First derivative test failed."
    assert err_laplacian < 1e-5, "Second derivative test failed."
    print("ALL DERIVATIVE TESTS PASSED CLEANLY.")

if __name__ == "__main__":
    test_autograd_analytical_accuracy()