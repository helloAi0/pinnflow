import torch
from typing import Tuple, Dict

def compute_first_derivatives(outputs: torch.Tensor, coords: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Computes first-order spatial and temporal derivatives for u, v, p.
    coords: (N, 3) tensor [x, y, t] requiring gradients
    outputs: (N, 3) tensor [u, v, p]
    
    Returns tuple of dictionaries: (velocity_grads, pressure_grads)
    """
    u = outputs[:, 0:1]
    v = outputs[:, 1:2]
    p = outputs[:, 2:3]

    ones = torch.ones_like(u)

    # Autograd gradients w.r.t. coords [x, y, t]
    u_g = torch.autograd.grad(u, coords, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    v_g = torch.autograd.grad(v, coords, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    p_g = torch.autograd.grad(p, coords, grad_outputs=ones, create_graph=True, retain_graph=True)[0]

    u_grads = {
        "x": u_g[:, 0:1],
        "y": u_g[:, 1:2],
        "t": u_g[:, 2:3]
    }
    v_grads = {
        "x": v_g[:, 0:1],
        "y": v_g[:, 1:2],
        "t": v_g[:, 2:3]
    }
    p_grads = {
        "x": p_g[:, 0:1],
        "y": p_g[:, 1:2],
        "t": p_g[:, 2:3]
    }

    return u_grads, v_grads, p_grads

def compute_second_derivatives(u_grads: Dict[str, torch.Tensor], 
                                v_grads: Dict[str, torch.Tensor], 
                                coords: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Computes second-order spatial derivatives required for viscous stress terms.
    """
    ones = torch.ones_like(u_grads["x"])

    # Second derivatives for u
    u_xx = torch.autograd.grad(u_grads["x"], coords, grad_outputs=ones, create_graph=True, retain_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_grads["y"], coords, grad_outputs=ones, create_graph=True, retain_graph=True)[0][:, 1:2]

    # Second derivatives for v
    v_xx = torch.autograd.grad(v_grads["x"], coords, grad_outputs=ones, create_graph=True, retain_graph=True)[0][:, 0:1]
    v_yy = torch.autograd.grad(v_grads["y"], coords, grad_outputs=ones, create_graph=True, retain_graph=True)[0][:, 1:2]

    u_2nd = {"xx": u_xx, "yy": u_yy, "laplacian": u_xx + u_yy}
    v_2nd = {"xx": v_xx, "yy": v_yy, "laplacian": v_xx + v_yy}

    return u_2nd, v_2nd