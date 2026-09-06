import torch
from typing import Dict
from src.physics.derivatives import compute_first_derivatives, compute_second_derivatives

class NavierStokes2D:
    def __init__(self, reynolds_number: float = 100.0):
        self.Re = reynolds_number
        self.nu = 1.0 / reynolds_number

    def compute_residuals(self, coords: torch.Tensor, outputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computes continuity, momentum-x, momentum-y residuals, and spatial vorticity.
        coords: (N, 3) tensor [x, y, t] requiring gradients
        outputs: (N, 3) tensor [u, v, p]
        
        Returns dictionary containing:
            - continuity: f_c
            - momentum_x: f_u
            - momentum_y: f_v
            - vorticity:  w
        """
        u = outputs[:, 0:1]
        v = outputs[:, 1:2]
        
        # 1st order spatial and temporal derivatives
        u_g, v_g, p_g = compute_first_derivatives(outputs, coords)
        
        # 2nd order spatial derivatives
        u_2nd, v_2nd = compute_second_derivatives(u_g, v_g, coords)

        # Residual Equations
        f_continuity = u_g["x"] + v_g["y"]
        
        f_momentum_x = (
            u_g["t"] 
            + u * u_g["x"] 
            + v * u_g["y"] 
            + p_g["x"] 
            - self.nu * u_2nd["laplacian"]
        )
        
        f_momentum_y = (
            v_g["t"] 
            + u * v_g["x"] 
            + v * v_g["y"] 
            + p_g["y"] 
            - self.nu * v_2nd["laplacian"]
        )

        # Physical Diagnostic: Vorticity w = dv/dx - du/dy
        vorticity = v_g["x"] - u_g["y"]

        return {
            "continuity": f_continuity,
            "momentum_x": f_momentum_x,
            "momentum_y": f_momentum_y,
            "vorticity": vorticity
        }