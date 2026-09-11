import torch
import torch.nn as nn
from typing import Dict, Tuple
from src.physics.derivatives import compute_first_derivatives, compute_second_derivatives
from src.config.physics_config import PhysicsConfig, DEFAULT_PHYSICS_CONFIG

class PDELoss(nn.Module):
    """
    Incompressible Navier-Stokes PDE Residual Loss.
    
    Mathematical Formulation (Dimensionless 2D Incompressible Navier-Stokes):
      Continuity equation (mass conservation):
        R_c = du/dx + dv/dy = 0
        
      X-Momentum equation:
        R_u = du/dt + u*(du/dx) + v*(du/dy) + (1/rho)*dp/dx - nu*(d^2u/dx^2 + d^2u/dy^2) = 0
        
      Y-Momentum equation:
        R_v = dv/dt + u*(dv/dx) + v*(dv/dy) + (1/rho)*dp/dy - nu*(d^2v/dx^2 + d^2v/dy^2) = 0
        
      PDE Loss:
        L_pde = w_c * mean(R_c^2) + w_u * mean(R_u^2) + w_v * mean(R_v^2)
        
    Reference:
      Batchelor, G. K. (2000). An Introduction to Fluid Dynamics. Cambridge University Press.
      Raissi, M., Yazdani, A., & Karniadakis, G. E. (2020). Hidden fluid mechanics:
      Learning velocity and pressure fields from flow visualizations. Science, 367(6481), 1026-1030.
    """
    def __init__(
        self,
        physics_config: PhysicsConfig = DEFAULT_PHYSICS_CONFIG,
        weight_continuity: float = 1.0,
        weight_momentum_x: float = 1.0,
        weight_momentum_y: float = 1.0
    ):
        super().__init__()
        self.reynolds_number = physics_config.reynolds_number
        self.nu = physics_config.kinematic_viscosity
        self.density = physics_config.density
        
        self.weight_c = weight_continuity
        self.weight_u = weight_momentum_x
        self.weight_v = weight_momentum_y

    def compute_residuals(self, coords: torch.Tensor, outputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computes pointwise PDE residuals and vorticity via automatic differentiation.
        coords: (N, 3) tensor [x, y, t], requires_grad=True
        outputs: (N, 3) tensor [u, v, p]
        """
        u = outputs[:, 0:1]
        v = outputs[:, 1:2]
        
        # 1st-order derivatives w.r.t [x, y, t]
        u_grads, v_grads, p_grads = compute_first_derivatives(outputs, coords)
        
        # 2nd-order derivatives w.r.t [x, y]
        u_2nd, v_2nd = compute_second_derivatives(u_grads, v_grads, coords)
        
        # Incompressibility (Continuity)
        r_c = u_grads["x"] + v_grads["y"]
        
        # Momentum-X
        r_u = (
            u_grads["t"]
            + u * u_grads["x"]
            + v * u_grads["y"]
            + (1.0 / self.density) * p_grads["x"]
            - self.nu * u_2nd["laplacian"]
        )
        
        # Momentum-Y
        r_v = (
            v_grads["t"]
            + u * v_grads["x"]
            + v * v_grads["y"]
            + (1.0 / self.density) * p_grads["y"]
            - self.nu * v_2nd["laplacian"]
        )
        
        # Vorticity diagnostic: omega = dv/dx - du/dy
        vorticity = v_grads["x"] - u_grads["y"]
        
        return {
            "continuity": r_c,
            "momentum_x": r_u,
            "momentum_y": r_v,
            "vorticity": vorticity
        }

    def forward(
        self,
        coords: torch.Tensor,
        outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Calculates MSE of Navier-Stokes residuals over collocation points.
        """
        residuals = self.compute_residuals(coords, outputs)
        
        loss_c = torch.mean(residuals["continuity"] ** 2)
        loss_u = torch.mean(residuals["momentum_x"] ** 2)
        loss_v = torch.mean(residuals["momentum_y"] ** 2)
        
        total_pde_loss = (
            self.weight_c * loss_c +
            self.weight_u * loss_u +
            self.weight_v * loss_v
        )
        
        loss_dict = {
            "loss_continuity": loss_c,
            "loss_momentum_x": loss_u,
            "loss_momentum_y": loss_v,
            "loss_pde_total": total_pde_loss
        }
        
        return total_pde_loss, loss_dict, residuals
