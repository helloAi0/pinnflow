import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from src.config.physics_config import PhysicsConfig, DEFAULT_PHYSICS_CONFIG

class PressureGaugeLoss(nn.Module):
    """
    Pressure Reference Gauge Loss.
    
    Mathematical Formulation:
      In incompressible Navier-Stokes equations, pressure appears exclusively via its spatial
      gradient nabla p. The absolute pressure field therefore has an arbitrary additive constant:
        p(x, y, t) -> p(x, y, t) + C(t)
      To anchor the pressure field uniquely, we pin the predicted pressure at a reference coordinate
      (x_0, y_0) to a reference value p_0 (typically 0.0 or reference DNS pressure).
      
      L_gauge = (1/N_gauge) * sum_{i=1}^{N_gauge} (p(x_0, y_0, t_i) - p_0)^2
      
    Reference:
      Gresho, P. M., & Sani, R. L. (1987). On pressure boundary conditions,
      vorticity boundary conditions, and the invalidity of the 'divergence-free'
      basis for the Navier-Stokes equations. International Journal for Numerical
      Methods in Fluids, 7(10), 1111-1145.
    """
    def __init__(
        self,
        physics_config: PhysicsConfig = DEFAULT_PHYSICS_CONFIG,
        weight_gauge: float = 1.0,
        reference_pressure: float = 0.0
    ):
        super().__init__()
        self.config = physics_config
        self.weight_gauge = weight_gauge
        self.reference_pressure = reference_pressure
        self.mse = nn.MSELoss(reduction="mean")

    def forward(
        self,
        gauge_preds: torch.Tensor,
        reference_targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        gauge_preds: (N, 3) tensor [u, v, p] at gauge coordinate (x_0, y_0, t)
        """
        p_pred = gauge_preds[:, 2:3]
        if reference_targets is not None:
            p_ref = reference_targets[:, 2:3] if reference_targets.shape[1] >= 3 else reference_targets
        else:
            p_ref = torch.full_like(p_pred, self.reference_pressure)

        loss_gauge = self.mse(p_pred, p_ref)
        weighted_loss = self.weight_gauge * loss_gauge

        return weighted_loss, {
            "loss_pressure_gauge": loss_gauge,
            "loss_pressure_gauge_weighted": weighted_loss
        }
