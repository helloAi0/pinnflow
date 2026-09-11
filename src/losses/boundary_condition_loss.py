import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from src.config.physics_config import PhysicsConfig, DEFAULT_PHYSICS_CONFIG

class BoundaryConditionLoss(nn.Module):
    """
    Physical Boundary Condition Penalty Losses.
    
    Mathematical Formulation:
      1. Cylinder Solid Wall No-Slip Condition:
         u(x, y, t) = 0,  v(x, y, t) = 0   for (x - x_c)^2 + (y - y_c)^2 = R^2
         L_noslip = (1/N_bc) * sum (u^2 + v^2)
         
      2. Inflow Dirichlet Condition (optional):
         u(x_in, y, t) = U_inf, v(x_in, y, t) = 0
         
      3. Outflow Convective Condition (optional):
         du/dt + U_inf * du/dx = 0
         
    Reference:
      Schlichting, H., & Gersten, K. (2017). Boundary-layer theory. Springer.
    """
    def __init__(
        self,
        physics_config: PhysicsConfig = DEFAULT_PHYSICS_CONFIG,
        weight_noslip: float = 1.0,
        weight_inflow: float = 1.0
    ):
        super().__init__()
        self.config = physics_config
        self.weight_noslip = weight_noslip
        self.weight_inflow = weight_inflow
        self.mse = nn.MSELoss(reduction="mean")

    def forward(
        self,
        cylinder_coords: Optional[torch.Tensor],
        cylinder_preds: Optional[torch.Tensor],
        inflow_coords: Optional[torch.Tensor] = None,
        inflow_preds: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        total_bc_loss = torch.tensor(0.0, device=cylinder_preds.device if cylinder_preds is not None else torch.device("cpu"))
        component_dict = {}

        if cylinder_preds is not None and cylinder_preds.shape[0] > 0:
            u_cyl = cylinder_preds[:, 0:1]
            v_cyl = cylinder_preds[:, 1:2]
            loss_noslip = self.mse(u_cyl, torch.zeros_like(u_cyl)) + self.mse(v_cyl, torch.zeros_like(v_cyl))
            total_bc_loss = total_bc_loss + self.weight_noslip * loss_noslip
            component_dict["loss_bc_noslip"] = loss_noslip
        else:
            component_dict["loss_bc_noslip"] = torch.tensor(0.0)

        if inflow_preds is not None and inflow_preds.shape[0] > 0:
            u_in = inflow_preds[:, 0:1]
            v_in = inflow_preds[:, 1:2]
            target_u = torch.full_like(u_in, self.config.u_inflow)
            target_v = torch.full_like(v_in, self.config.v_inflow)
            loss_inflow = self.mse(u_in, target_u) + self.mse(v_in, target_v)
            total_bc_loss = total_bc_loss + self.weight_inflow * loss_inflow
            component_dict["loss_bc_inflow"] = loss_inflow
        else:
            component_dict["loss_bc_inflow"] = torch.tensor(0.0)

        component_dict["loss_bc_total"] = total_bc_loss
        return total_bc_loss, component_dict
