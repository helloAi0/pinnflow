import torch
import torch.nn as nn
from typing import Dict, Tuple

class StaticPINNLoss(nn.Module):
    def __init__(self, w_data: float = 1.0, w_cont: float = 1.0, w_mom_u: float = 1.0, w_mom_v: float = 1.0):
        super().__init__()
        self.w_data = w_data
        self.w_cont = w_cont
        self.w_mom_u = w_mom_u
        self.w_mom_v = w_mom_v
        self.mse = nn.MSELoss()

    def forward(self, 
                data_preds: torch.Tensor, 
                data_targets: torch.Tensor, 
                pde_residuals: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes weighted total loss and returns itemized individual loss values for tracking.
        """
        l_data = self.mse(data_preds, data_targets)
        l_cont = torch.mean(pde_residuals["continuity"] ** 2)
        l_mom_u = torch.mean(pde_residuals["momentum_x"] ** 2)
        l_mom_v = torch.mean(pde_residuals["momentum_y"] ** 2)

        l_total = (
            self.w_data * l_data +
            self.w_cont * l_cont +
            self.w_mom_u * l_mom_u +
            self.w_mom_v * l_mom_v
        )

        loss_components = {
            "total_loss": l_total.item(),
            "data_loss": l_data.item(),
            "continuity_loss": l_cont.item(),
            "momentum_x_loss": l_mom_u.item(),
            "momentum_y_loss": l_mom_v.item(),
            "physics_loss": (l_cont + l_mom_u + l_mom_v).item()
        }

        return l_total, loss_components