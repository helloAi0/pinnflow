import torch
import torch.nn as nn
from typing import Dict, Tuple

class AdaptivePINNLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Learnable log-variance parameters for [data, cont, mom_x, mom_y]
        # Initialized to 0.0 (exp(0) = 1.0 weight)
        self.log_vars = nn.Parameter(torch.zeros(4))
        self.mse = nn.MSELoss()

    def forward(self, 
                data_preds: torch.Tensor, 
                data_targets: torch.Tensor, 
                pde_residuals: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        l_data = self.mse(data_preds, data_targets)
        l_cont = torch.mean(pde_residuals["continuity"] ** 2)
        l_mom_x = torch.mean(pde_residuals["momentum_x"] ** 2)
        l_mom_y = torch.mean(pde_residuals["momentum_y"] ** 2)

        losses = [l_data, l_cont, l_mom_x, l_mom_y]
        l_total = 0.0
        
        # L_total = sum( L_i * exp(-log_var_i) + log_var_i )
        for i, loss in enumerate(losses):
            l_total += loss * torch.exp(-self.log_vars[i]) + self.log_vars[i]

        loss_components = {
            "total_loss": l_total.item(),
            "data_loss": l_data.item(),
            "continuity_loss": l_cont.item(),
            "momentum_x_loss": l_mom_x.item(),
            "momentum_y_loss": l_mom_y.item(),
            "physics_loss": (l_cont + l_mom_x + l_mom_y).item(),
            "w_data": torch.exp(-self.log_vars[0]).item(),
            "w_cont": torch.exp(-self.log_vars[1]).item(),
            "w_mom_x": torch.exp(-self.log_vars[2]).item(),
            "w_mom_y": torch.exp(-self.log_vars[3]).item()
        }

        return l_total, loss_components