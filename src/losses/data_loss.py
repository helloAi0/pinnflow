import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

class DataLoss(nn.Module):
    """
    Supervised Data Loss for sparse velocity and optional pressure observations.
    
    Mathematical definition:
      L_u = (1/N_data) * sum_{i=1}^{N_data} (u_pred_i - u_obs_i)^2
      L_v = (1/N_data) * sum_{i=1}^{N_data} (v_pred_i - v_obs_i)^2
      L_p = (1/N_data) * sum_{i=1}^{N_data} (p_pred_i - p_obs_i)^2 (if pressure is observed)
      
      L_data = w_u * L_u + w_v * L_v + w_p * L_p
      
    Reference:
      Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).
      Physics-informed neural networks: A deep learning framework for solving forward
      and inverse problems involving nonlinear partial differential equations.
      Journal of Computational Physics, 378, 686-707.
    """
    def __init__(self, weight_u: float = 1.0, weight_v: float = 1.0, weight_p: float = 0.0, normalize: bool = False):
        super().__init__()
        self.weight_u = weight_u
        self.weight_v = weight_v
        self.weight_p = weight_p
        self.normalize = normalize
        self.mse = nn.MSELoss(reduction="mean")

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        target_variances: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        predictions: (N, 3) tensor containing [u, v, p]
        targets: (N, 3) tensor containing [u, v, p]
        """
        u_pred, v_pred, p_pred = predictions[:, 0:1], predictions[:, 1:2], predictions[:, 2:3]
        u_true, v_true, p_true = targets[:, 0:1], targets[:, 1:2], targets[:, 2:3]

        loss_u = self.mse(u_pred, u_true)
        loss_v = self.mse(v_pred, v_true)
        loss_p = self.mse(p_pred, p_true)

        if self.normalize and target_variances is not None:
            var_u, var_v, var_p = target_variances[0], target_variances[1], target_variances[2]
            loss_u = loss_u / torch.clamp(var_u, min=1e-8)
            loss_v = loss_v / torch.clamp(var_v, min=1e-8)
            loss_p = loss_p / torch.clamp(var_p, min=1e-8)

        total_data_loss = (
            self.weight_u * loss_u +
            self.weight_v * loss_v +
            self.weight_p * loss_p
        )

        component_dict = {
            "loss_u": loss_u,
            "loss_v": loss_v,
            "loss_p": loss_p,
            "loss_data_total": total_data_loss
        }

        return total_data_loss, component_dict
