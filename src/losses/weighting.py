import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Literal
from src.losses.data_loss import DataLoss
from src.losses.pde_loss import PDELoss
from src.losses.boundary_condition_loss import BoundaryConditionLoss
from src.losses.pressure_gauge_loss import PressureGaugeLoss
from src.config.physics_config import PhysicsConfig, DEFAULT_PHYSICS_CONFIG

class HomoscedasticAdaptiveWeighting(nn.Module):
    """
    Multi-task homoscedastic uncertainty loss weighting.
    
    Mathematical Formulation:
      L(theta, s) = sum_{k} [ 0.5 * exp(-s_k) * L_k(theta) + 0.5 * s_k ]
      where s_k = log(sigma_k^2) are learnable log-variance parameters.
      
    Reference:
      Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-task learning using
      uncertainty to weigh losses for scene geometry and semantics.
      CVPR 2018, pp. 7482-7491.
    """
    def __init__(self, num_losses: int = 4, initial_log_var: float = 0.0):
        super().__init__()
        self.log_vars = nn.Parameter(torch.full((num_losses,), initial_log_var, dtype=torch.float32))

    def forward(self, losses: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        assert len(losses) == len(self.log_vars), f"Expected {len(self.log_vars)} losses, got {len(losses)}"
        total_loss = torch.tensor(0.0, device=self.log_vars.device)
        weights_dict = {}

        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss = total_loss + 0.5 * precision * loss + 0.5 * self.log_vars[i]
            weights_dict[f"weight_{i}"] = float(precision.item())
            weights_dict[f"log_var_{i}"] = float(self.log_vars[i].item())

        return total_loss, weights_dict


class DynamicGradientWeighting:
    """
    Gradient pathology mitigation via dynamic learning rate / gradient norm balancing.
    
    Mathematical Formulation:
      lambda_hat_k = max_theta ||nabla_theta L_data|| / mean_theta ||nabla_theta L_k||
      lambda_k^(n+1) = (1 - alpha) * lambda_k^(n) + alpha * lambda_hat_k
      
    Reference:
      Wang, S., Teng, Y., & Perdikaris, P. (2021). Understanding and mitigating gradient
      pathologies in physics-informed neural networks. SIAM Journal on Scientific
      Computing, 43(5), A3055-A3081.
    """
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.weights: Dict[str, float] = {}

    def update_weights(
        self,
        loss_dict: Dict[str, torch.Tensor],
        shared_parameters: List[nn.Parameter],
        base_key: str = "loss_data"
    ) -> Dict[str, float]:
        if base_key not in loss_dict:
            return self.weights

        base_loss = loss_dict[base_key]
        grads_base = torch.autograd.grad(base_loss, shared_parameters, retain_graph=True, allow_unused=True)
        norm_base = torch.sqrt(sum(torch.sum(g ** 2) for g in grads_base if g is not None))

        new_weights = {}
        for k, loss_val in loss_dict.items():
            if k == base_key:
                new_weights[k] = 1.0
                continue
            grads_k = torch.autograd.grad(loss_val, shared_parameters, retain_graph=True, allow_unused=True)
            norm_k = torch.sqrt(sum(torch.sum(g ** 2) for g in grads_k if g is not None))
            
            if norm_k > 1e-8:
                lambda_hat = (norm_base / norm_k).item()
            else:
                lambda_hat = 1.0

            old_weight = self.weights.get(k, 1.0)
            new_weights[k] = (1.0 - self.alpha) * old_weight + self.alpha * lambda_hat

        self.weights = new_weights
        return self.weights


class UnifiedPINNLoss(nn.Module):
    """
    Unified Loss Engine combining Data, Navier-Stokes PDE, Boundary, and Pressure Gauge terms.
    Supports 'static', 'adaptive' (homoscedastic), and 'gradient_weighted' weighting strategies.
    """
    def __init__(
        self,
        physics_config: PhysicsConfig = DEFAULT_PHYSICS_CONFIG,
        strategy: Literal["static", "adaptive", "gradient_weighted"] = "static",
        weight_data: float = 1.0,
        weight_pde: float = 1.0,
        weight_bc: float = 1.0,
        weight_gauge: float = 0.1
    ):
        super().__init__()
        self.strategy = strategy
        self.physics_config = physics_config
        
        self.data_loss_fn = DataLoss(weight_u=1.0, weight_v=1.0, weight_p=0.0)
        self.pde_loss_fn = PDELoss(physics_config=physics_config)
        self.bc_loss_fn = BoundaryConditionLoss(physics_config=physics_config)
        self.gauge_loss_fn = PressureGaugeLoss(physics_config=physics_config)

        self.w_data = weight_data
        self.w_pde = weight_pde
        self.w_bc = weight_bc
        self.w_gauge = weight_gauge

        # 4 loss components: [data, continuity, momentum_x, momentum_y]
        if self.strategy == "adaptive":
            self.adaptive_weighting = HomoscedasticAdaptiveWeighting(num_losses=4)
        elif self.strategy == "gradient_weighted":
            self.grad_weighting = DynamicGradientWeighting(alpha=0.1)

    def forward(
        self,
        model: nn.Module,
        data_coords: torch.Tensor,
        data_targets: torch.Tensor,
        pde_coords: torch.Tensor,
        cylinder_coords: Optional[torch.Tensor] = None,
        gauge_coords: Optional[torch.Tensor] = None,
        gauge_targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
        # 1. Supervised Data Loss
        data_preds = model(data_coords)
        loss_data, data_dict = self.data_loss_fn(data_preds, data_targets)

        # 2. PDE Residual Loss
        pde_preds = model(pde_coords)
        loss_pde, pde_dict, pde_residuals = self.pde_loss_fn(pde_coords, pde_preds)

        # 3. Boundary Condition Loss
        if cylinder_coords is not None and cylinder_coords.shape[0] > 0:
            cyl_preds = model(cylinder_coords)
            loss_bc, bc_dict = self.bc_loss_fn(cylinder_coords, cyl_preds)
        else:
            loss_bc = torch.tensor(0.0, device=data_coords.device)
            bc_dict = {"loss_bc_total": loss_bc}

        # 4. Pressure Gauge Loss
        if gauge_coords is not None and gauge_coords.shape[0] > 0:
            gauge_preds = model(gauge_coords)
            loss_gauge, gauge_dict = self.gauge_loss_fn(gauge_preds, gauge_targets)
        else:
            loss_gauge = torch.tensor(0.0, device=data_coords.device)
            gauge_dict = {"loss_pressure_gauge": loss_gauge}

        # Combine losses based on weighting strategy
        metrics_dict: Dict[str, float] = {
            "loss_data": float(loss_data.item()),
            "loss_u": float(data_dict["loss_u"].item()),
            "loss_v": float(data_dict["loss_v"].item()),
            "loss_continuity": float(pde_dict["loss_continuity"].item()),
            "loss_momentum_x": float(pde_dict["loss_momentum_x"].item()),
            "loss_momentum_y": float(pde_dict["loss_momentum_y"].item()),
            "loss_pde": float(loss_pde.item()),
            "loss_bc": float(loss_bc.item()),
            "loss_gauge": float(loss_gauge.item())
        }

        if self.strategy == "adaptive":
            losses = [
                loss_data,
                pde_dict["loss_continuity"],
                pde_dict["loss_momentum_x"],
                pde_dict["loss_momentum_y"]
            ]
            total_loss, ad_weights = self.adaptive_weighting(losses)
            total_loss = total_loss + self.w_bc * loss_bc + self.w_gauge * loss_gauge
            for k, v in ad_weights.items():
                metrics_dict[k] = v
        else:
            total_loss = (
                self.w_data * loss_data +
                self.w_pde * loss_pde +
                self.w_bc * loss_bc +
                self.w_gauge * loss_gauge
            )

        metrics_dict["total_loss"] = float(total_loss.item())
        return total_loss, metrics_dict, pde_residuals
