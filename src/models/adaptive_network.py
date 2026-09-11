import torch
import torch.nn as nn
from typing import List

class AdaptiveActivationPINN(nn.Module):
    """
    PINN with Layer-wise Locally Adaptive Activation Functions (L-LAAF).
    
    Mathematical Formulation:
      phi(a * (W * x + b))
      where 'a' is a learnable scaling parameter that dynamically adapts the slope
      of the activation function to accelerate convergence and avoid local minima.
      
    Reference:
      Jagtap, A. D., Kawaguchi, K., & Karniadakis, G. E. (2020).
      Locally adaptive activation functions with slope recovery for deep and
      physics-informed neural networks. Proceedings of the Royal Society A, 476(2239), 20200334.
    """
    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 3,
        hidden_layers: int = 6,
        hidden_dim: int = 128,
        scale_factor: float = 1.0
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.scale_factor = scale_factor

        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(in_dim, hidden_dim))
        for _ in range(hidden_layers - 1):
            self.linears.append(nn.Linear(hidden_dim, hidden_dim))
        self.out_layer = nn.Linear(hidden_dim, out_dim)

        # Learnable scaling parameters initialized to 1.0
        self.a_params = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
            for _ in range(hidden_layers)
        ])

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.linears:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        nn.init.xavier_normal_(self.out_layer.weight)
        if self.out_layer.bias is not None:
            nn.init.zeros_(self.out_layer.bias)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = coords
        for i, linear in enumerate(self.linears):
            scaled_input = self.scale_factor * self.a_params[i] * linear(x)
            x = torch.tanh(scaled_input)
        return self.out_layer(x)

# Backward compatibility alias
AdaptivePINN = AdaptiveActivationPINN