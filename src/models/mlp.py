import torch
import torch.nn as nn
from typing import List, Literal

class NavierStokesMLP(nn.Module):
    """
    Canonical Multi-Layer Perceptron for 2D Navier-Stokes flow fields.
    Maps spatio-temporal coordinates (x, y, t) -> (u, v, p).
    
    Uses C^infty smooth activations (Tanh or Sine) to ensure continuous
    and non-trivial higher-order automatic differentiation derivatives.
    """
    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 3,
        hidden_layers: int = 6,
        hidden_dim: int = 128,
        activation_type: Literal["tanh", "sin", "gelu"] = "tanh"
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.activation_type = activation_type
        
        if activation_type.lower() == "tanh":
            activation = nn.Tanh()
        elif activation_type.lower() == "sin":
            class SineActivation(nn.Module):
                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return torch.sin(x)
            activation = SineActivation()
        elif activation_type.lower() == "gelu":
            activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation_type}")

        layers: List[nn.Module] = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(activation)

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation)

        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier normal initialization for linear layers."""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        """
        xyt: (N, 3) tensor [x, y, t]
        Returns: (N, 3) tensor [u, v, p]
        """
        return self.net(xyt)