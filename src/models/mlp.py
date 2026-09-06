import torch
import torch.nn as nn

class NavierStokesMLP(nn.Module):
    def __init__(self, in_dim: int = 3, out_dim: int = 3, hidden_layers: int = 8, hidden_dim: int = 200):
        """
        Multilayer Perceptron mapping (x, y, t) -> (u, v, p).
        Uses Tanh activations to support smooth higher-order autograd derivatives.
        """
        super().__init__()
        
        layers = []
        # Input layer
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
            
        # Output layer
        layers.append(nn.Linear(hidden_dim, out_dim))
        
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Applies Xavier/Glorot normal initialization."""
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