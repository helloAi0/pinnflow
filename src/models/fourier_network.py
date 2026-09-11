import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional
from src.config.physics_config import PhysicsConfig, DEFAULT_PHYSICS_CONFIG

class FourierFeaturePINN(nn.Module):
    """
    Random Fourier Feature Neural Network mapping (x, y, t) -> (u, v, p).
    Projects coordinates using static Gaussian frequencies B ~ N(0, sigma^2)
    to mitigate the spectral bias of standard coordinate MLPs.
    
    Reference:
      Tancik, M., Srinivasan, P., Mildenhall, B., et al. (2020).
      Fourier features let networks learn high frequency functions in low dimensional domains.
      NeurIPS 2020, 33, 7537-7547.
      Wang, S., Wang, H., & Perdikaris, P. (2021). On the eigenvector bias of Fourier
      features for physics-informed neural networks. Computer Methods in Applied Mechanics, 384, 113938.
    """
    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 3,
        fourier_features: int = 64,
        fourier_sigma: float = 1.0,
        hidden_layers: int = 6,
        hidden_dim: int = 128,
        apply_hard_constraints: bool = False,
        physics_config: PhysicsConfig = DEFAULT_PHYSICS_CONFIG
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fourier_features = fourier_features
        self.fourier_sigma = fourier_sigma
        self.apply_hard_constraints = apply_hard_constraints
        self.physics_config = physics_config

        # Static Gaussian projection matrix: shape (in_dim, fourier_features)
        # Register as a buffer so it is included in state_dict and moved with model.to(device)
        B_init = torch.randn(in_dim, fourier_features) * fourier_sigma
        self.register_buffer("B", B_init)

        fourier_dim = fourier_features * 2  # sin and cos components

        layers: List[nn.Module] = []
        layers.append(nn.Linear(fourier_dim, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, out_dim))
        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _distance_mask(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Distance scaling function D(x, y) = 1.0 - exp(-relu(r^2 - R^2)).
        Yields D = 0.0 at cylinder perimeter r = R, smoothly approaching 1.0 in free stream.
        """
        x = coords[:, 0:1] - self.physics_config.cylinder_center[0]
        y = coords[:, 1:2] - self.physics_config.cylinder_center[1]
        r_sq = x ** 2 + y ** 2
        R_sq = self.physics_config.cylinder_radius ** 2
        return 1.0 - torch.exp(-torch.relu(r_sq - R_sq))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: (N, 3) tensor [x, y, t]
        Returns: (N, 3) tensor [u, v, p]
        """
        # Fourier embedding projection: 2 * pi * x @ B
        projected = 2.0 * float(np.pi) * torch.matmul(coords, self.B)
        x_emb = torch.cat([torch.cos(projected), torch.sin(projected)], dim=-1)

        raw_outputs = self.mlp(x_emb)

        if not self.apply_hard_constraints:
            return raw_outputs

        # Enforce no-slip u = 0, v = 0 on cylinder boundary
        raw_u = raw_outputs[:, 0:1]
        raw_v = raw_outputs[:, 1:2]
        p = raw_outputs[:, 2:3]

        D = self._distance_mask(coords)
        u_constrained = D * raw_u
        v_constrained = D * raw_v

        return torch.cat([u_constrained, v_constrained, p], dim=1)

# Backward compatibility alias
FourierConstrainedPINN = FourierFeaturePINN