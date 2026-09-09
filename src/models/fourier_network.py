import torch
import torch.nn as nn
import numpy as np

class FourierConstrainedPINN(nn.Module):
    def __init__(self, layers, activation_type="tanh", cylinder_radius=0.5, fourier_features=64, sigma=1.0):
        super().__init__()
        self.cylinder_radius = cylinder_radius
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Static Fourier mapping matrix: Input dim (3) x Fourier features
        # Requires_grad=False ensures these are static embeddings, not learned weights
        self.B = nn.Parameter(torch.randn(3, fourier_features) * sigma, requires_grad=False).to(self.device)
        
        # Calculate new input size: fourier_features * 2 (sine and cosine components)
        in_features = fourier_features * 2
        
        # Build MLP layers
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(in_features, layers[1]))
        
        for i in range(1, len(layers) - 2):
            self.hidden_layers.append(nn.Linear(layers[i], layers[i+1]))
            
        self.output_layer = nn.Linear(layers[-2], layers[-1])
        self.activation = torch.tanh if activation_type == "tanh" else torch.relu
        self.to(self.device)

    def _distance_function(self, coords):
        """Zero exactly at the cylinder wall, approaching 1 elsewhere."""
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        r_squared = x**2 + y**2
        R_squared = self.cylinder_radius**2
        return 1.0 - torch.exp(-torch.relu(r_squared - R_squared))

    def forward(self, coords):
        # 1. Fourier Feature Embedding projection
        projected = 2.0 * np.pi * torch.matmul(coords, self.B)
        x_fourier = torch.cat([torch.cos(projected), torch.sin(projected)], dim=-1)
        
        # 2. MLP Forward Pass
        x_hidden = x_fourier
        for layer in self.hidden_layers:
            x_hidden = self.activation(layer(x_hidden))
        raw_outputs = self.output_layer(x_hidden)
        
        # 3. Hard Constraints enforcement
        raw_u = raw_outputs[:, 0:1]
        raw_v = raw_outputs[:, 1:2]
        p = raw_outputs[:, 2:3]
        
        D = self._distance_function(coords)
        constrained_u = D * raw_u
        constrained_v = D * raw_v
        
        return torch.cat([constrained_u, constrained_v, p], dim=1)