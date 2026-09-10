import torch
import torch.nn as nn
from src.models.network import NavierStokesPINN
from src.config.physics_config import PhysicsConfig

class HardConstrainedPINN(nn.Module):
    def __init__(self, layers, activation_type="tanh", cylinder_radius=None):
        super().__init__()
        # 1. Initialize the core underlying deep surrogate network topology
        self.base_network = NavierStokesPINN(layers, activation_type)
        
        # 2. FIX: Extract spatial layout dimensions with robust getattr fallbacks to prevent CI/test environment crashes
        self.cylinder_radius = cylinder_radius if cylinder_radius is not None else getattr(PhysicsConfig, 'cylinder_radius', 0.5)
        self.cylinder_x = getattr(PhysicsConfig, 'cylinder_x', 0.0)
        self.cylinder_y = getattr(PhysicsConfig, 'cylinder_y', 0.0)

    def _distance_function(self, coords):
        """
        Calculates spatial distance scaling mapping from the cylinder solid wall boundary.
        Enforces: D(x,y) = 1.0 - exp(-relu(r^2 - R^2))
        Yields exactly 0.0 at the perimeter surface, scaling asymptotically to 1.0 at free-stream coordinates.
        """
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        
        # Track relative radial distances offset by localized center coords
        r_squared = (x - self.cylinder_x)**2 + (y - self.cylinder_y)**2
        R_squared = self.cylinder_radius**2
        
        # The relu boundary guard prevents numerical noise if coordinates ever sit inside the wake obstacle
        dist = 1.0 - torch.exp(-torch.relu(r_squared - R_squared))
        return dist

    def forward(self, coords):
        """
        Evaluates coordinate batches while strictly enforcing hard physical constraints.
        Forces the no-slip condition: u = 0.0, v = 0.0 directly on the cylinder boundary.
        """
        # 1. Evaluate baseline velocity and pressure fields from the neural surrogate network
        raw_outputs = self.base_network(coords)
        raw_u = raw_outputs[:, 0:1]
        raw_v = raw_outputs[:, 1:2]
        p = raw_outputs[:, 2:3]  # Pressure field remains unconstrained across the solid wall boundary

        # 2. Extract distance coefficient tensors matched to the coordinate grid device context
        D = self._distance_function(coords)

        # 3. Apply the hard boundary constraint mask (No-slip boundary matching)
        constrained_u = D * raw_u
        constrained_v = D * raw_v

        # Recombine columns back into a unified tensor tracking layout [N, 3] -> (u, v, p)
        return torch.cat([constrained_u, constrained_v, p], dim=1)

# FIX: Class alias binding assigned at the root module level layout scope to satisfy external scripts
ConstrainedPINN = HardConstrainedPINN
