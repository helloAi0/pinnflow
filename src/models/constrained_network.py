import torch
import torch.nn as nn
from src.models.network import NavierStokesPINN

class HardConstrainedPINN(nn.Module):
    def __init__(self, layers, activation_type="tanh", cylinder_radius=0.5):
        super().__init__()
        # Initialize the base network to handle the heavy lifting
        self.base_network = NavierStokesPINN(layers, activation_type)
        self.cylinder_radius = cylinder_radius
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def _distance_function(self, coords):
        """
        Calculates distance from the cylinder boundary.
        Assuming cylinder is centered at (0,0).
        D(x,y) = 1 - exp(-(r^2 - R^2)) 
        This is 0 at the cylinder surface (r=R) and approaches 1 far away.
        """
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        
        r_squared = x**2 + y**2
        R_squared = self.cylinder_radius**2
        
        # We only want this constraint applied outside or on the cylinder
        # The relu ensures distance is 0 if inside the cylinder (though data shouldn't be there)
        dist = 1.0 - torch.exp(-torch.relu(r_squared - R_squared))
        return dist

    def forward(self, coords):
        # 1. Get raw outputs from the base network
        raw_outputs = self.base_network(coords)
        raw_u = raw_outputs[:, 0:1]
        raw_v = raw_outputs[:, 1:2]
        p = raw_outputs[:, 2:3] # Pressure usually isn't constrained at the wall

        # 2. Calculate the spatial distance function
        D = self._distance_function(coords)

        # 3. Apply the hard constraint (No-slip condition: u=0, v=0 at cylinder wall)
        constrained_u = D * raw_u
        constrained_v = D * raw_v

        # Recombine
        return torch.cat([constrained_u, constrained_v, p], dim=1)