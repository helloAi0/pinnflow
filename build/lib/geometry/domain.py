import torch

class CylinderDomain:
    def __init__(self, x_min=1.0, x_max=8.0, y_min=-2.0, y_max=2.0,
                 t_min=0.0, t_max=20.0, cyl_center=(0.0, 0.0), cyl_radius=0.5):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.t_min = t_min
        self.t_max = t_max
        self.cyl_center = cyl_center
        self.cyl_radius = cyl_radius

    def is_inside_fluid(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Returns a boolean mask where True means the point is in the fluid 
        (inside the bounding box and strictly outside the cylinder).
        """
        dx = x - self.cyl_center[0]
        dy = y - self.cyl_center[1]
        r_sq = dx**2 + dy**2
        
        outside_cylinder = r_sq >= (self.cyl_radius ** 2)
        inside_box_x = (x >= self.x_min) & (x <= self.x_max)
        inside_box_y = (y >= self.y_min) & (y <= self.y_max)
        
        return outside_cylinder & inside_box_x & inside_box_y