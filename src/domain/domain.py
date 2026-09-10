import torch

class PhysicsConfigFallback:
    x_min = -2.0
    x_max = 8.0
    y_min = -2.0
    y_max = 2.0
    cylinder_radius = 0.5
    cylinder_x = 0.0
    cylinder_y = 0.0

try:
    from src.models import PhysicsConfig
except ImportError:
    PhysicsConfig = PhysicsConfigFallback

class CylinderDomain:
    def __init__(self):
        self.x_min = getattr(PhysicsConfig, 'x_min', -2.0)
        self.x_max = getattr(PhysicsConfig, 'x_max', 8.0)
        self.y_min = getattr(PhysicsConfig, 'y_min', -2.0)
        self.y_max = getattr(PhysicsConfig, 'y_max', 2.0)
        self.cylinder_radius = getattr(PhysicsConfig, 'cylinder_radius', 0.5)
        self.cylinder_x = getattr(PhysicsConfig, 'cylinder_x', 0.0)
        self.cylinder_y = getattr(PhysicsConfig, 'cylinder_y', 0.0)

    def is_inside(self, x, y):
        """Returns mask of points that are inside the bounding box but OUTSIDE the cylinder."""
        in_box = (x >= self.x_min) & (x <= self.x_max) & (y >= self.y_min) & (y <= self.y_max)
        r_sq = (x - self.cylinder_x)**2 + (y - self.cylinder_y)**2
        outside_cylinder = r_sq >= self.cylinder_radius**2
        return in_box & outside_cylinder