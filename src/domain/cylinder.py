from src.config.physics_config import PhysicsConfig

class CylinderDomain:
    def __init__(
        self,
        # FIX: Converted parameter bindings from uppercase constants to lowercase attributes
        x_min: float = PhysicsConfig.x_min,
        x_max: float = PhysicsConfig.x_max,
        y_min: float = PhysicsConfig.y_min,
        y_max: float = PhysicsConfig.y_max,
        cylinder_x: float = PhysicsConfig.cylinder_x,
        cylinder_y: float = PhysicsConfig.cylinder_y,
        radius: float = PhysicsConfig.cylinder_radius,
        reynolds_number: float = PhysicsConfig.reynolds_number,
    ):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.cylinder_x = cylinder_x
        self.cylinder_y = cylinder_y
        self.radius = radius
        self.reynolds_number = reynolds_number
        self.kinematic_viscosity = 1.0 / reynolds_number

    def is_inside_cylinder(self, x, y):
        """Checks if a given coordinate falls inside the solid cylinder obstacle boundary."""
        return ((x - self.cylinder_x) ** 2 + (y - self.cylinder_y) ** 2) < (self.radius ** 2)

    def is_inside_fluid(self, x, y):
        """Validates if a coordinate point resides inside the surrounding active fluid domain mesh."""
        return not self.is_inside_cylinder(x, y)
