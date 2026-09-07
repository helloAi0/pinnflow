from src.config.physics_config import PhysicsConfig


class CylinderDomain:

  def __init__(
      self,
      x_min: float = PhysicsConfig.X_MIN,
      x_max: float = PhysicsConfig.X_MAX,
      y_min: float = PhysicsConfig.Y_MIN,
      y_max: float = PhysicsConfig.Y_MAX,
      cylinder_x: float = PhysicsConfig.CYLINDER_X,
      cylinder_y: float = PhysicsConfig.CYLINDER_Y,
      radius: float = PhysicsConfig.CYLINDER_RADIUS,
      reynolds_number: float = PhysicsConfig.REYNOLDS_NUMBER,
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
    return ((x - self.cylinder_x) ** 2 + (y - self.cylinder_y) ** 2) < (
        self.radius**2
    )

  def is_inside_fluid(self, x, y):
    return not self.is_inside_cylinder(x, y)