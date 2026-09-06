from dataclasses import dataclass

@dataclass(frozen=True)
class PhysicsConfig:
    reynolds_number: float = 100.0
    kinematic_viscosity: float = 0.01  # 1 / Re
    density: float = 1.0
    pressure_gauge_coord: tuple[float, float] = (8.0, 2.0)
    
    # Domain extents
    x_min: float = 1.0
    x_max: float = 8.0
    y_min: float = -2.0
    y_max: float = 2.0
    t_min: float = 0.0
    t_max: float = 20.0
    
    # Cylinder geometry
    cylinder_center: tuple[float, float] = (0.0, 0.0)
    cylinder_radius: float = 0.5