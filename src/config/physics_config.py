from dataclasses import dataclass

@dataclass(frozen=True)
class PhysicsConfig:
    # Physical settings
    reynolds_number: float = 100.0
    density: float = 1.0
    pressure_gauge_coord: tuple[float, float] = (8.0, 2.0)
    
    # FIX: Dynamic parameter derivation (1 / Re)
    @property
    def kinematic_viscosity(self) -> float:
        return 1.0 / self.reynolds_number
    
    # Domain extents
    # FIX: Extended x_min to -2.0 to safely encapsulate the cylinder boundary at the origin
    x_min: float = -2.0  
    x_max: float = 10.0
    y_min: float = -2.0
    y_max: float = 2.0
    t_min: float = 0.0
    t_max: float = 20.0
    
    # Cylinder geometry
    cylinder_center: tuple[float, float] = (0.0, 0.0)
    cylinder_radius: float = 0.5
