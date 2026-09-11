from dataclasses import dataclass, field
from typing import Tuple

@dataclass(frozen=True)
class PhysicsConfig:
    """
    Authoritative canonical physics and domain configuration for PINNFlow.
    All subsystems (physics, losses, data, training, API, frontend metadata)
    must consume configuration derived from this central definition.
    """
    # Fluid physical properties (Dimensionless 2D Navier-Stokes)
    reynolds_number: float = 100.0
    density: float = 1.0
    
    # Cylinder obstacle geometry (origin-centered in full CFD mesh)
    cylinder_center: Tuple[float, float] = (0.0, 0.0)
    cylinder_radius: float = 0.5  # Diameter D = 1.0
    
    # Observation / Wake evaluation domain (Raissi et al. benchmark coordinates)
    x_min: float = 1.0
    x_max: float = 8.0
    y_min: float = -2.0
    y_max: float = 2.0
    t_min: float = 0.0
    t_max: float = 20.0
    
    # Extended CFD simulation domain (including upstream inflow and cylinder)
    extended_x_min: float = -2.0
    extended_x_max: float = 10.0
    extended_y_min: float = -2.0
    extended_y_max: float = 2.0
    
    # Reference pressure gauge location (downstream wake boundary)
    pressure_gauge_coord: Tuple[float, float] = (8.0, 2.0)
    
    # Reference velocity scale (free-stream inflow velocity U_inf)
    u_inflow: float = 1.0
    v_inflow: float = 0.0

    @property
    def kinematic_viscosity(self) -> float:
        """nu = 1.0 / Re in non-dimensional Navier-Stokes formulation."""
        return 1.0 / self.reynolds_number

    def to_dict(self) -> dict:
        return {
            "reynolds_number": self.reynolds_number,
            "density": self.density,
            "kinematic_viscosity": self.kinematic_viscosity,
            "cylinder_center": list(self.cylinder_center),
            "cylinder_radius": self.cylinder_radius,
            "cylinder_diameter": self.cylinder_radius * 2.0,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "t_min": self.t_min,
            "t_max": self.t_max,
            "extended_x_min": self.extended_x_min,
            "extended_x_max": self.extended_x_max,
            "extended_y_min": self.extended_y_min,
            "extended_y_max": self.extended_y_max,
            "pressure_gauge_coord": list(self.pressure_gauge_coord),
            "u_inflow": self.u_inflow,
            "v_inflow": self.v_inflow,
        }

# Default singleton instance
DEFAULT_PHYSICS_CONFIG = PhysicsConfig()
