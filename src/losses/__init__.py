from src.losses.data_loss import DataLoss
from src.losses.pde_loss import PDELoss
from src.losses.boundary_condition_loss import BoundaryConditionLoss
from src.losses.pressure_gauge_loss import PressureGaugeLoss
from src.losses.weighting import (
    HomoscedasticAdaptiveWeighting,
    DynamicGradientWeighting,
    UnifiedPINNLoss
)

__all__ = [
    "DataLoss",
    "PDELoss",
    "BoundaryConditionLoss",
    "PressureGaugeLoss",
    "HomoscedasticAdaptiveWeighting",
    "DynamicGradientWeighting",
    "UnifiedPINNLoss"
]
