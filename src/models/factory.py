import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from src.config.physics_config import PhysicsConfig, DEFAULT_PHYSICS_CONFIG
from src.models.mlp import NavierStokesMLP
from src.models.fourier_network import FourierFeaturePINN
from src.models.adaptive_network import AdaptiveActivationPINN

def count_parameters(model: nn.Module) -> int:
    """Returns the total number of trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_model(config: Dict[str, Any], physics_config: PhysicsConfig = DEFAULT_PHYSICS_CONFIG) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Canonical Model Factory for PINNFlow.
    
    Supported model architectures:
      - 'standard_mlp' / 'mlp': Standard coordinate MLP mapping (x, y, t) -> (u, v, p).
      - 'fourier_pinn' / 'fourier': Fourier Feature PINN embedding high-frequency modes.
      - 'hard_constrained_pinn' / 'constrained': Fourier PINN with exact cylinder no-slip distance masking.
      - 'adaptive_activation_pinn' / 'adaptive': Layer-wise locally adaptive activation functions.
      
    Returns (model, model_metadata).
    """
    arch_type = config.get("architecture", config.get("model_type", "standard_mlp")).lower()
    in_dim = int(config.get("in_dim", 3))
    out_dim = int(config.get("out_dim", 3))
    hidden_layers = int(config.get("hidden_layers", 6))
    hidden_dim = int(config.get("hidden_dim", 128))
    activation = config.get("activation", "tanh")

    if arch_type in ["standard_mlp", "mlp", "baseline_mlp"]:
        model = NavierStokesMLP(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_layers=hidden_layers,
            hidden_dim=hidden_dim,
            activation_type=activation
        )
        canonical_name = "standard_mlp"

    elif arch_type in ["fourier_pinn", "fourier"]:
        fourier_features = int(config.get("fourier_features", 64))
        fourier_sigma = float(config.get("fourier_sigma", 1.0))
        model = FourierFeaturePINN(
            in_dim=in_dim,
            out_dim=out_dim,
            fourier_features=fourier_features,
            fourier_sigma=fourier_sigma,
            hidden_layers=hidden_layers,
            hidden_dim=hidden_dim,
            apply_hard_constraints=False,
            physics_config=physics_config
        )
        canonical_name = "fourier_pinn"

    elif arch_type in ["hard_constrained_pinn", "constrained", "fourier_constrained"]:
        fourier_features = int(config.get("fourier_features", 64))
        fourier_sigma = float(config.get("fourier_sigma", 1.0))
        model = FourierFeaturePINN(
            in_dim=in_dim,
            out_dim=out_dim,
            fourier_features=fourier_features,
            fourier_sigma=fourier_sigma,
            hidden_layers=hidden_layers,
            hidden_dim=hidden_dim,
            apply_hard_constraints=True,
            physics_config=physics_config
        )
        canonical_name = "hard_constrained_pinn"

    elif arch_type in ["adaptive_activation_pinn", "adaptive", "adaptive_pinn"]:
        scale_factor = float(config.get("scale_factor", 1.0))
        model = AdaptiveActivationPINN(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_layers=hidden_layers,
            hidden_dim=hidden_dim,
            scale_factor=scale_factor
        )
        canonical_name = "adaptive_activation_pinn"

    else:
        raise ValueError(f"Unknown model architecture '{arch_type}'. Choose from: standard_mlp, fourier_pinn, hard_constrained_pinn, adaptive_activation_pinn")

    metadata = {
        "architecture": canonical_name,
        "parameter_count": count_parameters(model),
        "in_dim": in_dim,
        "out_dim": out_dim,
        "hidden_layers": hidden_layers,
        "hidden_dim": hidden_dim,
        "activation": activation,
        "reynolds_number": physics_config.reynolds_number,
        "config": config
    }

    return model, metadata
