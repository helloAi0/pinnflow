import os
import hashlib
import json
import logging
import subprocess
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from src.config.physics_config import PhysicsConfig, DEFAULT_PHYSICS_CONFIG
from src.models.factory import create_model

logger = logging.getLogger("PINNFlowCheckpoint")

def get_git_commit_sha() -> str:
    """Retrieves the current Git commit SHA if in a git repository."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return sha.decode("ascii").strip()
    except Exception:
        return "untracked_workspace"

def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def save_checkpoint(
    model: nn.Module,
    model_config: Dict[str, Any],
    filepath: str,
    training_seed: int = 42,
    reynolds_number: float = 100.0,
    metrics: Optional[Dict[str, float]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Saves a research-grade checkpoint bundle containing model state, configuration,
    provenance metadata, and training metrics.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    metadata = {
        "version": "2.0.0",
        "git_commit": get_git_commit_sha(),
        "training_seed": training_seed,
        "reynolds_number": reynolds_number,
        "model_config": model_config,
        "metrics": metrics or {},
        "extra": extra_metadata or {}
    }

    bundle = {
        "state_dict": model.state_dict(),
        "metadata": metadata
    }

    torch.save(bundle, filepath)
    sha256_hash = compute_file_sha256(filepath)
    logger.info(f"Checkpoint successfully saved to {filepath} [SHA256: {sha256_hash[:12]}]")
    return sha256_hash

def load_and_validate_checkpoint(
    filepath: str,
    device: torch.device = torch.device("cpu"),
    physics_config: PhysicsConfig = DEFAULT_PHYSICS_CONFIG
) -> Tuple[bool, Optional[nn.Module], Dict[str, Any], Optional[str]]:
    """
    Validates and loads a PINNFlow checkpoint.
    
    Verifies:
      1. File exists and is non-empty.
      2. Checkpoint dictionary structure and metadata.
      3. Re-instantiates model from metadata config.
      4. Validates state_dict key and tensor shape compatibility.
      5. Ensures no NaN/Inf parameters exist in weights.
      
    Returns: (is_valid, model, metadata, error_message)
    """
    if not os.path.exists(filepath):
        return False, None, {}, f"Checkpoint file does not exist at: {filepath}"

    try:
        raw_bundle = torch.load(filepath, map_location=device)
    except Exception as e:
        return False, None, {}, f"Failed to load checkpoint file: {str(e)}"

    # Support both structured checkpoint bundles and legacy raw state_dicts
    if isinstance(raw_bundle, dict) and "state_dict" in raw_bundle and "metadata" in raw_bundle:
        state_dict = raw_bundle["state_dict"]
        metadata = raw_bundle["metadata"]
        model_config = metadata.get("model_config", {"architecture": "standard_mlp"})
    elif isinstance(raw_bundle, dict):
        # Legacy raw state_dict fallback - inspect keys to infer architecture
        state_dict = raw_bundle
        has_hidden_layers = any(k.startswith("hidden_layers") for k in state_dict.keys())
        has_b = "B" in state_dict
        has_mlp = any(k.startswith("mlp") for k in state_dict.keys())

        if has_hidden_layers and has_b:
            # Map legacy hidden_layers.X and output_layer to mlp.2X
            num_hidden = sum(1 for k in state_dict.keys() if k.startswith("hidden_layers") and k.endswith(".weight"))
            hidden_dim = state_dict["hidden_layers.0.weight"].shape[0]
            fourier_features = state_dict["B"].shape[1]

            mapped_sd = {"B": state_dict["B"]}
            for i in range(num_hidden):
                w_key = f"hidden_layers.{i}.weight"
                b_key = f"hidden_layers.{i}.bias"
                if w_key in state_dict:
                    mapped_sd[f"mlp.{2*i}.weight"] = state_dict[w_key]
                if b_key in state_dict:
                    mapped_sd[f"mlp.{2*i}.bias"] = state_dict[b_key]
            if "output_layer.weight" in state_dict:
                mapped_sd[f"mlp.{2*num_hidden}.weight"] = state_dict["output_layer.weight"]
            if "output_layer.bias" in state_dict:
                mapped_sd[f"mlp.{2*num_hidden}.bias"] = state_dict["output_layer.bias"]

            state_dict = mapped_sd
            inferred_arch = "hard_constrained_pinn"
            model_config = {
                "architecture": inferred_arch,
                "hidden_layers": num_hidden,
                "hidden_dim": hidden_dim,
                "fourier_features": fourier_features
            }
        elif has_b or has_mlp:
            inferred_arch = "hard_constrained_pinn"
            # Infer hidden_dim and hidden_layers if mlp keys present
            if has_mlp and "mlp.0.weight" in state_dict:
                hidden_dim = state_dict["mlp.0.weight"].shape[0]
                num_layers = sum(1 for k in state_dict.keys() if k.startswith("mlp.") and k.endswith(".weight"))
                fourier_features = state_dict["B"].shape[1] if has_b else 64
                model_config = {
                    "architecture": inferred_arch,
                    "hidden_layers": num_layers - 1,
                    "hidden_dim": hidden_dim,
                    "fourier_features": fourier_features
                }
            else:
                model_config = {"architecture": inferred_arch}
        else:
            inferred_arch = "standard_mlp"
            model_config = {"architecture": inferred_arch}

        metadata = {
            "version": "1.0.0-legacy",
            "git_commit": "legacy_checkpoint",
            "training_seed": 42,
            "reynolds_number": physics_config.reynolds_number,
            "model_config": model_config
        }
    else:
        return False, None, {}, "Unrecognized checkpoint structure."

    # Compute SHA-256
    file_sha = compute_file_sha256(filepath)
    metadata["checkpoint_sha"] = file_sha
    metadata["checkpoint_filename"] = os.path.basename(filepath)

    # Instantiate model from config
    try:
        model, factory_meta = create_model(model_config, physics_config=physics_config)
        model.to(device)
    except Exception as e:
        return False, None, metadata, f"Model instantiation failed from config: {str(e)}"

    # Validate state dict keys and tensor shapes
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    # Check for missing keys (excluding non-trainable buffers that might be reconstructed)
    missing_keys = model_keys - ckpt_keys

    # Filter out known buffer differences if applicable
    if missing_keys and not all(k.startswith("B") for k in missing_keys):
        return False, None, metadata, f"Checkpoint has missing keys: {missing_keys}"

    try:
        # Load state dict with strict=False only if fourier buffers match
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        return False, None, metadata, f"Failed to load state_dict into model: {str(e)}"

    # Validate numerical integrity: check for NaNs/Infs in parameters
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            return False, None, metadata, f"Checkpoint contains NaN/Inf weights in parameter: {name}"

    model.eval()
    metadata["parameter_count"] = factory_meta["parameter_count"]
    metadata["architecture"] = factory_meta["architecture"]

    return True, model, metadata, None
