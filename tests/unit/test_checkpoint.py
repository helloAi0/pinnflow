import os
import tempfile
import torch
import pytest
from src.models.factory import create_model
from src.models.checkpoint import save_checkpoint, load_and_validate_checkpoint
from src.config.physics_config import DEFAULT_PHYSICS_CONFIG

def test_checkpoint_roundtrip_and_validation():
    cfg = {"architecture": "standard_mlp", "hidden_layers": 3, "hidden_dim": 32}
    model, _ = create_model(cfg)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_model.pth")
        
        # 1. Save checkpoint
        sha = save_checkpoint(
            model=model,
            model_config=cfg,
            filepath=ckpt_path,
            training_seed=123,
            reynolds_number=100.0,
            metrics={"rel_l2_velocity": 0.042}
        )
        assert len(sha) == 64
        assert os.path.exists(ckpt_path)

        # 2. Validate and load
        is_valid, loaded_model, metadata, err = load_and_validate_checkpoint(ckpt_path, physics_config=DEFAULT_PHYSICS_CONFIG)
        assert is_valid is True
        assert err is None
        assert loaded_model is not None
        assert metadata["training_seed"] == 123
        assert metadata["metrics"]["rel_l2_velocity"] == pytest.approx(0.042)

def test_checkpoint_validation_fails_on_corrupt():
    with tempfile.TemporaryDirectory() as tmpdir:
        corrupt_path = os.path.join(tmpdir, "corrupt.pth")
        with open(corrupt_path, "wb") as f:
            f.write(b"not_a_torch_checkpoint")
        
        is_valid, loaded_model, metadata, err = load_and_validate_checkpoint(corrupt_path)
        assert is_valid is False
        assert loaded_model is None
        assert err is not None
