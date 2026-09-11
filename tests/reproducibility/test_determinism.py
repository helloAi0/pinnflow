import torch
import numpy as np
import pytest
from src.models.factory import create_model

def test_model_initialization_seed_determinism():
    cfg = {"architecture": "standard_mlp", "hidden_layers": 3, "hidden_dim": 32}

    # Run 1 with seed 42
    torch.manual_seed(42)
    model1, _ = create_model(cfg)
    w1 = next(model1.parameters()).clone()

    # Run 2 with seed 42
    torch.manual_seed(42)
    model2, _ = create_model(cfg)
    w2 = next(model2.parameters()).clone()

    # Run 3 with seed 999
    torch.manual_seed(999)
    model3, _ = create_model(cfg)
    w3 = next(model3.parameters()).clone()

    assert torch.equal(w1, w2), "Models initialized with identical seed must have identical parameters."
    assert not torch.equal(w1, w3), "Models initialized with different seeds must have different parameters."
