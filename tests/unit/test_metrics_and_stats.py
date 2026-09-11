import torch
import numpy as np
import pytest
from src.evaluation.metrics import relative_l2_error, l_infinity_error, compute_all_metrics
from src.evaluation.statistics import compute_confidence_interval_95, aggregate_seed_metrics, paired_comparison_test
from src.models.mlp import NavierStokesMLP

def test_relative_l2_error():
    true = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    pred = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    err = relative_l2_error(pred, true)
    assert err == pytest.approx(0.0, abs=1e-6)

    pred_offset = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    err_offset = relative_l2_error(pred_offset, true)
    assert err_offset == pytest.approx(1.0, abs=1e-5)

def test_l_infinity_error():
    true = torch.tensor([1.0, 2.0, 3.0])
    pred = torch.tensor([1.1, 2.0, 3.4])
    linf = l_infinity_error(pred, true)
    assert linf == pytest.approx(0.4, abs=1e-5)

def test_confidence_interval_95():
    samples = [0.05, 0.06, 0.04, 0.055, 0.045]
    res = compute_confidence_interval_95(samples)
    assert res["mean"] == pytest.approx(0.05, abs=1e-4)
    assert res["ci95_low"] < res["mean"] < res["ci95_high"]
    assert res["n"] == 5

def test_paired_comparison():
    base = [0.10, 0.12, 0.11, 0.13, 0.09]
    prop = [0.05, 0.06, 0.055, 0.065, 0.045]
    comp = paired_comparison_test(base, prop)
    assert comp["mean_difference"] < 0.0
    assert comp["relative_improvement_pct"] > 0.0
    assert comp["is_statistically_significant_005"] is True
