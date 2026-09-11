import torch
import pytest
from src.evaluation.splits import random_point_split, temporal_holdout_split, verify_disjointness

def test_random_point_split_disjoint():
    coords = torch.randn(1000, 3)
    fields = torch.randn(1000, 3)
    
    splits = random_point_split(coords, fields, train_budget=500, val_ratio=0.1, test_ratio=0.2, seed=42)
    train_c, train_f = splits["train"]
    val_c, val_f = splits["val"]
    test_c, test_f = splits["test"]

    assert train_c.shape[0] == 500
    assert val_c.shape[0] == 100
    assert test_c.shape[0] == 200

    assert verify_disjointness(train_c, test_c)
    assert verify_disjointness(train_c, val_c)

def test_temporal_holdout_split_no_leakage():
    t_vals = torch.linspace(0.0, 20.0, 1000)[:, None]
    xy = torch.randn(1000, 2)
    coords = torch.cat([xy, t_vals], dim=1)
    fields = torch.randn(1000, 3)

    splits = temporal_holdout_split(coords, fields, split_time=15.0, train_budget=500, seed=42)
    train_c, _ = splits["train"]
    test_c, _ = splits["test"]

    assert (train_c[:, 2] <= 15.0).all()
    assert (test_c[:, 2] > 15.0).all()
    assert verify_disjointness(train_c, test_c)
