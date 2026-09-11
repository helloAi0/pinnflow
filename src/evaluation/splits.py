import torch
import numpy as np
from typing import Dict, Tuple, List

def verify_disjointness(train_coords: torch.Tensor, test_coords: torch.Tensor) -> bool:
    """
    Programmatically asserts disjointness between training and test coordinate sets.
    Raises ValueError if data leakage is detected.
    """
    train_np = train_coords.detach().cpu().numpy()
    test_np = test_coords.detach().cpu().numpy()
    
    # Use structured view / hashing for exact row-wise intersection
    train_bytes = {row.tobytes() for row in train_np}
    test_bytes = {row.tobytes() for row in test_np}
    
    overlap = train_bytes.intersection(test_bytes)
    if len(overlap) > 0:
        raise ValueError(f"CRITICAL: Data leakage detected! {len(overlap)} exact coordinate points overlap between train and test sets.")
    return True

def random_point_split(
    coords: torch.Tensor,
    fields: torch.Tensor,
    train_budget: int,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Protocol 1: Random Spatio-Temporal Point Holdout.
    Uniform random subsampling across the full (x, y, t) domain.
    """
    total_points = coords.shape[0]
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total_points, generator=generator)

    n_test = int(total_points * test_ratio)
    n_val = int(total_points * val_ratio)
    n_train_pool = total_points - n_test - n_val

    if train_budget > n_train_pool:
        raise ValueError(f"Train budget {train_budget} exceeds available training pool {n_train_pool}")

    test_idx = perm[:n_test]
    val_idx = perm[n_test:n_test + n_val]
    train_pool_idx = perm[n_test + n_val:]
    train_idx = train_pool_idx[:train_budget]

    train_c, train_f = coords[train_idx], fields[train_idx]
    val_c, val_f = coords[val_idx], fields[val_idx]
    test_c, test_f = coords[test_idx], fields[test_idx]

    verify_disjointness(train_c, test_c)
    verify_disjointness(train_c, val_c)

    return {
        "train": (train_c, train_f),
        "val": (val_c, val_f),
        "test": (test_c, test_f),
        "protocol": "random_point_split"
    }

def sensor_location_split(
    coords: torch.Tensor,
    fields: torch.Tensor,
    spatial_coords: np.ndarray,
    num_sensors_train: int = 50,
    test_sensor_ratio: float = 0.2,
    seed: int = 42
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Protocol 2: Unseen Sensor Location Split (Spatial Generalization).
    Selects fixed spatial probes (sensors) across all time steps.
    A subset of spatial probe locations is completely held out for evaluation.
    """
    num_spatial = spatial_coords.shape[0]
    np_rng = np.random.RandomState(seed)
    spatial_perm = np_rng.permutation(num_spatial)

    num_test_sensors = int(num_spatial * test_sensor_ratio)
    test_sensor_indices = set(spatial_perm[:num_test_sensors])
    train_pool_sensor_indices = spatial_perm[num_test_sensors:]
    
    selected_train_sensor_indices = set(train_pool_sensor_indices[:min(num_sensors_train, len(train_pool_sensor_indices))])

    # In standard cylinder dataset: coords have N_spatial * N_time points
    # where point index i corresponds to spatial index (i // N_time) or similar
    # We match coords by exact spatial coordinate (x, y)
    x_c = coords[:, 0].cpu().numpy()
    y_c = coords[:, 1].cpu().numpy()

    # Pre-build lookup sets
    test_spatial_tuples = {(spatial_coords[idx, 0], spatial_coords[idx, 1]) for idx in test_sensor_indices}
    train_spatial_tuples = {(spatial_coords[idx, 0], spatial_coords[idx, 1]) for idx in selected_train_sensor_indices}

    train_mask = np.array([(x_c[i], y_c[i]) in train_spatial_tuples for i in range(len(coords))])
    test_mask = np.array([(x_c[i], y_c[i]) in test_spatial_tuples for i in range(len(coords))])

    train_c, train_f = coords[train_mask], fields[train_mask]
    test_c, test_f = coords[test_mask], fields[test_mask]

    verify_disjointness(train_c, test_c)

    return {
        "train": (train_c, train_f),
        "test": (test_c, test_f),
        "protocol": "sensor_location_split",
        "num_train_sensors": len(selected_train_sensor_indices),
        "num_test_sensors": len(test_sensor_indices)
    }

def temporal_holdout_split(
    coords: torch.Tensor,
    fields: torch.Tensor,
    split_time: float = 14.0,
    train_budget: int = 5000,
    seed: int = 42
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Protocol 3: Temporal Forecasting Holdout Split.
    Trains strictly on past snapshots t <= split_time.
    Evaluates on unseen future time window t > split_time.
    """
    t_vals = coords[:, 2]
    past_mask = t_vals <= split_time
    future_mask = t_vals > split_time

    past_coords = coords[past_mask]
    past_fields = fields[past_mask]

    future_coords = coords[future_mask]
    future_fields = fields[future_mask]

    generator = torch.Generator().manual_seed(seed)
    if train_budget < past_coords.shape[0]:
        perm = torch.randperm(past_coords.shape[0], generator=generator)
        train_idx = perm[:train_budget]
        train_c = past_coords[train_idx]
        train_f = past_fields[train_idx]
    else:
        train_c = past_coords
        train_f = past_fields

    test_c = future_coords
    test_f = future_fields

    verify_disjointness(train_c, test_c)

    return {
        "train": (train_c, train_f),
        "test": (test_c, test_f),
        "protocol": "temporal_holdout_split",
        "split_time": split_time
    }
