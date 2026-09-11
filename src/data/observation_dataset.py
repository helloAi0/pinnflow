import os
import hashlib
import logging
from typing import Dict, Tuple, Optional, Any
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader
from src.evaluation.splits import random_point_split, sensor_location_split, temporal_holdout_split

logger = logging.getLogger("ObservationDataset")

DATASET_PROVENANCE = {
    "name": "Karniadakis Cylinder 2D DNS Wake Benchmark",
    "authors": "Maziar Raissi, Paris Perdikaris, George Em Karniadakis",
    "year": 2019,
    "license": "MIT / Academic Research Open Access",
    "reynolds_number": 100.0,
    "fluid": "Incompressible Newtonian (2D laminar vortex shedding)",
    "numerical_method": "High-order spectral element solver (Nektar)",
    "spatial_points": 5000,
    "time_snapshots": 200,
    "total_space_time_points": 1000000,
    "x_range": [1.0, 8.0],
    "y_range": [-2.0, 2.0],
    "t_range": [0.0, 19.9],
    "dt": 0.1,
    "fields": ["u (x-velocity)", "v (y-velocity)", "p (kinematic pressure)"]
}

class FlowDataset(Dataset):
    """PyTorch Dataset wrapper for spatio-temporal coordinate and field tensors."""
    def __init__(self, coords: torch.Tensor, fields: torch.Tensor):
        assert coords.shape[0] == fields.shape[0], "Coords and fields must have matching row counts."
        self.coords = coords.float()
        self.fields = fields.float()

    def __len__(self) -> int:
        return self.coords.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.coords[idx], self.fields[idx]


class DataPipelineManager:
    """
    Manages loading, validation, splitting, and noise injection for CFD observation datasets.
    """
    DEFAULT_DATA_PATHS = [
        "datasets/raissi_cylinder/cylinder_nektar_wake.mat",
        "data/cylinder_nektar_wake.mat",
        "datasets/raissi_cylinder/reference_cylinder.h5"
    ]

    def __init__(self, data_path: Optional[str] = None, seed: int = 42):
        self.seed = seed
        self.data_path = self._resolve_data_path(data_path)
        self.dataset_sha256 = self._compute_checksum()
        
        # Raw DNS data storage
        self.X_star: Optional[np.ndarray] = None  # (N, 2)
        self.t_star: Optional[np.ndarray] = None  # (T,)
        self.U_star: Optional[np.ndarray] = None  # (N, 2, T)
        self.p_star: Optional[np.ndarray] = None  # (N, T)
        
        # Flattened full domain tensors: (N*T, 3)
        self.full_coords: Optional[torch.Tensor] = None
        self.full_fields: Optional[torch.Tensor] = None
        
        self.load_dataset()

    def _resolve_data_path(self, path: Optional[str]) -> str:
        if path and os.path.exists(path) and os.path.getsize(path) > 1000000:
            return path
        for default_path in self.DEFAULT_DATA_PATHS:
            if os.path.exists(default_path) and os.path.getsize(default_path) > 1000000:
                return default_path
        # If no 1MB+ file found, fallback to first path
        return self.DEFAULT_DATA_PATHS[0]

    def _compute_checksum(self) -> str:
        if not os.path.exists(self.data_path):
            return "dataset_not_found"
        hasher = hashlib.sha256()
        with open(self.data_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def load_dataset(self) -> None:
        """Loads and structures the raw CFD data matrix."""
        logger.info(f"Loading reference CFD dataset from: {self.data_path}")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"CFD dataset missing at {self.data_path}")

        mat_data = scipy.io.loadmat(self.data_path)
        self.X_star = mat_data["X_star"].astype(np.float32)       # (N, 2)
        self.t_star = mat_data["t"].astype(np.float32).flatten()   # (T,)
        self.U_star = mat_data["U_star"].astype(np.float32)       # (N, 2, T)
        self.p_star = mat_data["p_star"].astype(np.float32)       # (N, T)

        N_spatial = self.X_star.shape[0]
        N_time = self.t_star.shape[0]

        # Construct full spatio-temporal coordinate grid (N * T, 3)
        # Using standard tile arrangement matching Raissi et al. convention
        XX = np.tile(self.X_star[:, 0:1], (1, N_time))  # (N, T)
        YY = np.tile(self.X_star[:, 1:2], (1, N_time))  # (N, T)
        TT = np.tile(self.t_star, (N_spatial, 1))       # (N, T)

        x_flat = XX.flatten()[:, None]
        y_flat = YY.flatten()[:, None]
        t_flat = TT.flatten()[:, None]

        u_flat = self.U_star[:, 0, :].flatten()[:, None]
        v_flat = self.U_star[:, 1, :].flatten()[:, None]
        p_flat = self.p_star.flatten()[:, None]

        coords_np = np.hstack([x_flat, y_flat, t_flat])
        fields_np = np.hstack([u_flat, v_flat, p_flat])

        self.full_coords = torch.from_numpy(coords_np)
        self.full_fields = torch.from_numpy(fields_np)

        logger.info(f"Loaded {self.full_coords.shape[0]} spatio-temporal points. Dataset SHA: {self.dataset_sha256[:12]}")

    def get_split(
        self,
        protocol: str = "random",
        budget: int = 5000,
        noise_level: float = 0.0,
        seed: Optional[int] = None,
        **kwargs: Any
    ) -> Dict[str, FlowDataset]:
        """
        Extracts train/val/test FlowDatasets under the specified evaluation protocol.
        Noise is strictly injected ONLY into training fields.
        """
        split_seed = seed if seed is not None else self.seed
        
        if protocol in ["random", "random_point_split"]:
            val_ratio = kwargs.get("val_ratio", 0.1)
            test_ratio = kwargs.get("test_ratio", 0.2)
            splits = random_point_split(
                self.full_coords,
                self.full_fields,
                train_budget=budget,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=split_seed
            )
            train_c, train_f = splits["train"]
            val_c, val_f = splits["val"]
            test_c, test_f = splits["test"]

            # Apply observation noise ONLY to train
            if noise_level > 0.0:
                train_f = self._inject_noise(train_f, noise_level, split_seed)

            return {
                "train": FlowDataset(train_c, train_f),
                "val": FlowDataset(val_c, val_f),
                "test": FlowDataset(test_c, test_f)
            }

        elif protocol in ["sensor", "sensor_location_split"]:
            num_sensors = kwargs.get("num_sensors_train", 50)
            splits = sensor_location_split(
                self.full_coords,
                self.full_fields,
                spatial_coords=self.X_star,
                num_sensors_train=num_sensors,
                seed=split_seed
            )
            train_c, train_f = splits["train"]
            test_c, test_f = splits["test"]

            if noise_level > 0.0:
                train_f = self._inject_noise(train_f, noise_level, split_seed)

            return {
                "train": FlowDataset(train_c, train_f),
                "test": FlowDataset(test_c, test_f)
            }

        elif protocol in ["temporal", "temporal_holdout_split"]:
            split_time = kwargs.get("split_time", 14.0)
            splits = temporal_holdout_split(
                self.full_coords,
                self.full_fields,
                split_time=split_time,
                train_budget=budget,
                seed=split_seed
            )
            train_c, train_f = splits["train"]
            test_c, test_f = splits["test"]

            if noise_level > 0.0:
                train_f = self._inject_noise(train_f, noise_level, split_seed)

            return {
                "train": FlowDataset(train_c, train_f),
                "test": FlowDataset(test_c, test_f)
            }
        else:
            raise ValueError(f"Unknown split protocol '{protocol}'. Choose from: random, sensor, temporal")

    def _inject_noise(self, fields: torch.Tensor, noise_level: float, seed: int) -> torch.Tensor:
        """Injects zero-mean Gaussian noise scaled by standard deviation of each field."""
        generator = torch.Generator().manual_seed(seed)
        noisy = fields.clone()
        for i in range(fields.shape[1]):
            std = torch.std(fields[:, i])
            noise = torch.randn(fields[:, i].shape, generator=generator) * std * noise_level
            noisy[:, i] += noise
        return noisy

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "provenance": DATASET_PROVENANCE,
            "data_path": self.data_path,
            "sha256": self.dataset_sha256,
            "total_points": int(self.full_coords.shape[0]) if self.full_coords is not None else 0
        }