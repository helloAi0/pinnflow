import logging
from typing import Tuple
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger("DataPipeline")

class ObservationDataset(Dataset):
    def __init__(self, coords: torch.Tensor, fields: torch.Tensor):
        """
        coords: (N, 3) tensor containing [x, y, t]
        fields: (N, 3) tensor containing [u, v, p]
        """
        self.coords = coords
        self.fields = fields

    def __len__(self) -> int:
        return self.coords.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.coords[idx], self.fields[idx]

class DataPipelineManager:
    def __init__(self, h5_path: str = "datasets/raissi_cylinder/reference_cylinder.h5", seed: int = 42):
        self.h5_path = h5_path
        self.seed = seed
        self.generator = torch.Generator().manual_seed(seed)
        
        # Internal state for the parsed 1,000,000 point grid
        self.full_coords: torch.Tensor = None
        self.full_fields: torch.Tensor = None

    def load_and_flatten(self) -> None:
        """Loads HDF5 arrays and flattens them into (N*T, 3) coordinate and field tensors."""
        logger.info(f"Loading reference field from {self.h5_path}...")
        with h5py.File(self.h5_path, 'r') as h5f:
            t_raw = np.array(h5f['t'])
            x_raw = np.array(h5f['x'])
            y_raw = np.array(h5f['y'])
            
            u_raw = np.array(h5f['u'])
            v_raw = np.array(h5f['v'])
            p_raw = np.array(h5f['p'])

        N = x_raw.shape[0]
        T = t_raw.shape[0]

        # Meshgrid generation (N, T)
        T_grid, X_grid = np.meshgrid(t_raw, x_raw)
        _, Y_grid = np.meshgrid(t_raw, y_raw)

        # Flatten space-time matrices
        x_flat = X_grid.flatten()[:, None]
        y_flat = Y_grid.flatten()[:, None]
        t_flat = T_grid.flatten()[:, None]
        
        u_flat = u_raw.flatten()[:, None]
        v_flat = v_raw.flatten()[:, None]
        p_flat = p_raw.flatten()[:, None]

        self.full_coords = torch.tensor(np.hstack((x_flat, y_flat, t_flat)), dtype=torch.float32)
        self.full_fields = torch.tensor(np.hstack((u_flat, v_flat, p_flat)), dtype=torch.float32)
        
        logger.info(f"Data successfully flattened. Total spatio-temporal domain points: {self.full_coords.shape[0]}")

    def prepare_experiment_data(self, 
                                train_budget: int, 
                                noise_level: float = 0.0, 
                                val_ratio: float = 0.1, 
                                test_ratio: float = 0.1) -> dict[str, ObservationDataset]:
        """
        Splits data into train/val/test, applies the data scarcity budget to the training set,
        and injects variance-scaled Gaussian noise.
        """
        if self.full_coords is None:
            self.load_and_flatten()

        total_points = self.full_coords.shape[0]
        indices = torch.randperm(total_points, generator=self.generator)

        n_val = int(total_points * val_ratio)
        n_test = int(total_points * test_ratio)
        n_train_pool = total_points - n_val - n_test

        assert train_budget <= n_train_pool, f"Requested budget {train_budget} exceeds available training points {n_train_pool}."

        # Reserve validation and test sets (clean data, full availability)
        val_idx = indices[:n_val]
        test_idx = indices[n_val:n_val + n_test]
        
        # Extract train pool and apply scarcity budget limit
        train_pool_idx = indices[n_val + n_test:]
        train_budget_idx = train_pool_idx[:train_budget]

        train_coords = self.full_coords[train_budget_idx]
        train_fields = self.full_fields[train_budget_idx]

        # Inject observation noise ONLY into the training set
        if noise_level > 0.0:
            train_fields = self._inject_noise(train_fields, noise_level)

        logger.info(f"Dataset Pipeline Complete:")
        logger.info(f"  Training observations: {train_coords.shape[0]} (Budget constrained, {noise_level*100}% noise)")
        logger.info(f"  Validation points: {n_val} (Clean)")
        logger.info(f"  Test points: {n_test} (Clean)")

        return {
            "train": ObservationDataset(train_coords, train_fields),
            "val": ObservationDataset(self.full_coords[val_idx], self.full_fields[val_idx]),
            "test": ObservationDataset(self.full_coords[test_idx], self.full_fields[test_idx])
        }

    def _inject_noise(self, fields: torch.Tensor, noise_level: float) -> torch.Tensor:
        """Applies zero-mean Gaussian noise scaled by the standard deviation of each field [u, v, p]."""
        noisy_fields = fields.clone()
        for i, field_name in enumerate(['u', 'v', 'p']):
            std_val = torch.std(fields[:, i])
            noise = torch.randn(fields[:, i].shape, generator=self.generator) * std_val * noise_level
            noisy_fields[:, i] += noise
            logger.info(f"  Injected {noise_level*100}% noise into {field_name} (std_scale={std_val:.4f})")
        return noisy_fields

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    pipeline = DataPipelineManager(seed=42)
    # Test strict budget of 1000 points with 5% noise
    datasets = pipeline.prepare_experiment_data(train_budget=1000, noise_level=0.05)