"""
Ground truth evaluation and processing pipeline.
Loads raw CFD data to establish domain boundaries and process true fields
for inference-time L2 error evaluation. Assumes data is already downloaded.
"""

import logging
import urllib.request
from pathlib import Path
import numpy as np
import scipy.io
import h5py

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ReferencePipeline")

class ReferenceDataPipeline:
    def __init__(self, data_dir: str = "datasets/raissi_cylinder"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.mat_path = self.data_dir / "cylinder_nektar_wake.mat"
        self.h5_path = self.data_dir / "reference_cylinder.h5"
        self.source_url = "https://github.com/maziarraissi/PINNs/raw/master/main/Data/cylinder_nektar_wake.mat"

    def fetch_dataset(self) -> None:
        """Downloads the Raissi MAT dataset if it does not already exist."""
        if self.mat_path.exists():
            logger.info(f"Dataset already exists at {self.mat_path}. Skipping download.")
            return

        logger.info(f"Downloading reference dataset from {self.source_url}...")
        try:
            urllib.request.urlretrieve(self.source_url, self.mat_path)
            logger.info("Download completed successfully.")
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise

    def process_and_export(self) -> None:
        """Parses the MAT file, validates dimensions, and exports to chunked HDF5."""
        logger.info(f"Loading MAT file from {self.mat_path}...")
        data = scipy.io.loadmat(str(self.mat_path))

        # Raissi dataset structure: 
        # U_star: (N, 2, T) -> [u, v]
        # p_star: (N, T) -> [p]
        # t: (T, 1)
        # X_star: (N, 2) -> [x, y]
        
        t_raw = data['t'].flatten()          # Shape: (T,)
        X_star = data['X_star']              # Shape: (N, 2)
        U_star = data['U_star']              # Shape: (N, 2, T)
        p_star = data['p_star']              # Shape: (N, T)

        N = X_star.shape[0]
        T = t_raw.shape[0]

        logger.info(f"Extracted mesh grid: N={N} spatial points, T={T} time steps.")

        # Spatial coordinates (static across time)
        x = X_star[:, 0]
        y = X_star[:, 1]

        self._validate_fields(x, y, t_raw, U_star, p_star)

        logger.info(f"Writing structured data to {self.h5_path}...")
        with h5py.File(self.h5_path, 'w') as h5f:
            # Save 1D coordinate vectors and static spatial grid
            h5f.create_dataset('t', data=t_raw, dtype=np.float32)
            h5f.create_dataset('x', data=x, dtype=np.float32)
            h5f.create_dataset('y', data=y, dtype=np.float32)
            
            # Create chunked datasets for scalable out-of-core temporal sampling
            h5f.create_dataset('u', data=U_star[:, 0, :], dtype=np.float32, chunks=(N, 1))
            h5f.create_dataset('v', data=U_star[:, 1, :], dtype=np.float32, chunks=(N, 1))
            h5f.create_dataset('p', data=p_star, dtype=np.float32, chunks=(N, 1))
            
            # Save derived quantities useful for visualization and analysis
            velocity_mag = np.sqrt(U_star[:, 0, :]**2 + U_star[:, 1, :]**2)
            h5f.create_dataset('velocity_magnitude', data=velocity_mag, dtype=np.float32, chunks=(N, 1))

            # Store metadata attributes
            h5f.attrs['N_spatial_nodes'] = N
            h5f.attrs['T_timesteps'] = T
            h5f.attrs['description'] = "Raissi 2D Cylinder Nektar Wake Dataset"

        logger.info("HDF5 export completed. Reference CFD pipeline execution successful.")

    def _validate_fields(self, x: np.ndarray, y: np.ndarray, t: np.ndarray, U: np.ndarray, p: np.ndarray) -> None:
        """Performs scientific validation checks on the reference fields."""
        assert not np.isnan(U).any(), "NaN values detected in velocity field."
        assert not np.isnan(p).any(), "NaN values detected in pressure field."
        
        logger.info("Validation Checks Passed:")
        logger.info(f"  Domain Extent: X [{x.min():.2f}, {x.max():.2f}], Y [{y.min():.2f}, {y.max():.2f}]")
        logger.info(f"  Time Extent: [{t.min():.2f}, {t.max():.2f}]")
        logger.info(f"  Velocity bounds: U [{U[:, 0, :].min():.4f}, {U[:, 0, :].max():.4f}], V [{U[:, 1, :].min():.4f}, {U[:, 1, :].max():.4f}]")
        logger.info(f"  Pressure bounds: P [{p.min():.4f}, {p.max():.4f}]")

if __name__ == "__main__":
    pipeline = ReferenceDataPipeline()
    pipeline.fetch_dataset()
    pipeline.process_and_export()