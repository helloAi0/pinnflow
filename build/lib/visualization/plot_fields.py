import logging
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.patches import Circle

logger = logging.getLogger("Visualizer")

class FieldVisualizer:
    def __init__(self, h5_path: str = "datasets/raissi_cylinder/reference_cylinder.h5"):
        self.h5_path = h5_path
        
    def plot_snapshot(self, model: torch.nn.Module, time_idx: int = 100, device: str = "cpu", save_path: str = "snapshot.png"):
        """
        Reconstructs and plots the CFD fields at a specific time snapshot.
        time_idx=100 corresponds to t=10.0s (midway through the dataset).
        """
        logger.info(f"Loading reference data for time index {time_idx}...")
        with h5py.File(self.h5_path, 'r') as h5f:
            x_ref = np.array(h5f['x']).flatten()
            y_ref = np.array(h5f['y']).flatten()
            t_val = np.array(h5f['t'])[time_idx].item()
            
            u_ref = np.array(h5f['u'])[:, time_idx]
            v_ref = np.array(h5f['v'])[:, time_idx]
            p_ref = np.array(h5f['p'])[:, time_idx]
            
        t_array = np.full_like(x_ref, t_val)
        
        # Prepare tensor inputs
        coords = np.stack([x_ref, y_ref, t_array], axis=1)
        coords_tensor = torch.tensor(coords, dtype=torch.float32).to(device)
        
        # Inference
        model.eval()
        with torch.no_grad():
            preds = model(coords_tensor).cpu().numpy()
            
        u_pred, v_pred, p_pred = preds[:, 0], preds[:, 1], preds[:, 2]
        
        # Calculate Absolute Errors
        u_err, v_err, p_err = np.abs(u_ref - u_pred), np.abs(v_ref - v_pred), np.abs(p_ref - p_pred)
        
        logger.info("Rendering spatial triangulations...")
        self._render_plot(x_ref, y_ref, 
                          [u_ref, v_ref, p_ref], 
                          [u_pred, v_pred, p_pred], 
                          [u_err, v_err, p_err], 
                          time_idx, save_path)

    def _render_plot(self, x, y, refs, preds, errs, t_idx, save_path):
        triang = tri.Triangulation(x, y)
        fig, axes = plt.subplots(3, 3, figsize=(18, 10), sharex=True, sharey=True)
        
        fields = ['Velocity U', 'Velocity V', 'Pressure P']
        colormaps = ['RdBu_r', 'RdBu_r', 'viridis']
        
        for i in range(3):
            # 1. Reference Ground Truth
            ax_ref = axes[i, 0]
            c1 = ax_ref.tricontourf(triang, refs[i], levels=50, cmap=colormaps[i])
            ax_ref.set_ylabel(fields[i], fontsize=14, weight='bold')
            if i == 0: ax_ref.set_title("Exact Dynamics", fontsize=14)
            fig.colorbar(c1, ax=ax_ref, fraction=0.046, pad=0.04)
            
            # 2. PINN Prediction
            ax_pred = axes[i, 1]
            c2 = ax_pred.tricontourf(triang, preds[i], levels=50, cmap=colormaps[i])
            if i == 0: ax_pred.set_title("PINN Prediction", fontsize=14)
            fig.colorbar(c2, ax=ax_pred, fraction=0.046, pad=0.04)
            
            # 3. Absolute Error
            ax_err = axes[i, 2]
            c3 = ax_err.tricontourf(triang, errs[i], levels=50, cmap='magma')
            if i == 0: ax_err.set_title("Absolute Error", fontsize=14)
            fig.colorbar(c3, ax=ax_err, fraction=0.046, pad=0.04)

        # Overlay the cylinder geometry constraint on all subplots
        for ax in axes.flatten():
            cylinder = Circle((0.0, 0.0), radius=0.5, color='gray', zorder=10)
            ax.add_patch(cylinder)
            ax.set_aspect('equal')

        plt.suptitle(f"Navier-Stokes PINN Reconstruction (Time Index = {t_idx})", fontsize=16, y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    # Instantiate visualizer and load an untrained model to verify the pipeline
    from src.models.mlp import NavierStokesMLP
    dummy_model = NavierStokesMLP(hidden_layers=8, hidden_dim=200)
    
    visualizer = FieldVisualizer()
    # Generates snapshot based on the randomly initialized model weights
    visualizer.plot_snapshot(dummy_model, time_idx=100, save_path="test_snapshot.png")