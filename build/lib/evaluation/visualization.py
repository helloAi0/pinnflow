import os
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from pathlib import Path

from src.models.mlp import NavierStokesMLP

class PINNVisualizer:
    def __init__(self, model_path="final_pinn_model.pth", data_path="datasets/raissi_cylinder/reference_cylinder.h5"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NavierStokesMLP().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.data_path = data_path
        self.output_dir = Path("results/figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_publication_plots(self, time_idx=-1):
        """Generates a 3x3 subplot grid of Exact vs Predicted vs Error fields."""
        print(f"Generating flow field visualization for time index: {time_idx}...")
        
        # 1. Load Data (handling the (5000, 200) shape where axis 0 is space, axis 1 is time)
        with h5py.File(self.data_path, 'r') as f:
            x = np.squeeze(f['x'][:])
            y = np.squeeze(f['y'][:])
            t = np.squeeze(f['t'][:])
            
            t_val = t[time_idx]
            u_exact = np.squeeze(f['u'][:])[:, time_idx]
            v_exact = np.squeeze(f['v'][:])[:, time_idx]
            p_exact = np.squeeze(f['p'][:])[:, time_idx]

        # 2. Predict with PINN
        t_flat = np.full_like(x, t_val)
        coords = np.stack([x, y, t_flat], axis=1)
        coords_tensor = torch.tensor(coords, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            preds = self.model(coords_tensor).cpu().numpy()

        u_pred = preds[:, 0]
        v_pred = preds[:, 1]
        p_pred = preds[:, 2]

        # 3. Compute Absolute Error
        u_err = np.abs(u_exact - u_pred)
        v_err = np.abs(v_exact - v_pred)
        p_err = np.abs(p_exact - p_pred)

        # 4. Triangulation for Unstructured Grid
        triang = tri.Triangulation(x, y)
        
        # Mask out the inner cylinder (radius = 0.5) from the triangulation
        radii = np.sqrt(x[triang.triangles].mean(axis=1)**2 + y[triang.triangles].mean(axis=1)**2)
        triang.set_mask(radii < 0.5)

        # 5. Plotting Setup
        fig, axs = plt.subplots(3, 3, figsize=(16, 10), constrained_layout=True)
        
        fields = [
            (u_exact, u_pred, u_err, "Velocity u (m/s)"),
            (v_exact, v_pred, v_err, "Velocity v (m/s)"),
            (p_exact, p_pred, p_err, "Pressure p")
        ]
        
        cmaps = ['RdBu_r', 'RdBu_r', 'magma'] # Diverging for fields, sequential for error

        for i, (f_exact, f_pred, f_err, title) in enumerate(fields):
            # Shared color scale for exact and predicted
            vmin = min(f_exact.min(), f_pred.min())
            vmax = max(f_exact.max(), f_pred.max())
            levels = np.linspace(vmin, vmax, 50)
            err_levels = np.linspace(0, f_err.max(), 50)

            # Exact
            ax1 = axs[i, 0]
            tc1 = ax1.tricontourf(triang, f_exact, levels=levels, cmap=cmaps[0], extend='both')
            fig.colorbar(tc1, ax=ax1, fraction=0.046, pad=0.04)
            ax1.set_title(f"Exact {title}", fontweight='bold')
            ax1.set_aspect('equal')
            ax1.set_axis_off()

            # Predicted
            ax2 = axs[i, 1]
            tc2 = ax2.tricontourf(triang, f_pred, levels=levels, cmap=cmaps[0], extend='both')
            fig.colorbar(tc2, ax=ax2, fraction=0.046, pad=0.04)
            ax2.set_title(f"PINN {title}", fontweight='bold')
            ax2.set_aspect('equal')
            ax2.set_axis_off()

            # Error
            ax3 = axs[i, 2]
            tc3 = ax3.tricontourf(triang, f_err, levels=err_levels, cmap=cmaps[2], extend='max')
            fig.colorbar(tc3, ax=ax3, fraction=0.046, pad=0.04)
            ax3.set_title(f"Absolute Error |{title.split()[1]}|", fontweight='bold')
            ax3.set_aspect('equal')
            ax3.set_axis_off()

        # Save High-Resolution Outputs
        t_rounded = round(t_val, 2)
        png_path = self.output_dir / f"flow_fields_t{t_rounded}.png"
        pdf_path = self.output_dir / f"flow_fields_t{t_rounded}.pdf"
        
        plt.savefig(png_path, dpi=300, bbox_inches='tight', transparent=False, facecolor='white')
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.close()
        
        print(f"Saved publication plots to:\n - {png_path}\n - {pdf_path}")

if __name__ == "__main__":
    vis = PINNVisualizer()
    # Generate plot for the midpoint time step
    vis.generate_publication_plots(time_idx=100)
    # Generate plot for the final time step
    vis.generate_publication_plots(time_idx=-1)