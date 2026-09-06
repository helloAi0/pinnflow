import os
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
import json
from pathlib import Path

from src.models.mlp import NavierStokesMLP

class PINNFailureAnalyzer:
    def __init__(self, model_path="final_pinn_model.pth", data_path="datasets/raissi_cylinder/reference_cylinder.h5"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NavierStokesMLP().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.data_path = data_path
        self.output_dir = Path("results/failure_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_reference_data(self):
        """Loads reference data and normalizes spatial/temporal array shapes."""
        with h5py.File(self.data_path, 'r') as f:
            self.x = np.squeeze(f['x'][:])
            self.y = np.squeeze(f['y'][:])
            self.t = np.squeeze(f['t'][:])
            self.u_ref = np.squeeze(f['u'][:])
            self.v_ref = np.squeeze(f['v'][:])
            self.p_ref = np.squeeze(f['p'][:])

        # Ensure u_ref is shaped (num_times, num_points)
        num_times = len(self.t)
        if self.u_ref.shape[0] != num_times and self.u_ref.shape[-1] == num_times:
            self.u_ref = np.swapaxes(self.u_ref, 0, -1)
            self.v_ref = np.swapaxes(self.v_ref, 0, -1)
            self.p_ref = np.swapaxes(self.p_ref, 0, -1)

    def _get_spatial_coords(self):
        """Generates flattened coordinate arrays depending on whether x, y are 1D lists or 1D grid axes."""
        num_pts = self.u_ref.shape[1]
        
        if len(self.x) == num_pts and len(self.y) == num_pts:
            return self.x.flatten(), self.y.flatten()
        else:
            X, Y = np.meshgrid(self.x, self.y)
            return X.flatten(), Y.flatten()

    def analyze_wake_and_boundary_errors(self, time_idx=-1):
        """Investigates boundary error (cylinder wall) and wake-region error."""
        x_flat, y_flat = self._get_spatial_coords()
        t_val = self.t[time_idx]
        t_flat = np.full_like(x_flat, t_val)
        
        coords = np.stack([x_flat, y_flat, t_flat], axis=1)
        coords_tensor = torch.tensor(coords, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            preds = self.model(coords_tensor).cpu().numpy()
            
        u_pred = preds[:, 0]
        u_true = self.u_ref[time_idx].flatten()
        
        error_u = np.abs(u_pred - u_true)
        
        # 1. Boundary Error Mask (r near 0.5)
        radii = np.sqrt(x_flat**2 + y_flat**2)
        boundary_mask = (radii > 0.45) & (radii < 0.55)
        boundary_error = float(np.mean(error_u[boundary_mask])) if np.any(boundary_mask) else 0.0
        
        # 2. Wake-Region Error Mask (Downstream x > 1.0, |y| < 2.0)
        wake_mask = (x_flat > 1.0) & (np.abs(y_flat) < 2.0)
        wake_error = float(np.mean(error_u[wake_mask])) if np.any(wake_mask) else 0.0
        
        # 3. Global Error
        global_error = float(np.mean(error_u))
        
        return {
            "global_mae": global_error,
            "boundary_layer_mae": boundary_error,
            "wake_region_mae": wake_error,
            "wake_to_global_ratio": wake_error / global_error if global_error > 0 else 0
        }

    def analyze_pressure_drift(self):
        """Investigates temporal pressure drift due to unconstrained pressure gauges."""
        mean_pressures_pred = []
        mean_pressures_true = []
        
        x_flat, y_flat = self._get_spatial_coords()
        
        for i, t_val in enumerate(self.t):
            t_flat = np.full_like(x_flat, t_val)
            coords = np.stack([x_flat, y_flat, t_flat], axis=1)
            coords_tensor = torch.tensor(coords, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                preds = self.model(coords_tensor).cpu().numpy()
            
            p_pred = preds[:, 2]
            p_true = self.p_ref[i].flatten()
            
            mean_pressures_pred.append(np.mean(p_pred))
            mean_pressures_true.append(np.mean(p_true))
            
        plt.figure(figsize=(10, 5))
        plt.plot(self.t, mean_pressures_pred, label="PINN Mean Pressure (Drift)", color='red')
        plt.plot(self.t, mean_pressures_true, label="Reference Mean Pressure", color='black', linestyle='--')
        plt.xlabel("Time (t)")
        plt.ylabel("Mean Spatial Pressure")
        plt.title("Failure Analysis: Temporal Pressure Drift")
        plt.legend()
        plt.grid(True)
        
        plot_path = self.output_dir / "pressure_drift_analysis.png"
        plt.savefig(plot_path)
        plt.close()
        
        drift_variance = float(np.var(np.array(mean_pressures_pred) - np.array(mean_pressures_true)))
        return {"pressure_drift_variance": drift_variance}

    def run_full_analysis(self):
        """Executes all failure mode diagnostics and saves a report."""
        print("Starting PINN Failure Analysis...")
        self.load_reference_data()
        
        report = {}
        
        print("1. Analyzing Spatial Error Distributions (Wake & Boundary)...")
        spatial_metrics = self.analyze_wake_and_boundary_errors()
        report.update(spatial_metrics)
        
        print("2. Analyzing Temporal Pressure Drift...")
        pressure_metrics = self.analyze_pressure_drift()
        report.update(pressure_metrics)
        
        report_path = self.output_dir / "failure_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
            
        print(f"Analysis complete. Report saved to {report_path}")
        print("\n--- Failure Analysis Summary ---")
        for key, value in report.items():
            print(f"{key}: {value:.6f}")

if __name__ == "__main__":
    analyzer = PINNFailureAnalyzer()
    analyzer.run_full_analysis()