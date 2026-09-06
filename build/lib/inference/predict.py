import argparse
import json
import torch
import numpy as np
from pathlib import Path

from src.models.mlp import NavierStokesMLP

class PINNInferenceEngine:
    """Standalone inference engine for deployed PINN fluid field predictions."""
    
    def __init__(self, model_path="final_pinn_model.pth", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model = NavierStokesMLP().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict_fields(self, x: np.ndarray, y: np.ndarray, t: float):
        """
        Predicts velocity (u, v) and pressure (p) fields at given spatial coordinates and time t.
        
        Args:
            x (np.ndarray): 1D array or flat points for x-coordinates.
            y (np.ndarray): 1D array or flat points for y-coordinates.
            t (float): Evaluation time timestamp.
            
        Returns:
            dict: Dictionary containing predicted u, v, p arrays.
        """
        x_flat = np.asarray(x, dtype=np.float32).flatten()
        y_flat = np.asarray(y, dtype=np.float32).flatten()
        
        if x_flat.shape != y_flat.shape:
            raise ValueError("x and y input arrays must have the same length.")
            
        t_flat = np.full_like(x_flat, t, dtype=np.float32)
        
        coords = np.stack([x_flat, y_flat, t_flat], axis=1)
        coords_tensor = torch.tensor(coords, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            preds = self.model(coords_tensor).cpu().numpy()
            
        return {
            "u": preds[:, 0],
            "v": preds[:, 1],
            "p": preds[:, 2],
            "velocity_magnitude": np.sqrt(preds[:, 0]**2 + preds[:, 1]**2)
        }

    def predict_vorticity(self, x: np.ndarray, y: np.ndarray, t: float):
        """
        Calculates exact analytical vorticity w = dv/dx - du/dy using Automatic Differentiation.
        """
        x_flat = np.asarray(x, dtype=np.float32).flatten()
        y_flat = np.asarray(y, dtype=np.float32).flatten()
        t_flat = np.full_like(x_flat, t, dtype=np.float32)
        
        coords = np.stack([x_flat, y_flat, t_flat], axis=1)
        coords_tensor = torch.tensor(coords, dtype=torch.float32, requires_grad=True).to(self.device)
        
        preds = self.model(coords_tensor)
        u = preds[:, 0]
        v = preds[:, 1]
        
        # Compute spatial gradients via Autograd (retain_graph=True added here)
        du_dcoords = torch.autograd.grad(u.sum(), coords_tensor, retain_graph=True, create_graph=False)[0]
        dv_dcoords = torch.autograd.grad(v.sum(), coords_tensor, create_graph=False)[0]
        
        du_dy = du_dcoords[:, 1].detach().cpu().numpy()
        dv_dx = dv_dcoords[:, 0].detach().cpu().numpy()
        
        vorticity = dv_dx - du_dy
        return vorticity

def main():
    parser = argparse.ArgumentParser(description="PINN Inference Engine for Navier-Stokes")
    parser.add_argument("--model", type=str, default="final_pinn_model.pth", help="Path to saved model checkpoint")
    parser.add_argument("--x", type=float, default=1.0, help="x coordinate")
    parser.add_argument("--y", type=float, default=0.0, help="y coordinate")
    parser.add_argument("--t", type=float, default=10.0, help="time stamp")
    parser.add_argument("--compute_vorticity", action="store_true", help="Compute vorticity field using autograd")
    parser.add_argument("--out", type=str, default=None, help="Path to output JSON file")
    
    args = parser.parse_args()
    
    engine = PINNInferenceEngine(model_path=args.model)
    
    x_in = np.array([args.x])
    y_in = np.array([args.y])
    
    results = engine.predict_fields(x_in, y_in, args.t)
    out_dict = {
        "query": {"x": args.x, "y": args.y, "t": args.t},
        "predictions": {
            "u": float(results["u"][0]),
            "v": float(results["v"][0]),
            "p": float(results["p"][0]),
            "velocity_magnitude": float(results["velocity_magnitude"][0])
        }
    }
    
    if args.compute_vorticity:
        w = engine.predict_vorticity(x_in, y_in, args.t)
        out_dict["predictions"]["vorticity"] = float(w[0])
        
    print("\n--- PINN Point Inference Result ---")
    print(json.dumps(out_dict, indent=4))
    
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out_dict, f, indent=4)
        print(f"\nResult saved to: {out_path}")

if __name__ == "__main__":
    main()