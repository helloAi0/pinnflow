import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.physics_config import PhysicsConfig
from src.models.fourier_network import FourierConstrainedPINN
from src.data.pipeline import CylinderDataPipeline

def plot_field(ax, x, y, field, title, cmap='RdBu_r'):
    """Helper to plot fluid fields using unstructured triangular grids."""
    # Add a tiny epsilon to the max value to prevent errors if the field is uniform
    levels = np.linspace(field.min(), field.max() + 1e-8, 50)
    
    im = ax.tricontourf(x, y, field, levels=levels, cmap=cmap, extend='both')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    
    # Add cylinder boundary for visual context
    circle = plt.Circle((0, 0), 0.5, color='black', fill=False, linewidth=1.5)
    ax.add_patch(circle)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    return im

def run_inference_and_visualize(seed=8008, time_snapshot_idx=100):
    config = PhysicsConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("[*] Loading full dataset for evaluation...")
    pipeline = CylinderDataPipeline(data_dir="data")
    raw_data = pipeline.load_and_preprocess(observation_budget=int(1e9)) 
    
    # Extract data for a specific time snapshot
    t_val = np.unique(raw_data['train_coords'][:, 2].numpy())[time_snapshot_idx]
    mask = raw_data['train_coords'][:, 2] == t_val
    
    coords_t = raw_data['train_coords'][mask].to(device)
    exact_fields = raw_data['train_fields'][mask].numpy()
    
    print(f"[*] Loading Phase 4 weights for seed {seed}...")
    model = FourierConstrainedPINN(
        layers=[3] + [200] * 8 + [3],
        activation_type="tanh",
        cylinder_radius=0.5,
        fourier_features=64,
        sigma=1.0
    ).to(device)
    
    weight_path = f"wandb/fourier_model_seed_{seed}.pth"
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    
    print("[*] Running network inference...")
    with torch.no_grad():
        preds = model(coords_t).cpu().numpy()
        
    # Extract 1D spatial coordinates for scattered plotting
    x = coords_t[:, 0].cpu().numpy()
    y = coords_t[:, 1].cpu().numpy()

    # Keep target and prediction arrays in their 1D shapes 
    exact_u = exact_fields[:, 0]
    exact_v = exact_fields[:, 1]
    exact_p = exact_fields[:, 2]
    
    pred_u = preds[:, 0]
    pred_v = preds[:, 1]
    pred_p = preds[:, 2]
    
    err_u = np.abs(exact_u - pred_u)
    err_v = np.abs(exact_v - pred_v)
    err_p = np.abs(exact_p - pred_p)
    
    print("[*] Generating contour plots via Delaunay triangulation...")
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(f"Flow Field Visualization at t={t_val:.2f} (Seed: {seed})", fontsize=16)
    
    # U-Velocity
    plot_field(axes[0, 0], x, y, exact_u, "Exact U-Velocity")
    plot_field(axes[0, 1], x, y, pred_u, "Predicted U-Velocity")
    plot_field(axes[0, 2], x, y, err_u, "Absolute Error U", cmap='magma')
    
    # V-Velocity
    plot_field(axes[1, 0], x, y, exact_v, "Exact V-Velocity")
    plot_field(axes[1, 1], x, y, pred_v, "Predicted V-Velocity")
    plot_field(axes[1, 2], x, y, err_v, "Absolute Error V", cmap='magma')
    
    # Pressure
    plot_field(axes[2, 0], x, y, exact_p, "Exact Pressure")
    plot_field(axes[2, 1], x, y, pred_p, "Predicted Pressure")
    plot_field(axes[2, 2], x, y, err_p, "Absolute Error P", cmap='magma')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("flow_visualization.png", dpi=300)
    print("[+] Visualization saved to flow_visualization.png")
    plt.show()

if __name__ == "__main__":
    run_inference_and_visualize(seed=8008)