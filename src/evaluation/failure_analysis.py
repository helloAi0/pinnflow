import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, Any, Optional
from src.config.physics_config import PhysicsConfig, DEFAULT_PHYSICS_CONFIG

def generate_spatial_error_map(
    model: torch.nn.Module,
    X_star: np.ndarray,
    U_star: np.ndarray,
    p_star: np.ndarray,
    time_val: float = 10.0,
    save_path: str = "paper/figures/figure_spatial_error_map.png",
    physics_config: PhysicsConfig = DEFAULT_PHYSICS_CONFIG
) -> Dict[str, float]:
    """
    Generates a spatial absolute error heatmap at a given time snapshot.
    Saves publication-quality figure.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    model.eval()

    num_spatial = X_star.shape[0]
    t_col = np.full((num_spatial, 1), time_val, dtype=np.float32)
    query_coords = np.hstack([X_star, t_col])
    
    with torch.no_grad():
        coords_t = torch.from_numpy(query_coords).float()
        preds = model(coords_t).cpu().numpy()

    u_pred = preds[:, 0]
    v_pred = preds[:, 1]
    p_pred = preds[:, 2]

    # Find closest time snapshot index
    t_idx = int(np.clip(int(time_val / 0.1), 0, U_star.shape[2] - 1))
    u_true = U_star[:, 0, t_idx]
    v_true = U_star[:, 1, t_idx]
    p_true = p_star[:, t_idx]

    err_u = np.abs(u_pred - u_true)
    err_v = np.abs(v_pred - v_true)
    err_vel = np.sqrt(err_u ** 2 + err_v ** 2)
    err_p = np.abs(p_pred - p_true)

    # Wake partition: near-cylinder (x < 3.0) vs far wake (x >= 3.0)
    near_cyl_mask = (X_star[:, 0] < 3.0)
    far_wake_mask = (X_star[:, 0] >= 3.0)

    near_cyl_vel_err = float(np.mean(err_vel[near_cyl_mask]))
    far_wake_vel_err = float(np.mean(err_vel[far_wake_mask]))
    mean_vel_err = float(np.mean(err_vel))

    # Matplotlib publication-quality plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # 1. True Velocity Magnitude
    vel_true_mag = np.sqrt(u_true**2 + v_true**2)
    sc0 = axes[0].scatter(X_star[:, 0], X_star[:, 1], c=vel_true_mag, cmap="viridis", s=6, vmin=0, vmax=1.5)
    axes[0].set_title(r"Reference DNS Velocity Magnitude $|\mathbf{u}_{DNS}|$ ($t = " + f"{time_val:.1f}" + r"$)")
    axes[0].set_ylabel("y")
    plt.colorbar(sc0, ax=axes[0], label=r"$|\mathbf{u}|$")

    # 2. Predicted Velocity Magnitude
    vel_pred_mag = np.sqrt(u_pred**2 + v_pred**2)
    sc1 = axes[1].scatter(X_star[:, 0], X_star[:, 1], c=vel_pred_mag, cmap="viridis", s=6, vmin=0, vmax=1.5)
    axes[1].set_title(r"PINN Predicted Velocity Magnitude $|\hat{\mathbf{u}}|$")
    axes[1].set_ylabel("y")
    plt.colorbar(sc1, ax=axes[1], label=r"$|\hat{\mathbf{u}}|$")

    # 3. Absolute Pointwise Error
    sc2 = axes[2].scatter(X_star[:, 0], X_star[:, 1], c=err_vel, cmap="inferno", s=6)
    axes[2].set_title(r"Pointwise Velocity Error $|\hat{\mathbf{u}} - \mathbf{u}_{DNS}|$")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    plt.colorbar(sc2, ax=axes[2], label="Error")

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(physics_config.x_min, physics_config.x_max)
        ax.set_ylim(physics_config.y_min, physics_config.y_max)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return {
        "mean_velocity_error": mean_vel_err,
        "near_cylinder_error": near_cyl_vel_err,
        "far_wake_error": far_wake_vel_err,
        "figure_saved": save_path
    }