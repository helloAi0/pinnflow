import logging
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from src.models.mlp import NavierStokesMLP

logger = logging.getLogger("ErrorMapping")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def generate_error_maps(h5_path="datasets/raissi_cylinder/reference_cylinder.h5", 
                        model_path="final_pinn_model.pth", 
                        time_idx=100, 
                        save_path="spatial_error_map.png"):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load exact reference data
    logger.info(f"Loading reference dataset from {h5_path}...")
    with h5py.File(h5_path, 'r') as h5f:
        x_raw = np.array(h5f['x']).squeeze()
        y_raw = np.array(h5f['y']).squeeze()
        t_raw = np.array(h5f['t']).squeeze()
        u_raw = np.array(h5f['u'])
        v_raw = np.array(h5f['v'])
        p_raw = np.array(h5f['p'])

    # Ensure time_idx stays within dataset bounds
    time_idx = time_idx % len(t_raw)
    t_val = t_raw[time_idx]
    logger.info(f"Extracting time snapshot index {time_idx} (t = {t_val:.2f}s)...")

    # Handle 2D matrix shapes vs 1D flattened shapes
    if u_raw.ndim == 2:
        x_snap = x_raw
        y_snap = y_raw
        t_snap = np.full_like(x_snap, t_val)
        u_exact = u_raw[:, time_idx]
        v_exact = v_raw[:, time_idx]
        p_exact = p_raw[:, time_idx]
    else:
        idx = np.isclose(t_raw, t_val, atol=1e-3)
        x_snap = x_raw[idx]
        y_snap = y_raw[idx]
        t_snap = t_raw[idx]
        u_exact = u_raw[idx]
        v_exact = v_raw[idx]
        p_exact = p_raw[idx]

    logger.info(f"Extracted snapshot with {len(x_snap)} spatial points.")

    # 2. Load Trained PINN Model
    logger.info("Loading trained PINN model...")
    model = NavierStokesMLP().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Predict Fields via Forward Pass
    logger.info("Running inference on snapshot domain...")
    coords = np.stack([x_snap, y_snap, t_snap], axis=1)
    coords_tensor = torch.tensor(coords, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        preds = model(coords_tensor).cpu().numpy()
        
    u_pred = preds[:, 0]
    v_pred = preds[:, 1]
    p_pred = preds[:, 2]

    # 4. Calculate Pointwise Absolute Errors
    logger.info("Calculating spatial error distributions...")
    err_u = np.abs(u_exact - u_pred)
    err_v = np.abs(v_exact - v_pred)
    err_p = np.abs(p_exact - p_pred)

    # 5. Render Heatmaps
    logger.info("Rendering high-contrast error heatmaps...")
    triang = tri.Triangulation(x_snap, y_snap)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True, sharey=True)
    
    errors = [err_u, err_v, err_p]
    titles = ['Velocity U Absolute Error', 'Velocity V Absolute Error', 'Pressure P Absolute Error']
    
    for i, ax in enumerate(axes):
        # 'magma' colormap: darker colors represent low error, bright yellow is high error
        contour = ax.tricontourf(triang, errors[i], levels=50, cmap='magma')
        fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04, label="Absolute Error")
        
        # Superimpose the cylinder geometry boundary
        circle = plt.Circle((0, 0), 0.5, color='white', zorder=10, fill=False, linewidth=2)
        ax.add_patch(circle)
        
        ax.set_aspect('equal')
        ax.set_title(titles[i], weight='bold')
        ax.set_xlim([x_snap.min(), x_snap.max()])
        ax.set_ylim([y_snap.min(), y_snap.max()])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Error maps saved successfully to {save_path}")

if __name__ == "__main__":
    generate_error_maps()