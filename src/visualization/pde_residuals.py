import logging
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from src.models.mlp import NavierStokesMLP

logger = logging.getLogger("PDEResiduals")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def calculate_physics_residuals(h5_path="datasets/raissi_cylinder/reference_cylinder.h5", 
                                model_path="final_pinn_model.pth", 
                                time_idx=100, 
                                nu=0.01,
                                save_path="pde_residuals_map.png"):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load domain coordinates (We don't need exact u, v, p for this!)
    logger.info("Loading spatial grid...")
    with h5py.File(h5_path, 'r') as h5f:
        x_raw = np.array(h5f['x']).squeeze()
        y_raw = np.array(h5f['y']).squeeze()
        t_raw = np.array(h5f['t']).squeeze()

    time_idx = time_idx % len(t_raw)
    t_val = t_raw[time_idx]
    
    # Use the 5000 spatial points for the given time snapshot
    x_snap = x_raw
    y_snap = y_raw
    t_snap = np.full_like(x_snap, t_val)

    # Convert to individual tensors with requires_grad=True for autograd
    x_t = torch.tensor(x_snap, dtype=torch.float32, requires_grad=True).unsqueeze(1).to(device)
    y_t = torch.tensor(y_snap, dtype=torch.float32, requires_grad=True).unsqueeze(1).to(device)
    t_t = torch.tensor(t_snap, dtype=torch.float32, requires_grad=True).unsqueeze(1).to(device)
    
    # 2. Load Model
    logger.info("Loading trained PINN model...")
    model = NavierStokesMLP().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Forward Pass
    coords = torch.cat([x_t, y_t, t_t], dim=1)
    preds = model(coords)
    
    u = preds[:, 0:1]
    v = preds[:, 1:2]
    p = preds[:, 2:3]

    # 4. Automatic Differentiation (Computing Gradients)
    logger.info("Computing exact derivatives via autograd...")
    
    # First derivatives for u
    du = torch.autograd.grad(u, coords, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x, u_y, u_t = du[:, 0:1], du[:, 1:2], du[:, 2:3]
    
    # First derivatives for v
    dv = torch.autograd.grad(v, coords, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_x, v_y, v_t = dv[:, 0:1], dv[:, 1:2], dv[:, 2:3]
    
    # First derivatives for p
    dp = torch.autograd.grad(p, coords, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_x, p_y = dp[:, 0:1], dp[:, 1:2]
    
    # Second derivatives for u
    du_x = torch.autograd.grad(u_x, coords, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_xx = du_x[:, 0:1]
    du_y = torch.autograd.grad(u_y, coords, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    u_yy = du_y[:, 1:2]
    
    # Second derivatives for v
    dv_x = torch.autograd.grad(v_x, coords, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_xx = dv_x[:, 0:1]
    dv_y = torch.autograd.grad(v_y, coords, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    v_yy = dv_y[:, 1:2]

    # 5. Compute PDE Residuals
    logger.info("Calculating Navier-Stokes physical residuals...")
    
    # Continuity (Mass Conservation): u_x + v_y = 0
    f_c = u_x + v_y
    
    # Momentum X: u_t + u*u_x + v*u_y + p_x - nu*(u_xx + u_yy) = 0
    f_u = u_t + (u * u_x) + (v * u_y) + p_x - nu * (u_xx + u_yy)
    
    # Momentum Y: v_t + u*v_x + v*v_y + p_y - nu*(v_xx + v_yy) = 0
    f_v = v_t + (u * v_x) + (v * v_y) + p_y - nu * (v_xx + v_yy)

    # Detach and convert to numpy for plotting
    f_c_np = np.abs(f_c.detach().cpu().numpy().flatten())
    f_u_np = np.abs(f_u.detach().cpu().numpy().flatten())
    f_v_np = np.abs(f_v.detach().cpu().numpy().flatten())

    # 6. Render Residual Maps
    logger.info("Rendering physics violation heatmaps...")
    triang = tri.Triangulation(x_snap, y_snap)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True, sharey=True)
    
    residuals = [f_c_np, f_u_np, f_v_np]
    titles = ['Continuity Residual $|f_c|$', 'Momentum X Residual $|f_u|$', 'Momentum Y Residual $|f_v|$']
    
    for i, ax in enumerate(axes):
        # 'inferno' highlights extreme physics violations
        contour = ax.tricontourf(triang, residuals[i], levels=50, cmap='inferno')
        fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04, label="Residual Magnitude")
        
        # Superimpose the cylinder geometry boundary
        circle = plt.Circle((0, 0), 0.5, color='white', zorder=10, fill=False, linewidth=2)
        ax.add_patch(circle)
        
        ax.set_aspect('equal')
        ax.set_title(titles[i], weight='bold')
        ax.set_xlim([x_snap.min(), x_snap.max()])
        ax.set_ylim([y_snap.min(), y_snap.max()])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Physics residual maps saved successfully to {save_path}")

if __name__ == "__main__":
    calculate_physics_residuals()