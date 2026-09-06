import logging
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

from src.models.mlp import NavierStokesMLP

logger = logging.getLogger("Animator")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def create_animation(h5_path: str = "datasets/raissi_cylinder/reference_cylinder.h5", 
                     model_path: str = "final_pinn_model.pth", 
                     save_path: str = "pinn_vortex_street.gif",
                     frames: int = 50):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the spatial domain points
    logger.info("Loading spatial grid...")
    with h5py.File(h5_path, 'r') as h5f:
        # We only need the unique spatial coordinates (x, y) for a single time step
        # The Raissi dataset has 5000 spatial points per time step
        x_ref = np.array(h5f['x']).flatten()
        y_ref = np.array(h5f['y']).flatten()
        
        # Get min and max time to interpolate
        t_all = np.array(h5f['t']).flatten()
        t_min, t_max = t_all.min(), t_all.max()

    # Create a triangulation object once for rendering
    triang = tri.Triangulation(x_ref, y_ref)

    # 2. Load the trained model
    logger.info("Loading trained PINN model...")
    model = NavierStokesMLP().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Generate evenly spaced time steps for the animation
    time_steps = np.linspace(t_min, t_max, frames)

    # 3. Setup the Matplotlib Figure
    logger.info("Initializing animation canvas...")
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True, sharey=True)
    
    fields = ['Velocity U', 'Velocity V', 'Pressure P']
    colormaps = ['RdBu_r', 'RdBu_r', 'viridis']
    
    # Add cylinder patches
    for ax in axes:
        cylinder = Circle((0.0, 0.0), radius=0.5, color='gray', zorder=10)
        ax.add_patch(cylinder)
        ax.set_aspect('equal')
        ax.set_xlim([x_ref.min(), x_ref.max()])
        ax.set_ylim([y_ref.min(), y_ref.max()])

    # 4. Define the Animation Update Function
    def update(frame_idx):
        t_val = time_steps[frame_idx]
        logger.info(f"Rendering frame {frame_idx + 1}/{frames} (t = {t_val:.2f}s)...")
        
        # Create input tensor for current time
        t_array = np.full_like(x_ref, t_val)
        coords = np.stack([x_ref, y_ref, t_array], axis=1)
        coords_tensor = torch.tensor(coords, dtype=torch.float32).to(device)
        
        # PINN Inference
        with torch.no_grad():
            preds = model(coords_tensor).cpu().numpy()
            
        u_pred, v_pred, p_pred = preds[:, 0], preds[:, 1], preds[:, 2]
        predictions = [u_pred, v_pred, p_pred]
        
        # Clear previous contours
        for ax in axes:
            for collection in ax.collections:
                collection.remove()
                
        # Draw new contours
        for i in range(3):
            ax = axes[i]
            ax.tricontourf(triang, predictions[i], levels=40, cmap=colormaps[i])
            ax.set_ylabel(fields[i], fontsize=12, weight='bold')
            
        fig.suptitle(f"PINN Continuous Fluid Dynamics (t = {t_val:.2f}s)", fontsize=16)

    # 5. Compile and Save Animation
    logger.info(f"Generating {frames}-frame animation...")
    ani = FuncAnimation(fig, update, frames=frames, blit=False)
    
    # Save using Pillow
    writer = PillowWriter(fps=10)
    ani.save(save_path, writer=writer)
    logger.info(f"Animation successfully saved to {save_path}")

if __name__ == "__main__":
    # We require the pillow library to save gifs
    try:
        import PIL
    except ImportError:
        logger.error("The 'pillow' library is required to save GIFs. Run: pip install pillow")
        exit(1)
        
    create_animation(frames=40)