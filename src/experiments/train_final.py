import logging
import torch
from torch.utils.data import DataLoader

from src.data.observation_dataset import DataPipelineManager
from src.geometry.domain import CylinderDomain
from src.sampling.collocation import CollocationSampler
from src.models.mlp import NavierStokesMLP
from src.physics.navier_stokes import NavierStokes2D
from src.losses.adaptive_pinn_loss import AdaptivePINNLoss
from src.training.adaptive_pinn_trainer import AdaptivePINNTrainer
from src.visualization.plot_fields import FieldVisualizer

logger = logging.getLogger("FinalTraining")

def run_final_training():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Execution Hyperparameters 
    # (Set low for demonstration; production typically uses 5000+ Adam and 50+ L-BFGS epochs)
    adam_epochs = 500   
    lbfgs_epochs = 5   
    budget = 5000
    noise = 0.0
    
    # 1. Prepare Data & Collocation
    manager = DataPipelineManager(seed=42)
    datasets = manager.prepare_experiment_data(train_budget=budget, noise_level=noise)
    
    # L-BFGS requires the full batch in memory for accurate Hessian approximation
    train_loader = DataLoader(datasets["train"], batch_size=budget, shuffle=True) 
    
    domain = CylinderDomain()
    sampler = CollocationSampler(domain, seed=42)
    coll_points = sampler.sample_lhs(10000).to(device)
    
    # Extract full dataset tensors for the L-BFGS closure function
    full_coords, full_targets = next(iter(train_loader))
    full_coords, full_targets = full_coords.to(device), full_targets.to(device)

    # 2. Initialize Architecture
    model = NavierStokesMLP().to(device)
    pde_solver = NavierStokes2D(reynolds_number=100.0)
    loss_fn = AdaptivePINNLoss().to(device)
    
    # ==========================================
    # STAGE 1: Adam Optimizer (Global Exploration)
    # ==========================================
    logger.info(f"--- STAGE 1: Adam Optimization ({adam_epochs} epochs) ---")
    trainer = AdaptivePINNTrainer(model, pde_solver, loss_fn, lr=1e-3, device=device)
    
    for epoch in range(1, adam_epochs + 1):
        metrics = trainer.train_epoch(train_loader, coll_points)
        if epoch % 100 == 0 or epoch == 1:
            logger.info(f"Adam Epoch {epoch:04d} | Total: {metrics['total_loss']:.4e} | PDE Res: {metrics['physics_loss']:.4e}")

    # ==========================================
    # STAGE 2: L-BFGS Optimizer (Local Fine-Tuning)
    # ==========================================
    logger.info("--- STAGE 2: L-BFGS Optimization ---")
    lbfgs_optimizer = torch.optim.LBFGS(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=0.1,
        max_iter=20,          # Internal line-search iterations per step
        max_eval=25,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=50,
        line_search_fn="strong_wolfe"
    )
    
    lbfgs_iter = 0
    def closure():
        nonlocal lbfgs_iter
        lbfgs_optimizer.zero_grad()
        
        data_preds = model(full_coords)
        coll_preds = model(coll_points)
        pde_res = pde_solver.compute_residuals(coll_points, coll_preds)
        
        loss, components = loss_fn(data_preds, full_targets, pde_res)
        loss.backward()
        
        if lbfgs_iter % 10 == 0:
            logger.info(f"L-BFGS Iter {lbfgs_iter:03d} | Total: {components['total_loss']:.4e} | PDE Res: {components['physics_loss']:.4e}")
        lbfgs_iter += 1
        return loss
        
    for epoch in range(lbfgs_epochs):
        lbfgs_optimizer.step(closure)

    # ==========================================
    # STAGE 3: Save Final Weights & Visualize
    # ==========================================
    model_path = "final_pinn_model.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"--- STAGE 3: Model Saved to {model_path} ---")
    
    logger.info("Generating Visualization from Trained Model...")
    visualizer = FieldVisualizer()
    visualizer.plot_snapshot(model, time_idx=100, device=device, save_path="trained_pinn_snapshot.png")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    run_final_training()