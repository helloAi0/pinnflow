import logging
from typing import Dict
import torch
from torch.utils.data import DataLoader
from src.models.mlp import NavierStokesMLP
from src.physics.navier_stokes import NavierStokes2D
from src.losses.adaptive_pinn_loss import AdaptivePINNLoss
from src.geometry.domain import CylinderDomain
from src.sampling.collocation import CollocationSampler
from src.data.observation_dataset import DataPipelineManager

logger = logging.getLogger("AdaptivePINNTrainer")

class AdaptivePINNTrainer:
    def __init__(self, 
                 model: torch.nn.Module, 
                 pde_solver: NavierStokes2D, 
                 loss_fn: AdaptivePINNLoss, 
                 lr: float = 1e-3, 
                 device: str = "cpu"):
        self.model = model.to(device)
        self.pde_solver = pde_solver
        self.loss_fn = loss_fn.to(device)
        self.device = device
        
        # CRITICAL: Optimize BOTH model weights and the learnable loss parameters
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.loss_fn.parameters()), 
            lr=lr
        )

    def train_epoch(self, data_loader: DataLoader, collocation_pts: torch.Tensor) -> Dict[str, float]:
        self.model.train()
        accumulated_losses = {
            "total_loss": 0.0, "data_loss": 0.0, "physics_loss": 0.0,
            "w_data": 0.0, "w_cont": 0.0
        }
        
        n_batches = len(data_loader)
        coll_batch_size = max(1, collocation_pts.shape[0] // n_batches)

        for i, (coords, targets) in enumerate(data_loader):
            coords = coords.to(self.device)
            targets = targets.to(self.device)

            coll_slice = collocation_pts[i * coll_batch_size:(i + 1) * coll_batch_size].to(self.device)
            coll_slice.requires_grad_(True)

            self.optimizer.zero_grad()

            data_preds = self.model(coords)
            coll_preds = self.model(coll_slice)
            pde_res = self.pde_solver.compute_residuals(coll_slice, coll_preds)

            loss, components = self.loss_fn(data_preds, targets, pde_res)

            loss.backward()
            self.optimizer.step()

            for key in accumulated_losses:
                accumulated_losses[key] += components[key]

        return {k: v / n_batches for k, v in accumulated_losses.items()}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    manager = DataPipelineManager(seed=42)
    datasets = manager.prepare_experiment_data(train_budget=1000, noise_level=0.0)
    train_loader = DataLoader(datasets["train"], batch_size=256, shuffle=True)

    domain = CylinderDomain()
    sampler = CollocationSampler(domain, seed=42)
    coll_points = sampler.sample_lhs(10000)

    model = NavierStokesMLP()
    pde_solver = NavierStokes2D(reynolds_number=100.0)
    loss_fn = AdaptivePINNLoss()
    
    trainer = AdaptivePINNTrainer(model, pde_solver, loss_fn, lr=1e-3, device=device)

    logger.info(f"Starting Adaptive PINN (Model C) test run on {device}...")
    for epoch in range(1, 6):
        metrics = trainer.train_epoch(train_loader, coll_points)
        logger.info(
            f"Epoch {epoch:02d} | Total: {metrics['total_loss']:.4e} | "
            f"Data MSE: {metrics['data_loss']:.4e} | PDE Res: {metrics['physics_loss']:.4e} | "
            f"w_data: {metrics['w_data']:.3f} | w_cont: {metrics['w_cont']:.3f}"
        )