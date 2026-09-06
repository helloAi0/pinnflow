import logging
from typing import Dict
import torch
from torch.utils.data import DataLoader
from src.models.mlp import NavierStokesMLP
from src.physics.navier_stokes import NavierStokes2D
from src.losses.static_pinn_loss import StaticPINNLoss
from src.geometry.domain import CylinderDomain
from src.sampling.collocation import CollocationSampler
from src.data.observation_dataset import DataPipelineManager

logger = logging.getLogger("StandardPINNTrainer")

class StandardPINNTrainer:
    def __init__(self, 
                 model: torch.nn.Module, 
                 pde_solver: NavierStokes2D, 
                 loss_fn: StaticPINNLoss, 
                 lr: float = 1e-3, 
                 device: str = "cpu"):
        self.model = model.to(device)
        self.pde_solver = pde_solver
        self.loss_fn = loss_fn.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train_epoch(self, data_loader: DataLoader, collocation_pts: torch.Tensor) -> Dict[str, float]:
        self.model.train()
        accumulated_losses = {
            "total_loss": 0.0, "data_loss": 0.0, "physics_loss": 0.0,
            "continuity_loss": 0.0, "momentum_x_loss": 0.0, "momentum_y_loss": 0.0
        }
        
        n_batches = len(data_loader)
        # Sample collocation batch slice per data batch
        coll_batch_size = max(1, collocation_pts.shape[0] // n_batches)

        for i, (coords, targets) in enumerate(data_loader):
            coords = coords.to(self.device)
            targets = targets.to(self.device)

            # Collocation batch slice
            coll_slice = collocation_pts[i * coll_batch_size:(i + 1) * coll_batch_size].to(self.device)
            coll_slice.requires_grad_(True)

            self.optimizer.zero_grad()

            # 1. Supervised Data Forward Pass
            data_preds = self.model(coords)

            # 2. PDE Collocation Forward Pass & Derivatives
            coll_preds = self.model(coll_slice)
            pde_res = self.pde_solver.compute_residuals(coll_slice, coll_preds)

            # 3. Static Loss Computation
            loss, components = self.loss_fn(data_preds, targets, pde_res)

            loss.backward()
            self.optimizer.step()

            for key in accumulated_losses:
                accumulated_losses[key] += components[key]

        return {k: v / n_batches for k, v in accumulated_losses.items()}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Setup Data & Collocation
    manager = DataPipelineManager(seed=42)
    datasets = manager.prepare_experiment_data(train_budget=1000, noise_level=0.0)
    train_loader = DataLoader(datasets["train"], batch_size=256, shuffle=True)

    domain = CylinderDomain()
    sampler = CollocationSampler(domain, seed=42)
    coll_points = sampler.sample_lhs(10000)

    # 2. Setup Physics & Models
    model = NavierStokesMLP()
    pde_solver = NavierStokes2D(reynolds_number=100.0)
    loss_fn = StaticPINNLoss(w_data=1.0, w_cont=1.0, w_mom_u=1.0, w_mom_v=1.0)
    
    trainer = StandardPINNTrainer(model, pde_solver, loss_fn, lr=1e-3, device=device)

    logger.info(f"Starting Standard PINN (Baseline B) test run on {device}...")
    for epoch in range(1, 6):
        metrics = trainer.train_epoch(train_loader, coll_points)
        logger.info(
            f"Epoch {epoch:02d} | Total: {metrics['total_loss']:.4e} | "
            f"Data MSE: {metrics['data_loss']:.4e} | PDE Res: {metrics['physics_loss']:.4e}"
        )