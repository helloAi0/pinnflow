import logging
import torch
from torch.utils.data import DataLoader

from src.data.observation_dataset import DataPipelineManager
from src.geometry.domain import CylinderDomain
from src.sampling.collocation import CollocationSampler
from src.models.mlp import NavierStokesMLP
from src.physics.navier_stokes import NavierStokes2D
from src.losses.static_pinn_loss import StaticPINNLoss
from src.losses.adaptive_pinn_loss import AdaptivePINNLoss

from src.training.baseline_trainer import BaselineTrainer
from src.training.standard_pinn_trainer import StandardPINNTrainer
from src.training.adaptive_pinn_trainer import AdaptivePINNTrainer

logger = logging.getLogger("ExperimentMatrix")

def evaluate_model(model: torch.nn.Module, dataloader: DataLoader, device: str) -> float:
    """Computes pure data MSE on the validation or test set."""
    model.eval()
    criterion = torch.nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for coords, targets in dataloader:
            coords, targets = coords.to(device), targets.to(device)
            preds = model(coords)
            total_loss += criterion(preds, targets).item() * coords.size(0)
    return total_loss / len(dataloader.dataset)

def run_experiment_matrix():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 10 # Set low for pipeline verification; scale to 2000+ for actual research
    
    budgets = [500, 5000]
    noise_levels = [0.0, 0.05]
    
    manager = DataPipelineManager(seed=42)
    domain = CylinderDomain()
    sampler = CollocationSampler(domain, seed=42)
    
    # Pre-sample collocation points to ensure consistency across all PINN models
    coll_points = sampler.sample_lhs(10000)
    
    results = []

    for noise in noise_levels:
        for budget in budgets:
            logger.info(f"\n{'='*50}\nSTARTING EXP: Budget={budget}, Noise={noise*100}%\n{'='*50}")
            
            datasets = manager.prepare_experiment_data(train_budget=budget, noise_level=noise)
            train_loader = DataLoader(datasets["train"], batch_size=256, shuffle=True)
            test_loader = DataLoader(datasets["test"], batch_size=1024, shuffle=False)
            
            # --- Model A: Baseline MLP ---
            logger.info("Training Model A: Pure Data-Driven MLP")
            model_a = NavierStokesMLP().to(device)
            trainer_a = BaselineTrainer(model_a, lr=1e-3, device=device)
            for _ in range(epochs):
                trainer_a.train_epoch(train_loader)
            test_mse_a = evaluate_model(model_a, test_loader, device)
            
            # --- Model B: Static PINN ---
            logger.info("Training Model B: Static PINN (Weights=1.0)")
            model_b = NavierStokesMLP().to(device)
            pde_solver = NavierStokes2D(reynolds_number=100.0)
            loss_b = StaticPINNLoss()
            trainer_b = StandardPINNTrainer(model_b, pde_solver, loss_b, lr=1e-3, device=device)
            for _ in range(epochs):
                trainer_b.train_epoch(train_loader, coll_points)
            test_mse_b = evaluate_model(model_b, test_loader, device)

            # --- Model C: Adaptive PINN ---
            logger.info("Training Model C: Adaptive PINN (Learnable Weights)")
            model_c = NavierStokesMLP().to(device)
            loss_c = AdaptivePINNLoss()
            trainer_c = AdaptivePINNTrainer(model_c, pde_solver, loss_c, lr=1e-3, device=device)
            for _ in range(epochs):
                trainer_c.train_epoch(train_loader, coll_points)
            test_mse_c = evaluate_model(model_c, test_loader, device)

            # Record
            results.append({
                "budget": budget,
                "noise": noise,
                "MSE_A": test_mse_a,
                "MSE_B": test_mse_b,
                "MSE_C": test_mse_c
            })

    # Print Summary Matrix
    print("\n" + "="*60)
    print(f"{'Budget':<8} | {'Noise':<6} | {'Model A (MLP)':<13} | {'Model B (Static)':<16} | {'Model C (Adaptive)':<16}")
    print("-" * 60)
    for r in results:
        print(f"{r['budget']:<8} | {r['noise']:<6} | {r['MSE_A']:<13.4e} | {r['MSE_B']:<16.4e} | {r['MSE_C']:<16.4e}")
    print("="*60 + "\n")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run_experiment_matrix()