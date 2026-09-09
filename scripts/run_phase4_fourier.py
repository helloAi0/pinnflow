import os
import sys
import torch
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.reproducibility import set_seed
from src.config.physics_config import PhysicsConfig
from src.models.fourier_network import FourierConstrainedPINN
from src.data.pipeline import CylinderDataPipeline
from src.utils.softadapt import SoftAdapt
from scripts.run_phase2_adaptive import compute_navier_stokes_loss

def run_fourier_experiments():
    seeds = [42, 108, 314, 2718, 8008]
    config_obj = PhysicsConfig()
    kinematic_viscosity = 1.0 / config_obj.reynolds_number
    
    hyperparams = {
        "learning_rate": 1e-3,
        "epochs": 1000,
        "layer_topology": [3] + [200] * 8 + [3],
        "activation": "tanh",
        "reynolds_number": config_obj.reynolds_number,
        "strategy": "SoftAdapt + HardConstraints + Fourier",
        "cylinder_radius": 0.5,
        "fourier_features": 64,  # Expands 3D input to 128D embedded space
        "fourier_sigma": 1.0     # Controls the frequency spread
    }
    
    pipeline = CylinderDataPipeline(data_dir="data")
    data_dict = pipeline.load_and_preprocess(observation_budget=2000)
    
    for seed in seeds:
        print(f"\n{'='*40}\nStarting Fourier Run | Seed: {seed}\n{'='*40}")
        set_seed(seed)
        
        run = wandb.init(
            project="pinnflow-phase4",
            name=f"fourier-seed-{seed}",
            config={**hyperparams, "seed": seed},
            reinit=True
        )
        
        model = FourierConstrainedPINN(
            layers=hyperparams["layer_topology"], 
            activation_type=hyperparams["activation"],
            cylinder_radius=hyperparams["cylinder_radius"],
            fourier_features=hyperparams["fourier_features"],
            sigma=hyperparams["fourier_sigma"]
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
        balancer = SoftAdapt(num_losses=2, beta=0.1)
        
        coords = data_dict["train_coords"].to(model.device)
        targets = data_dict["train_fields"].to(model.device)
        
        for epoch in range(hyperparams["epochs"]):
            optimizer.zero_grad()
            
            preds = model(coords)
            
            loss_data = torch.mean((preds - targets).pow(2))
            loss_pde = compute_navier_stokes_loss(model, coords, kinematic_viscosity)
            
            weights = balancer.get_weights().to(model.device)
            w_data, w_pde = weights[0], weights[1]
            
            total_loss = (w_data * loss_data) + (w_pde * loss_pde)
            total_loss.backward()
            optimizer.step()
            
            balancer.update([loss_data, loss_pde])
            
            if epoch % 200 == 0:
                wandb.log({
                    "epoch": epoch,
                    "total_loss": total_loss.item(),
                    "data_loss": loss_data.item(),
                    "pde_loss": loss_pde.item(),
                    "w_data": w_data.item(),
                    "w_pde": w_pde.item()
                })
                print(f"Epoch {epoch}/{hyperparams['epochs']} | Loss: {total_loss.item():.4e}")
                
        wandb.log({
            "final_total_loss": total_loss.item(),
            "final_data_loss": loss_data.item(),
            "final_pde_loss": loss_pde.item()
        })
        
        # Save model weights for potential inference/visualization later
        torch.save(model.state_dict(), f"wandb/fourier_model_seed_{seed}.pth")
        run.finish()

if __name__ == "__main__":
    run_fourier_experiments()