import os
import sys
import torch
import wandb
import numpy as np

# Force Python to recognize your root pinnflow project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.reproducibility import set_seed
from src.config.physics_config import PhysicsConfig
from src.models.network import NavierStokesPINN
from src.data.pipeline import CylinderDataPipeline

def run_baseline_experiments():
    seeds = [42, 108, 314, 2718, 8008]
    project_name = "pinnflow-phase1"
    
    # Configuration boundaries setup
    config_obj = PhysicsConfig()
    hyperparams = {
        "learning_rate": 1e-3,
        "epochs": 1000,
        # Change from: [200] * 8
        "layer_topology": [3] + [200] * 8 + [3],  # [Input: x,y,t] -> 8x200 hidden -> [Output: u,v,p]
        "activation": "tanh",
        "reynolds_number": config_obj.reynolds_number
    }
    
    # Initialize your data pipeline
    pipeline = CylinderDataPipeline(data_dir="data")
    data_dict = pipeline.load_and_preprocess(observation_budget=2000)
    
    for seed in seeds:
        print(f"\n{'='*40}\nStarting Run | Seed: {seed}\n{'='*40}")
        
        # 1. Lock seed before initializing the neural network layers
        set_seed(seed)
        
        # 2. Boot up W&B Logger context
        run = wandb.init(
            project=project_name,
            name=f"baseline-raissi-seed-{seed}",
            config={**hyperparams, "seed": seed},
            reinit=True
        )
        
        # 3. Model instantiation aligned with active parameters
        model = NavierStokesPINN(layers=hyperparams["layer_topology"], activation_type=hyperparams["activation"])
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
        
        # 4. Step through training execution
        coords = data_dict["train_coords"].to(model.device)
        targets = data_dict["train_fields"].to(model.device)
        
        for epoch in range(hyperparams["epochs"]):
            optimizer.zero_grad()
            preds = model(coords)
            
            # Formulating data losses
            loss_data = torch.mean((preds - targets).pow(2))
            loss_pde = torch.tensor(0.0).to(model.device) 
            total_loss = loss_data + loss_pde
            
            total_loss.backward()
            optimizer.step()
            
            # Log metrics to dashboard in steps
            if epoch % 200 == 0:
                wandb.log({
                    "epoch": epoch,
                    "total_loss": total_loss.item(),
                    "data_loss": loss_data.item(),
                    "pde_loss": loss_pde.item()
                })
                print(f"Epoch {epoch}/{hyperparams['epochs']} | Loss: {total_loss.item():.4e}")
                
        # Final parameters save registration
        wandb.log({
            "final_total_loss": total_loss.item(),
            "final_data_loss": loss_data.item(),
            "final_pde_loss": loss_pde.item()
        })
        run.finish()

if __name__ == "__main__":
    run_baseline_experiments()
