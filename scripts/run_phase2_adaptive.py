import os
import sys
import torch
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.reproducibility import set_seed
from src.config.physics_config import PhysicsConfig
from src.models.network import NavierStokesPINN
from src.data.pipeline import CylinderDataPipeline
from src.utils.softadapt import SoftAdapt

def compute_navier_stokes_loss(model, coords, nu):
    """Computes the physics residuals. Requires coords to have requires_grad=True."""
    coords.requires_grad_(True)
    preds = model(coords)
    
    u, v, p = preds[:, 0:1], preds[:, 1:2], preds[:, 2:3]
    x, y, t = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]
    
    # 1st order derivatives (u)
    u_g = torch.autograd.grad(u, coords, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x, u_y, u_t = u_g[:, 0:1], u_g[:, 1:2], u_g[:, 2:3]
    
    # 1st order derivatives (v)
    v_g = torch.autograd.grad(v, coords, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_x, v_y, v_t = v_g[:, 0:1], v_g[:, 1:2], v_g[:, 2:3]
    
    # 1st order derivatives (p)
    p_g = torch.autograd.grad(p, coords, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_x, p_y = p_g[:, 0:1], p_g[:, 1:2]
    
    # 2nd order derivatives (Laplacians)
    u_xx = torch.autograd.grad(u_x, coords, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, coords, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    
    v_xx = torch.autograd.grad(v_x, coords, grad_outputs=torch.ones_like(v_x), create_graph=True)[0][:, 0:1]
    v_yy = torch.autograd.grad(v_y, coords, grad_outputs=torch.ones_like(v_y), create_graph=True)[0][:, 1:2]
    
    # Navier-Stokes Equations
    f_u = u_t + u*u_x + v*u_y + p_x - nu*(u_xx + u_yy)
    f_v = v_t + u*v_x + v*v_y + p_y - nu*(v_xx + v_yy)
    f_mass = u_x + v_y  # Continuity equation
    
    return torch.mean(f_u**2) + torch.mean(f_v**2) + torch.mean(f_mass**2)

def run_adaptive_experiments():
    seeds = [42, 108, 314, 2718, 8008]
    config_obj = PhysicsConfig()
    kinematic_viscosity = 1.0 / config_obj.reynolds_number
    
    hyperparams = {
        "learning_rate": 1e-3,
        "epochs": 1000,
        "layer_topology": [3] + [200] * 8 + [3],
        "activation": "tanh",
        "reynolds_number": config_obj.reynolds_number,
        "strategy": "SoftAdapt",
        "beta": 0.1
    }
    
    pipeline = CylinderDataPipeline(data_dir="data")
    data_dict = pipeline.load_and_preprocess(observation_budget=2000)
    
    for seed in seeds:
        print(f"\n{'='*40}\nStarting Adaptive Run | Seed: {seed}\n{'='*40}")
        set_seed(seed)
        
        run = wandb.init(
            project="pinnflow-phase2",
            name=f"adaptive-softadapt-seed-{seed}",
            config={**hyperparams, "seed": seed},
            reinit=True
        )
        
        model = NavierStokesPINN(layers=hyperparams["layer_topology"], activation_type=hyperparams["activation"])
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
        
        # Initialize the dynamic loss balancer
        balancer = SoftAdapt(num_losses=2, beta=hyperparams["beta"])
        
        coords = data_dict["train_coords"].to(model.device)
        targets = data_dict["train_fields"].to(model.device)
        
        for epoch in range(hyperparams["epochs"]):
            optimizer.zero_grad()
            
            # Forward pass & Data Loss
            preds = model(coords)
            loss_data = torch.mean((preds - targets).pow(2))
            
            # Physics Loss
            loss_pde = compute_navier_stokes_loss(model, coords, kinematic_viscosity)
            
            # Dynamic Weighting
            weights = balancer.get_weights().to(model.device)
            w_data, w_pde = weights[0], weights[1]
            
            total_loss = (w_data * loss_data) + (w_pde * loss_pde)
            
            total_loss.backward()
            optimizer.step()
            
            # Update balancer at the end of the epoch
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
                print(f"Epoch {epoch}/{hyperparams['epochs']} | "
                      f"W_Data: {w_data.item():.2f}, W_PDE: {w_pde.item():.2f} | "
                      f"Loss: {total_loss.item():.4e}")
                
        wandb.log({
            "final_total_loss": total_loss.item(),
            "final_data_loss": loss_data.item(),
            "final_pde_loss": loss_pde.item()
        })
        run.finish()

if __name__ == "__main__":
    run_adaptive_experiments()