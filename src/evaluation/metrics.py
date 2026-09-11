import time
import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional
from src.physics.derivatives import compute_first_derivatives
from src.config.physics_config import PhysicsConfig, DEFAULT_PHYSICS_CONFIG
from src.losses.pde_loss import PDELoss

def relative_l2_error(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Relative L2 norm error: ||pred - true||_2 / (||true||_2 + eps)
    Direction: Lower is better (0.0 is perfect).
    """
    diff_norm = torch.linalg.norm(pred - true)
    true_norm = torch.linalg.norm(true)
    return float((diff_norm / (true_norm + eps)).item())

def l_infinity_error(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    L-infinity (maximum absolute discrepancy) error: max |pred - true|
    Direction: Lower is better (0.0 is perfect).
    """
    return float(torch.max(torch.abs(pred - true)).item())

def compute_all_metrics(
    model: torch.nn.Module,
    eval_coords: torch.Tensor,
    eval_targets: torch.Tensor,
    physics_config: PhysicsConfig = DEFAULT_PHYSICS_CONFIG,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 20000
) -> Dict[str, float]:
    """
    Computes complete research-grade scientific validation metrics over an evaluation dataset.
    All outputs include clear units and directions.
    """
    model.eval()
    eval_coords = eval_coords.to(device)
    eval_targets = eval_targets.to(device)
    
    # 1. Inference Latency Benchmark
    start_time = time.perf_counter()
    with torch.no_grad():
        preds_list = []
        for i in range(0, eval_coords.shape[0], batch_size):
            batch = eval_coords[i:i + batch_size]
            preds_list.append(model(batch))
        all_preds = torch.cat(preds_list, dim=0)
    inference_time = time.perf_counter() - start_time
    latency_per_sample_us = (inference_time / eval_coords.shape[0]) * 1e6

    u_pred, v_pred, p_pred = all_preds[:, 0:1], all_preds[:, 1:2], all_preds[:, 2:3]
    u_true, v_true, p_true = eval_targets[:, 0:1], eval_targets[:, 1:2], eval_targets[:, 2:3]

    vel_pred = torch.cat([u_pred, v_pred], dim=1)
    vel_true = torch.cat([u_true, v_true], dim=1)

    # 2. Velocity & Pressure Relative Errors
    rel_l2_u = relative_l2_error(u_pred, u_true)
    rel_l2_v = relative_l2_error(v_pred, v_true)
    rel_l2_p = relative_l2_error(p_pred, p_true)
    rel_l2_vel = relative_l2_error(vel_pred, vel_true)

    linf_u = l_infinity_error(u_pred, u_true)
    linf_v = l_infinity_error(v_pred, v_true)
    linf_p = l_infinity_error(p_pred, p_true)
    linf_vel = l_infinity_error(vel_pred, vel_true)

    # 3. Autograd Physical Residuals on a representative evaluation subset
    pde_loss_module = PDELoss(physics_config=physics_config)
    sub_sample_size = min(5000, eval_coords.shape[0])
    sub_coords = eval_coords[:sub_sample_size].clone().detach().requires_grad_(True)
    sub_preds = model(sub_coords)
    
    residuals = pde_loss_module.compute_residuals(sub_coords, sub_preds)
    
    cont_res_norm = float(torch.linalg.norm(residuals["continuity"]).item() / np.sqrt(sub_sample_size))
    mom_x_res_norm = float(torch.linalg.norm(residuals["momentum_x"]).item() / np.sqrt(sub_sample_size))
    mom_y_res_norm = float(torch.linalg.norm(residuals["momentum_y"]).item() / np.sqrt(sub_sample_size))
    mean_divergence = float(torch.mean(torch.abs(residuals["continuity"])).item())
    vorticity_mean = float(torch.mean(torch.abs(residuals["vorticity"])).item())

    # Count parameters
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "rel_l2_velocity": rel_l2_vel,
        "rel_l2_u": rel_l2_u,
        "rel_l2_v": rel_l2_v,
        "rel_l2_p": rel_l2_p,
        "linf_velocity": linf_vel,
        "linf_u": linf_u,
        "linf_v": linf_v,
        "linf_p": linf_p,
        "continuity_residual_l2": cont_res_norm,
        "momentum_x_residual_l2": mom_x_res_norm,
        "momentum_y_residual_l2": mom_y_res_norm,
        "mean_divergence": mean_divergence,
        "vorticity_magnitude_mean": vorticity_mean,
        "latency_per_sample_us": latency_per_sample_us,
        "eval_inference_time_sec": inference_time,
        "parameter_count": param_count
    }
