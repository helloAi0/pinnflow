import os
import sys
import time
import json
import logging
import argparse
import yaml
import torch
import numpy as np
from typing import Dict, Any, List, Optional
from torch.utils.data import DataLoader

from src.config.physics_config import PhysicsConfig, DEFAULT_PHYSICS_CONFIG
from src.data.observation_dataset import DataPipelineManager, FlowDataset
from src.models.factory import create_model
from src.models.checkpoint import save_checkpoint, get_git_commit_sha
from src.losses.weighting import UnifiedPINNLoss, DynamicGradientWeighting
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.statistics import aggregate_seed_metrics, paired_comparison_test
from src.evaluation.failure_analysis import generate_spatial_error_map

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PINNBenchmark")

def sample_collocation_points(
    num_points: int,
    physics_config: PhysicsConfig = DEFAULT_PHYSICS_CONFIG,
    device: torch.device = torch.device("cpu"),
    seed: int = 42
) -> torch.Tensor:
    """Samples spatio-temporal collocation points uniformly across the observation domain."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(physics_config.x_min, physics_config.x_max, size=(num_points, 1))
    y = rng.uniform(physics_config.y_min, physics_config.y_max, size=(num_points, 1))
    t = rng.uniform(physics_config.t_min, physics_config.t_max, size=(num_points, 1))
    coords = np.hstack([x, y, t]).astype(np.float32)
    return torch.tensor(coords, dtype=torch.float32, device=device, requires_grad=True)

def sample_boundary_points(
    num_points: int,
    physics_config: PhysicsConfig = DEFAULT_PHYSICS_CONFIG,
    device: torch.device = torch.device("cpu"),
    seed: int = 42
) -> torch.Tensor:
    """Samples points on the cylinder perimeter surface."""
    rng = np.random.RandomState(seed)
    theta = rng.uniform(0, 2 * np.pi, size=(num_points, 1))
    x = physics_config.cylinder_center[0] + physics_config.cylinder_radius * np.cos(theta)
    y = physics_config.cylinder_center[1] + physics_config.cylinder_radius * np.sin(theta)
    t = rng.uniform(physics_config.t_min, physics_config.t_max, size=(num_points, 1))
    coords = np.hstack([x, y, t]).astype(np.float32)
    return torch.tensor(coords, dtype=torch.float32, device=device)

def run_single_experiment(
    config: Dict[str, Any],
    physics_config: PhysicsConfig = DEFAULT_PHYSICS_CONFIG,
    data_manager: Optional[DataPipelineManager] = None
) -> Dict[str, Any]:
    """
    Executes a single scientific PINN training and evaluation experiment.
    """
    experiment_id = f"{config.get('experiment_name', 'exp')}_seed{config['training'].get('seed', 42)}_{int(time.time())}"
    logger.info(f"=== Starting Experiment: {experiment_id} ===")
    
    # 1. Device selection
    dev_str = config["training"].get("device", "auto")
    if dev_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(dev_str)
    logger.info(f"Execution device: {device}")

    # 2. Seed determinism
    seed = int(config["training"].get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 3. Data preparation
    if data_manager is None:
        data_manager = DataPipelineManager(seed=seed)

    budget = int(config["training"].get("budget", 5000))
    noise_level = float(config["training"].get("noise_level", 0.0))
    protocol = config["training"].get("split_protocol", "random")

    splits = data_manager.get_split(
        protocol=protocol,
        budget=budget,
        noise_level=noise_level,
        seed=seed,
        val_ratio=float(config["training"].get("val_ratio", 0.1)),
        test_ratio=float(config["training"].get("test_ratio", 0.2))
    )

    train_ds = splits["train"]
    test_ds = splits["test"]
    val_ds = splits.get("val", test_ds)

    train_loader = DataLoader(train_ds, batch_size=min(budget, int(config["training"].get("batch_size", 5000))), shuffle=True)

    # 4. Collocation points & Boundary points
    n_coll = int(config["training"].get("collocation_points", 10000))
    pde_coords = sample_collocation_points(n_coll, physics_config=physics_config, device=device, seed=seed) if n_coll > 0 else torch.empty((0, 3), device=device)
    cyl_coords = sample_boundary_points(min(1000, n_coll // 10), physics_config=physics_config, device=device, seed=seed) if n_coll > 0 else torch.empty((0, 3), device=device)

    # 5. Model instantiation
    model_config = config.get("model", {"architecture": "standard_mlp"})
    model, model_meta = create_model(model_config, physics_config=physics_config)
    model.to(device)

    # 6. Loss engine
    loss_cfg = config.get("loss", {})
    loss_strategy = loss_cfg.get("strategy", "static")
    loss_module = UnifiedPINNLoss(
        physics_config=physics_config,
        strategy=loss_strategy,
        weight_data=float(loss_cfg.get("weight_data", 1.0)),
        weight_pde=float(loss_cfg.get("weight_pde", 1.0)),
        weight_bc=float(loss_cfg.get("weight_bc", 1.0)),
        weight_gauge=float(loss_cfg.get("weight_gauge", 0.1))
    ).to(device)

    # Dynamic gradient weighting baseline support
    grad_weighting = DynamicGradientWeighting(alpha=0.1) if loss_strategy == "gradient_weighted" else None

    # 7. Optimizer setup (Adam Stage)
    lr = float(config["training"].get("learning_rate", 1e-3))
    trainable_params = list(model.parameters())
    if loss_strategy == "adaptive":
        trainable_params.extend(list(loss_module.adaptive_weighting.parameters()))

    adam_optimizer = torch.optim.Adam(trainable_params, lr=lr)
    adam_epochs = int(config["training"].get("adam_epochs", 500))
    
    start_train_time = time.perf_counter()
    loss_history = []

    model.train()
    for epoch in range(1, adam_epochs + 1):
        for data_c, data_f in train_loader:
            data_c, data_f = data_c.to(device), data_f.to(device)
            adam_optimizer.zero_grad()

            loss, metrics_step, residuals = loss_module(
                model=model,
                data_coords=data_c,
                data_targets=data_f,
                pde_coords=pde_coords,
                cylinder_coords=cyl_coords
            )

            # Adaptive sampling resampling step (RAD)
            if config.get("experiment_name") == "adaptive_sampling_pinn" and epoch % config["training"].get("resample_every", 200) == 0:
                with torch.no_grad():
                    res_mag = (residuals["continuity"]**2 + residuals["momentum_x"]**2 + residuals["momentum_y"]**2).flatten()
                    top_k_idx = torch.topk(res_mag, k=min(n_coll // 4, res_mag.shape[0])).indices
                    refined_points = pde_coords[top_k_idx] + torch.randn_like(pde_coords[top_k_idx]) * 0.05
                    pde_coords = torch.cat([pde_coords, refined_points], dim=0).detach().requires_grad_(True)

            loss.backward()
            adam_optimizer.step()

        if epoch % max(1, adam_epochs // 5) == 0 or epoch == 1:
            loss_history.append({"epoch": epoch, "loss": float(loss.item())})
            logger.info(f"Epoch {epoch:04d}/{adam_epochs:04d} | Total Loss: {loss.item():.4e}")

    # Optional L-BFGS fine-tuning stage
    lbfgs_epochs = int(config["training"].get("lbfgs_epochs", 0))
    if lbfgs_epochs > 0:
        logger.info(f"Starting L-BFGS refinement ({lbfgs_epochs} steps)...")
        # Extract full data batch for L-BFGS
        full_data_c = train_ds.coords.to(device)
        full_data_f = train_ds.fields.to(device)

        lbfgs_opt = torch.optim.LBFGS(
            trainable_params,
            lr=0.1,
            max_iter=20,
            history_size=50,
            line_search_fn="strong_wolfe"
        )

        def closure():
            lbfgs_opt.zero_grad()
            l_val, _, _ = loss_module(
                model=model,
                data_coords=full_data_c,
                data_targets=full_data_f,
                pde_coords=pde_coords,
                cylinder_coords=cyl_coords
            )
            l_val.backward()
            return l_val

        for step in range(lbfgs_epochs):
            lbfgs_opt.step(closure)

    training_duration = time.perf_counter() - start_train_time
    logger.info(f"Training completed in {training_duration:.2f}s")

    # 8. Evaluation on clean holdout test set
    test_coords = test_ds.coords
    test_fields = test_ds.fields
    eval_metrics = compute_all_metrics(
        model=model,
        eval_coords=test_coords,
        eval_targets=test_fields,
        physics_config=physics_config,
        device=device
    )
    eval_metrics["training_time_sec"] = training_duration

    logger.info(f"Test Relative L2 Velocity Error: {eval_metrics['rel_l2_velocity']:.6f}")
    logger.info(f"Test Mean Divergence Error:     {eval_metrics['mean_divergence']:.6e}")

    # 9. Output persistence & Checkpointing
    out_dir = config.get("output", {}).get("output_dir", f"results/raw/{experiment_id}")
    os.makedirs(out_dir, exist_ok=True)

    ckpt_path = os.path.join(out_dir, "model_checkpoint.pth")
    ckpt_sha = save_checkpoint(
        model=model,
        model_config=model_config,
        filepath=ckpt_path,
        training_seed=seed,
        reynolds_number=physics_config.reynolds_number,
        metrics=eval_metrics,
        extra_metadata={"config": config}
    )

    # 10. Generate failure analysis / error map plot
    fig_path = os.path.join(out_dir, "spatial_error_map.png")
    generate_spatial_error_map(
        model=model,
        X_star=data_manager.X_star,
        U_star=data_manager.U_star,
        p_star=data_manager.p_star,
        time_val=10.0,
        save_path=fig_path,
        physics_config=physics_config
    )

    # Experiment summary record
    result_record = {
        "experiment_id": experiment_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": get_git_commit_sha(),
        "model_architecture": model_meta["architecture"],
        "parameter_count": model_meta["parameter_count"],
        "seed": seed,
        "budget": budget,
        "noise_level": noise_level,
        "split_protocol": protocol,
        "checkpoint_path": ckpt_path,
        "checkpoint_sha256": ckpt_sha,
        "training_time_sec": training_duration,
        "metrics": eval_metrics,
        "config": config
    }

    record_path = os.path.join(out_dir, "results.json")
    with open(record_path, "w") as f:
        json.dump(result_record, f, indent=2)

    logger.info(f"Experiment results saved to {record_path}")
    return result_record

def run_benchmark_matrix(
    matrix_config_path: str = "configs/benchmark.yaml",
    seeds: Optional[List[int]] = None,
    budgets: Optional[List[int]] = None,
    noise_levels: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Executes full multi-seed baseline matrix and computes statistical aggregations.
    """
    with open(matrix_config_path, "r") as f:
        matrix_cfg = yaml.safe_load(f)

    data_manager = DataPipelineManager(seed=42)
    models_to_run = matrix_cfg["matrices"]["models"]
    seeds_to_run = seeds or matrix_cfg["matrices"].get("seeds", [0, 1, 2])
    budgets_to_run = budgets or [5000]
    noise_to_run = noise_levels or [0.0]

    all_raw_results: List[Dict[str, Any]] = []
    grouped_by_model: Dict[str, List[Dict[str, float]]] = {}

    for model_name in models_to_run:
        grouped_by_model[model_name] = []
        for seed in seeds_to_run:
            # Construct specific experiment config
            exp_config = {
                "experiment_name": f"benchmark_{model_name}",
                "model": {
                    "architecture": "fourier_pinn" if "fourier" in model_name else ("hard_constrained_pinn" if "constrained" in model_name else "standard_mlp"),
                    "hidden_layers": matrix_cfg["defaults"]["hidden_layers"],
                    "hidden_dim": matrix_cfg["defaults"]["hidden_dim"],
                    "activation": matrix_cfg["defaults"]["activation"]
                },
                "training": {
                    "budget": budgets_to_run[0],
                    "noise_level": noise_to_run[0],
                    "split_protocol": "random",
                    "collocation_points": matrix_cfg["defaults"]["collocation_points"],
                    "adam_epochs": matrix_cfg["defaults"]["adam_epochs"],
                    "lbfgs_epochs": matrix_cfg["defaults"]["lbfgs_epochs"],
                    "learning_rate": matrix_cfg["defaults"]["learning_rate"],
                    "seed": seed,
                    "device": matrix_cfg["defaults"]["device"]
                },
                "loss": {
                    "strategy": "adaptive" if "adaptive_pinn" in model_name else ("gradient_weighted" if "gradient" in model_name else "static"),
                    "weight_data": 1.0,
                    "weight_pde": 0.0 if "baseline_mlp" in model_name else 1.0,
                    "weight_bc": 1.0,
                    "weight_gauge": 0.1
                },
                "output": {
                    "output_dir": f"results/raw/{model_name}_seed{seed}"
                }
            }

            res = run_single_experiment(exp_config, data_manager=data_manager)
            all_raw_results.append(res)
            grouped_by_model[model_name].append(res["metrics"])

    # Statistical aggregation
    aggregated_results = {}
    for model_name, runs in grouped_by_model.items():
        aggregated_results[model_name] = aggregate_seed_metrics(runs)

    # Save summary
    out_dir = matrix_cfg.get("output_dir", "results/processed")
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "benchmark_summary.json")
    
    with open(summary_path, "w") as f:
        json.dump({
            "raw": all_raw_results,
            "aggregated": aggregated_results
        }, f, indent=2)

    logger.info(f"=== Multi-seed Benchmark Matrix Complete ===")
    logger.info(f"Summary persisted to {summary_path}")
    return aggregated_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PINNFlow Scientific Benchmark Runner")
    parser.add_argument("--config", type=str, default="configs/smoke.yaml", help="Path to experiment YAML config")
    parser.add_argument("--matrix", action="store_true", help="Run full benchmark matrix across baseline suite")
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    args = parser.parse_args()

    if args.matrix:
        run_benchmark_matrix()
    else:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        run_single_experiment(cfg)
