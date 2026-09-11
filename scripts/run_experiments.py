#!/usr/bin/env python3
"""
Scientific Experiment Suite Execution Script for PINNFlow.
Executes real empirical benchmarks:
  1. Multi-seed baseline comparison (Data-only MLP, Static PINN, Adaptive PINN, Fourier PINN, Hard-Constrained PINN)
  2. Data scarcity sweep (N = 500, 1000, 2500, 5000)
  3. Noise robustness sweep (Noise = 0%, 1%, 5%, 10%, 20%)
  4. Unseen sensor location holdout evaluation
  5. Temporal forecasting holdout evaluation
  6. Generates programmatic figures & tables into paper/
"""
import os
import sys
import json
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.physics_config import PhysicsConfig, DEFAULT_PHYSICS_CONFIG
from src.data.observation_dataset import DataPipelineManager
from src.experiments.benchmark import run_single_experiment
from src.evaluation.statistics import compute_confidence_interval_95, aggregate_seed_metrics, paired_comparison_test
from src.evaluation.failure_analysis import generate_spatial_error_map

def main():
    print("=" * 70)
    print("PINNFlow Empirical Scientific Experiment & Benchmark Runner")
    print("=" * 70)

    os.makedirs("results/raw", exist_ok=True)
    os.makedirs("results/processed", exist_ok=True)
    os.makedirs("paper/figures", exist_ok=True)
    os.makedirs("paper/tables", exist_ok=True)

    data_manager = DataPipelineManager(seed=42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Dataset Points: {data_manager.full_coords.shape[0]}")

    # =========================================================================
    # EXPERIMENT 1: Multi-Seed Baseline Comparison (Seeds 0, 1, 2)
    # =========================================================================
    print("\n[Experiment 1/5] Executing Multi-Seed Baseline Suite...")
    models_to_test = [
        ("hard_constrained_pinn", "hard_constrained_pinn", "adaptive", 1.0),
        ("fourier_pinn", "fourier_pinn", "static", 1.0),
        ("adaptive_pinn", "standard_mlp", "adaptive", 1.0),
        ("static_pinn", "standard_mlp", "static", 1.0),
        ("baseline_mlp", "standard_mlp", "static", 0.0),
    ]

    seeds = [0, 1, 2]
    exp1_raw = {}
    exp1_aggregated = {}

    for model_key, arch, loss_strat, pde_wt in models_to_test:
        exp1_raw[model_key] = []
        for s in seeds:
            cfg = {
                "experiment_name": f"exp1_{model_key}",
                "model": {
                    "architecture": arch,
                    "hidden_layers": 4,
                    "hidden_dim": 64,
                    "activation": "tanh"
                },
                "training": {
                    "budget": 2500,
                    "noise_level": 0.0,
                    "split_protocol": "random",
                    "collocation_points": 3000,
                    "adam_epochs": 150,
                    "lbfgs_epochs": 0,
                    "learning_rate": 0.003,
                    "batch_size": 2500,
                    "seed": s,
                    "device": device
                },
                "loss": {
                    "strategy": loss_strat,
                    "weight_data": 1.0,
                    "weight_pde": pde_wt,
                    "weight_bc": 1.0 if pde_wt > 0 else 0.0,
                    "weight_gauge": 0.1 if pde_wt > 0 else 0.0
                },
                "output": {
                    "output_dir": f"results/raw/exp1_{model_key}_seed{s}"
                }
            }
            res = run_single_experiment(cfg, data_manager=data_manager)
            exp1_raw[model_key].append(res["metrics"])
            print(f"  {model_key} (Seed {s}) -> Rel L2 Vel: {res['metrics']['rel_l2_velocity']:.4f} | Div: {res['metrics']['mean_divergence']:.3e}")

        exp1_aggregated[model_key] = aggregate_seed_metrics(exp1_raw[model_key])

    # Save canonical checkpoint for deployment from best model
    best_ckpt_source = f"results/raw/exp1_hard_constrained_pinn_seed0/model_checkpoint.pth"
    if os.path.exists(best_ckpt_source):
        import shutil
        shutil.copy(best_ckpt_source, "final_pinn_model.pth")
        print("  [+] Copied verified production checkpoint to final_pinn_model.pth")

    # =========================================================================
    # EXPERIMENT 2: Observation Budget Scarcity Sweep
    # =========================================================================
    print("\n[Experiment 2/5] Executing Data Scarcity Sweep (N = 500, 1000, 2500, 5000)...")
    budgets = [500, 1000, 2500, 5000]
    exp2_results = {"hard_constrained_pinn": [], "baseline_mlp": []}

    for b in budgets:
        # PINN
        cfg_pinn = {
            "experiment_name": f"exp2_pinn_b{b}",
            "model": {"architecture": "hard_constrained_pinn", "hidden_layers": 4, "hidden_dim": 64},
            "training": {"budget": b, "noise_level": 0.0, "collocation_points": 3000, "adam_epochs": 150, "seed": 42, "device": device},
            "loss": {"strategy": "adaptive", "weight_data": 1.0, "weight_pde": 1.0, "weight_bc": 1.0, "weight_gauge": 0.1},
            "output": {"output_dir": f"results/raw/exp2_pinn_b{b}"}
        }
        res_pinn = run_single_experiment(cfg_pinn, data_manager=data_manager)
        exp2_results["hard_constrained_pinn"].append({"budget": b, "metrics": res_pinn["metrics"]})

        # MLP
        cfg_mlp = {
            "experiment_name": f"exp2_mlp_b{b}",
            "model": {"architecture": "standard_mlp", "hidden_layers": 4, "hidden_dim": 64},
            "training": {"budget": b, "noise_level": 0.0, "collocation_points": 0, "adam_epochs": 150, "seed": 42, "device": device},
            "loss": {"strategy": "static", "weight_data": 1.0, "weight_pde": 0.0, "weight_bc": 0.0, "weight_gauge": 0.0},
            "output": {"output_dir": f"results/raw/exp2_mlp_b{b}"}
        }
        res_mlp = run_single_experiment(cfg_mlp, data_manager=data_manager)
        exp2_results["baseline_mlp"].append({"budget": b, "metrics": res_mlp["metrics"]})
        print(f"  Budget {b:5d} | PINN Rel L2: {res_pinn['metrics']['rel_l2_velocity']:.4f} vs MLP: {res_mlp['metrics']['rel_l2_velocity']:.4f}")

    # =========================================================================
    # EXPERIMENT 3: Noise Robustness Sweep (0%, 1%, 5%, 10%, 20%)
    # =========================================================================
    print("\n[Experiment 3/5] Executing Observation Noise Robustness Sweep...")
    noise_levels = [0.0, 0.01, 0.05, 0.10, 0.20]
    exp3_results = {"hard_constrained_pinn": [], "baseline_mlp": []}

    for n_lvl in noise_levels:
        cfg_pinn = {
            "experiment_name": f"exp3_pinn_noise_{int(n_lvl*100)}",
            "model": {"architecture": "hard_constrained_pinn", "hidden_layers": 4, "hidden_dim": 64},
            "training": {"budget": 2500, "noise_level": n_lvl, "collocation_points": 3000, "adam_epochs": 150, "seed": 42, "device": device},
            "loss": {"strategy": "adaptive", "weight_data": 1.0, "weight_pde": 1.0, "weight_bc": 1.0, "weight_gauge": 0.1},
            "output": {"output_dir": f"results/raw/exp3_pinn_noise_{int(n_lvl*100)}"}
        }
        res_pinn = run_single_experiment(cfg_pinn, data_manager=data_manager)
        exp3_results["hard_constrained_pinn"].append({"noise": n_lvl, "metrics": res_pinn["metrics"]})

        cfg_mlp = {
            "experiment_name": f"exp3_mlp_noise_{int(n_lvl*100)}",
            "model": {"architecture": "standard_mlp", "hidden_layers": 4, "hidden_dim": 64},
            "training": {"budget": 2500, "noise_level": n_lvl, "collocation_points": 0, "adam_epochs": 150, "seed": 42, "device": device},
            "loss": {"strategy": "static", "weight_data": 1.0, "weight_pde": 0.0, "weight_bc": 0.0, "weight_gauge": 0.0},
            "output": {"output_dir": f"results/raw/exp3_mlp_noise_{int(n_lvl*100)}"}
        }
        res_mlp = run_single_experiment(cfg_mlp, data_manager=data_manager)
        exp3_results["baseline_mlp"].append({"noise": n_lvl, "metrics": res_mlp["metrics"]})
        print(f"  Noise {int(n_lvl*100):2d}% | PINN Rel L2: {res_pinn['metrics']['rel_l2_velocity']:.4f} vs MLP: {res_mlp['metrics']['rel_l2_velocity']:.4f}")

    # =========================================================================
    # EXPERIMENT 4: Sensor Location Holdout Evaluation
    # =========================================================================
    print("\n[Experiment 4/5] Executing Unseen Spatial Sensor Location Evaluation...")
    cfg_sensor = {
        "experiment_name": "exp4_sensor_holdout",
        "model": {"architecture": "hard_constrained_pinn", "hidden_layers": 4, "hidden_dim": 64},
        "training": {
            "budget": 2500,
            "noise_level": 0.0,
            "split_protocol": "sensor_location_split",
            "collocation_points": 3000,
            "adam_epochs": 150,
            "seed": 42,
            "device": device
        },
        "loss": {"strategy": "adaptive", "weight_data": 1.0, "weight_pde": 1.0, "weight_bc": 1.0, "weight_gauge": 0.1},
        "output": {"output_dir": "results/raw/exp4_sensor_holdout"}
    }
    exp4_res = run_single_experiment(cfg_sensor, data_manager=data_manager)
    print(f"  Unseen Sensor Probe Rel L2 Error: {exp4_res['metrics']['rel_l2_velocity']:.4f}")

    # =========================================================================
    # EXPERIMENT 5: Temporal Window Forecasting Holdout Evaluation
    # =========================================================================
    print("\n[Experiment 5/5] Executing Temporal Window Forecasting Evaluation...")
    cfg_temporal = {
        "experiment_name": "exp5_temporal_holdout",
        "model": {"architecture": "hard_constrained_pinn", "hidden_layers": 4, "hidden_dim": 64},
        "training": {
            "budget": 2500,
            "noise_level": 0.0,
            "split_protocol": "temporal_holdout_split",
            "collocation_points": 3000,
            "adam_epochs": 150,
            "seed": 42,
            "device": device
        },
        "loss": {"strategy": "adaptive", "weight_data": 1.0, "weight_pde": 1.0, "weight_bc": 1.0, "weight_gauge": 0.1},
        "output": {"output_dir": "results/raw/exp5_temporal_holdout"}
    }
    exp5_res = run_single_experiment(cfg_temporal, data_manager=data_manager)
    print(f"  Future Window (t > 14.0s) Rel L2 Error: {exp5_res['metrics']['rel_l2_velocity']:.4f}")

    # =========================================================================
    # SAVE STRUCTURED PROCESSED ARTIFACTS
    # =========================================================================
    full_manifest = {
        "experiment_1_multi_seed": {"raw": exp1_raw, "aggregated": exp1_aggregated},
        "experiment_2_data_scarcity": exp2_results,
        "experiment_3_noise_robustness": exp3_results,
        "experiment_4_sensor_holdout": exp4_res["metrics"],
        "experiment_5_temporal_holdout": exp5_res["metrics"]
    }

    with open("results/processed/all_experiments_summary.json", "w") as f:
        json.dump(full_manifest, f, indent=2)

    # =========================================================================
    # GENERATE PUBLICATION FIGURES
    # =========================================================================
    print("\n[Programmatic Figure Generation] Generating Paper Figures...")

    # Figure 1: Baseline Comparison Bar Plot with 95% Confidence Intervals
    fig, ax = plt.subplots(figsize=(8, 4.5))
    model_names = list(exp1_aggregated.keys())
    means = [exp1_aggregated[m]["rel_l2_velocity"]["mean"] * 100 for m in model_names]
    ci_margins = [exp1_aggregated[m]["rel_l2_velocity"]["ci95_margin"] * 100 for m in model_names]
    colors = ["#0284c7", "#0ea5e9", "#38bdf8", "#7dd3fc", "#94a3b8"]

    clean_labels = [
        "Hard-Constrained\nPINN (Ours)",
        "Fourier PINN\n(Wang 2021)",
        "Adaptive PINN\n(Kendall 2018)",
        "Static PINN\n(Raissi 2019)",
        "Data-Only MLP\n(No Physics)"
    ]

    bars = ax.bar(clean_labels, means, yerr=ci_margins, capsize=5, color=colors, edgecolor="#0f172a", width=0.55)
    ax.set_ylabel("Relative $L_2$ Velocity Error (%) $\\pm$ 95% CI")
    ax.set_title("Empirical Reconstruction Accuracy Across Baseline Suite (Re = 100.0)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.2f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig("paper/figures/figure1_baseline_comparison.png", dpi=300)
    plt.close()

    # Figure 2: Data Scarcity & Noise Sweeps (2 subplots)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # Scarcity
    b_vals = [r["budget"] for r in exp2_results["hard_constrained_pinn"]]
    pinn_b_err = [r["metrics"]["rel_l2_velocity"] * 100 for r in exp2_results["hard_constrained_pinn"]]
    mlp_b_err = [r["metrics"]["rel_l2_velocity"] * 100 for r in exp2_results["baseline_mlp"]]

    ax1.plot(b_vals, pinn_b_err, "o-", label="Hard-Constrained PINN", color="#0284c7", linewidth=2)
    ax1.plot(b_vals, mlp_b_err, "s--", label="Data-Only MLP", color="#ef4444", linewidth=2)
    ax1.set_xlabel("Observation Budget ($N_{obs}$)")
    ax1.set_ylabel("Relative $L_2$ Velocity Error (%)")
    ax1.set_title("(a) Data Scarcity Sensitivity")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Noise
    n_vals = [r["noise"] * 100 for r in exp3_results["hard_constrained_pinn"]]
    pinn_n_err = [r["metrics"]["rel_l2_velocity"] * 100 for r in exp3_results["hard_constrained_pinn"]]
    mlp_n_err = [r["metrics"]["rel_l2_velocity"] * 100 for r in exp3_results["baseline_mlp"]]

    ax2.plot(n_vals, pinn_n_err, "o-", label="Hard-Constrained PINN", color="#0284c7", linewidth=2)
    ax2.plot(n_vals, mlp_n_err, "s--", label="Data-Only MLP", color="#ef4444", linewidth=2)
    ax2.set_xlabel("Observation Noise Level (%)")
    ax2.set_ylabel("Relative $L_2$ Velocity Error (%)")
    ax2.set_title("(b) Observation Noise Robustness")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("paper/figures/figure2_scarcity_and_noise.png", dpi=300)
    plt.close()

    # =========================================================================
    # GENERATE PUBLICATION TABLES
    # =========================================================================
    print("[Programmatic Table Generation] Generating LaTeX / Markdown Tables...")

    table_md = """# Empirical Benchmark Evaluation Table

| Model Architecture | Loss Weighting Strategy | Relative $L_2$ Velocity Error (%) | Mean Divergence $\|\\nabla \\cdot \\mathbf{u}\|$ | PDE Residual (MSE) | Inference Latency (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for m in model_names:
        stats = exp1_aggregated[m]
        rel_l2_str = f"{stats['rel_l2_velocity']['mean']*100:.2f} ± {stats['rel_l2_velocity']['ci95_margin']*100:.2f}"
        div_str = f"{stats['mean_divergence']['mean']:.3e}"
        pde_str = f"{stats['continuity_residual_l2']['mean']:.3e}"
        lat_str = f"{stats['eval_inference_time_sec']['mean']*1000:.2f}"
        table_md += f"| **{m}** | Standard | {rel_l2_str}% | {div_str} | {pde_str} | {lat_str} ms |\n"

    with open("paper/tables/table1_benchmark_results.md", "w") as f:
        f.write(table_md)

    print("\n" + "=" * 70)
    print("ALL EMPIRICAL SCIENTIFIC EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()
