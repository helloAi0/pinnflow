#!/usr/bin/env python3
"""
Standalone end-to-end smoke test script for PINNFlow.
Validates:
  1. Data loading
  2. Model training on mini-budget
  3. Evaluation metrics calculation
  4. Checkpoint saving and integrity verification
  5. API query test via TestClient
"""
import os
import sys
import tempfile
import yaml
import torch

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starlette.testclient import TestClient

from src.experiments.benchmark import run_single_experiment
from src.data.observation_dataset import DataPipelineManager
from src.models.checkpoint import load_and_validate_checkpoint
from src.api.main import app
import src.api.main as api_main

def main():
    print("=" * 60)
    print("PINNFlow End-to-End Scientific Smoke Test")
    print("=" * 60)

    # 1. Load data pipeline
    print("\n[Step 1/5] Initializing dataset pipeline...")
    data_mgr = DataPipelineManager(seed=42)
    print(f"  Total spatio-temporal points: {data_mgr.full_coords.shape[0]}")
    print(f"  Dataset SHA256: {data_mgr.dataset_sha256[:16]}...")

    # 2. Fast training run
    print("\n[Step 2/5] Training tiny PINN model (smoke config)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        smoke_cfg = {
            "experiment_name": "smoke_run",
            "model": {
                "architecture": "standard_mlp",
                "hidden_layers": 3,
                "hidden_dim": 32,
                "activation": "tanh"
            },
            "training": {
                "budget": 200,
                "noise_level": 0.0,
                "split_protocol": "random",
                "val_ratio": 0.1,
                "test_ratio": 0.2,
                "collocation_points": 500,
                "adam_epochs": 15,
                "lbfgs_epochs": 0,
                "learning_rate": 0.005,
                "batch_size": 200,
                "seed": 42,
                "device": "cpu"
            },
            "loss": {
                "strategy": "static",
                "weight_data": 1.0,
                "weight_pde": 1.0,
                "weight_bc": 0.0,
                "weight_gauge": 0.0
            },
            "output": {
                "output_dir": tmpdir
            }
        }

        results = run_single_experiment(smoke_cfg, data_manager=data_mgr)
        ckpt_path = results["checkpoint_path"]
        print(f"  Training finished in {results['training_time_sec']:.2f}s")
        print(f"  Test Rel L2 Velocity Error: {results['metrics']['rel_l2_velocity']:.6f}")
        print(f"  Saved checkpoint to: {ckpt_path}")

        # 3. Checkpoint verification
        print("\n[Step 3/5] Validating saved checkpoint integrity...")
        is_valid, model, meta, err = load_and_validate_checkpoint(ckpt_path)
        assert is_valid, f"Checkpoint validation failed: {err}"
        print(f"  Checkpoint verified! SHA: {meta['checkpoint_sha'][:16]}...")

        # 4. Bind to API state and query
        print("\n[Step 4/5] Testing FastAPI endpoints against verified checkpoint...")
        api_main.MODEL = model
        api_main.MODEL_METADATA = meta
        api_main.CHECKPOINT_VALID = True

        client = TestClient(app)

        # Query health
        h_res = client.get("/api/v1/health")
        assert h_res.status_code == 200, f"Health check failed: {h_res.text}"
        h_json = h_res.json()
        assert h_json["status"] == "ok"
        assert h_json["trained"] is True
        print(f"  /api/v1/health response: status={h_json['status']}, arch={h_json['architecture']}")

        # Query point prediction
        p_res = client.post("/api/v1/predict/point", json={"x": 2.0, "y": 0.0, "t": 5.0, "compute_vorticity": True})
        assert p_res.status_code == 200, f"Point prediction failed: {p_res.text}"
        p_json = p_res.json()
        print(f"  /api/v1/predict/point: u={p_json['u']:.4f}, v={p_json['v']:.4f}, p={p_json['p']:.4f}")

        # Query field prediction
        f_res = client.post("/api/v1/predict/field", json={
            "x_min": 1.0, "x_max": 4.0, "y_min": -1.0, "y_max": 1.0, "nx": 20, "ny": 10, "t": 5.0
        })
        assert f_res.status_code == 200, f"Field prediction failed: {f_res.text}"
        f_json = f_res.json()
        print(f"  /api/v1/predict/field: grid={len(f_json['u'])}x{len(f_json['u'][0])}, pde_loss={f_json['metrics']['pde_loss']:.4e}")

    print("\n[Step 5/5] Smoke test PASSED successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
