import os
import tempfile
import pytest
import yaml
from src.experiments.benchmark import run_single_experiment
from src.data.observation_dataset import DataPipelineManager

def test_fast_training_pipeline_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "experiment_name": "test_integration_train",
            "model": {
                "architecture": "standard_mlp",
                "hidden_layers": 2,
                "hidden_dim": 16,
                "activation": "tanh"
            },
            "training": {
                "budget": 200,
                "noise_level": 0.0,
                "split_protocol": "random",
                "val_ratio": 0.1,
                "test_ratio": 0.2,
                "collocation_points": 300,
                "adam_epochs": 10,
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

        data_manager = DataPipelineManager(seed=42)
        results = run_single_experiment(config, data_manager=data_manager)

        assert results["metrics"]["rel_l2_velocity"] >= 0.0
        assert os.path.exists(results["checkpoint_path"])
        assert os.path.exists(os.path.join(tmpdir, "results.json"))
        assert os.path.exists(os.path.join(tmpdir, "spatial_error_map.png"))
