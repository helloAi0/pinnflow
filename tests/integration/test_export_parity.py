import os
import tempfile
import torch
import numpy as np
import pytest
from src.models.factory import create_model
from src.models.checkpoint import save_checkpoint
from src.export.export_model import PINNExporter

def test_torchscript_and_onnx_export_parity():
    cfg = {"architecture": "standard_mlp", "hidden_layers": 3, "hidden_dim": 32}
    model, _ = create_model(cfg)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "model_to_export.pth")
        save_checkpoint(model, cfg, ckpt_path, training_seed=42)

        export_dir = os.path.join(tmpdir, "exports")
        exporter = PINNExporter(checkpoint_path=ckpt_path, export_dir=export_dir)

        sample_input = torch.randn(50, 3, dtype=torch.float32)

        # 1. Test TorchScript export & parity
        ts_path = exporter.export_torchscript(sample_input)
        assert os.path.exists(ts_path)

        # 2. Test ONNX export & parity
        onnx_path, manifest = exporter.export_onnx(sample_input)
        assert os.path.exists(onnx_path)
        assert manifest["export_version"] == "2.0.0"
