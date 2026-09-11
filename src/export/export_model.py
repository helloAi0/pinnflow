import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

from src.models.checkpoint import load_and_validate_checkpoint
from src.config.physics_config import PhysicsConfig, DEFAULT_PHYSICS_CONFIG

logger = logging.getLogger("PINNExporter")

class PINNExporter:
    """
    Exports canonical trained PINN models to TorchScript JIT and ONNX runtime formats
    with strict numerical parity verification.
    """
    def __init__(
        self,
        checkpoint_path: str = "final_pinn_model.pth",
        export_dir: str = "exports",
        physics_config: PhysicsConfig = DEFAULT_PHYSICS_CONFIG
    ):
        self.device = torch.device("cpu")
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = checkpoint_path

        # Load and validate canonical checkpoint
        is_valid, model, metadata, err = load_and_validate_checkpoint(checkpoint_path, device=self.device, physics_config=physics_config)
        if not is_valid or model is None:
            raise ValueError(f"Cannot export unverified checkpoint: {err}")

        self.model = model
        self.model.eval()
        self.metadata = metadata

    def export_torchscript(self, sample_input: torch.Tensor, filename: str = "pinn_model.pt") -> Path:
        """Exports model to TorchScript via tracing and validates output parity."""
        output_path = self.export_dir / filename
        traced = torch.jit.trace(self.model, sample_input)
        traced.save(str(output_path))
        
        # Verify parity
        with torch.no_grad():
            py_out = self.model(sample_input).numpy()
            ts_model = torch.jit.load(str(output_path))
            ts_out = ts_model(sample_input).numpy()

        max_diff = float(np.max(np.abs(py_out - ts_out)))
        logger.info(f"TorchScript exported to {output_path} (Max diff: {max_diff:.8e})")
        assert max_diff < 1e-5, f"TorchScript parity check failed: max_diff={max_diff:.8e} >= 1e-5"
        return output_path

    def export_onnx(
        self,
        sample_input: torch.Tensor,
        filename: str = "pinn_model.onnx",
        opset_version: int = 14
    ) -> Tuple[Path, Dict[str, Any]]:
        """Exports model to ONNX format with dynamic batch sizing."""
        output_path = self.export_dir / filename
        
        input_names = ["coords_x_y_t"]
        output_names = ["fields_u_v_p"]
        dynamic_axes = {
            "coords_x_y_t": {0: "batch_size"},
            "fields_u_v_p": {0: "batch_size"}
        }

        torch.onnx.export(
            self.model,
            sample_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes
        )

        parity_result = self.verify_onnx_parity(sample_input, output_path)

        manifest = {
            "export_version": "2.0.0",
            "opset_version": opset_version,
            "checkpoint_sha256": self.metadata.get("checkpoint_sha", "unknown"),
            "architecture": self.metadata.get("architecture", "standard_mlp"),
            "input_spec": {"name": "coords_x_y_t", "shape": ["batch_size", 3], "dtype": "float32"},
            "output_spec": {"name": "fields_u_v_p", "shape": ["batch_size", 3], "dtype": "float32"},
            "parity_verification": parity_result
        }

        manifest_path = self.export_dir / "export_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"ONNX model and manifest saved to {output_path}")
        return output_path, manifest

    def verify_onnx_parity(self, sample_input: torch.Tensor, onnx_path: Path, tolerance: float = 1e-5) -> Dict[str, Any]:
        """Validates numerical equivalence between PyTorch model and ONNX Runtime execution."""
        try:
            import onnxruntime as ort
        except ImportError:
            logger.warning("onnxruntime not installed. Skipping ONNX runtime parity test.")
            return {"status": "skipped", "reason": "onnxruntime_not_installed"}

        with torch.no_grad():
            torch_out = self.model(sample_input).numpy()

        session = ort.InferenceSession(str(onnx_path))
        ort_inputs = {session.get_inputs()[0].name: sample_input.numpy()}
        ort_out = session.run(None, ort_inputs)[0]

        max_diff = float(np.max(np.abs(torch_out - ort_out)))
        passed = bool(max_diff < tolerance)

        logger.info(f"ONNX Parity Verification: Max Discrepancy = {max_diff:.8e} (Tolerance: {tolerance:.1e}) -> {'PASSED' if passed else 'FAILED'}")
        if not passed:
            raise ValueError(f"ONNX parity check failed! Discrepancy {max_diff:.8e} exceeds tolerance {tolerance:.1e}")

        return {
            "status": "passed" if passed else "failed",
            "max_absolute_error": max_diff,
            "tolerance": tolerance,
            "passed": passed
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample_input = torch.randn(64, 3, dtype=torch.float32)
    
    ckpt_candidates = [
        "artifacts/fourier_constrained_seed8008.pth",
        "final_pinn_model.pth",
        "results/smoke/model_checkpoint.pth"
    ]
    ckpt_to_use = None
    for p in ckpt_candidates:
        if os.path.exists(p):
            ckpt_to_use = p
            break

    if ckpt_to_use:
        exporter = PINNExporter(checkpoint_path=ckpt_to_use)
        exporter.export_torchscript(sample_input)
        exporter.export_onnx(sample_input)
    else:
        print("No checkpoint found to export. Run benchmark or smoke test first.")