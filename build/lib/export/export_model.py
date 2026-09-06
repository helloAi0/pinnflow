import os
import torch
import numpy as np
from pathlib import Path

from src.models.mlp import NavierStokesMLP

class PINNExporter:
    """Exports trained PINN models to TorchScript JIT and ONNX runtime formats."""
    
    def __init__(self, model_path="final_pinn_model.pth", export_dir="exports"):
        self.device = torch.device("cpu")  # Export on CPU for standard portable deployment
        self.model = NavierStokesMLP().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def export_torchscript(self, sample_input: torch.Tensor):
        """Exports model to standalone TorchScript format via JIT tracing."""
        traced_model = torch.jit.trace(self.model, sample_input)
        output_path = self.export_dir / "pinn_model.pt"
        traced_model.save(str(output_path))
        print(f"Saved TorchScript model to: {output_path}")
        return output_path

    def export_onnx(self, sample_input: torch.Tensor):
        """Exports model to ONNX format with dynamic batch sizes."""
        output_path = self.export_dir / "pinn_model.onnx"
        torch.onnx.export(
            self.model,
            sample_input,
            str(output_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['coords_x_y_t'],
            output_names=['fields_u_v_p'],
            dynamic_axes={
                'coords_x_y_t': {0: 'batch_size'},
                'fields_u_v_p': {0: 'batch_size'}
            }
        )
        print(f"Saved ONNX model to: {output_path}")
        return output_path

    def verify_onnx_parity(self, sample_input: torch.Tensor, onnx_path: Path):
        """Validates numerical parity between PyTorch and ONNX Runtime outputs."""
        try:
            import onnxruntime as ort
        except ImportError:
            print("onnxruntime package not found. Skipping ONNX parity check.")
            print("Install onnxruntime via 'pip install onnxruntime' if required.")
            return

        torch_out = self.model(sample_input).detach().numpy()

        ort_session = ort.InferenceSession(str(onnx_path))
        ort_inputs = {ort_session.get_inputs()[0].name: sample_input.numpy()}
        ort_out = ort_session.run(None, ort_inputs)[0]

        max_diff = float(np.max(np.abs(torch_out - ort_out)))
        print(f"Max absolute numerical discrepancy (PyTorch vs ONNX): {max_diff:.8e}")
        
        is_valid = max_diff < 1e-5
        print(f"Parity check: {'PASSED' if is_valid else 'FAILED'}")

if __name__ == "__main__":
    exporter = PINNExporter()
    
    # Dummy input batch of 100 sample points: (x, y, t)
    sample_input = torch.randn(100, 3, dtype=torch.float32)
    
    print("1. Exporting TorchScript model...")
    exporter.export_torchscript(sample_input)
    
    print("2. Exporting ONNX model...")
    onnx_file = exporter.export_onnx(sample_input)
    
    print("3. Verifying output parity...")
    exporter.verify_onnx_parity(sample_input, onnx_file)