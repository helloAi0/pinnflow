import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.fourier_network import FourierConstrainedPINN
# Import BaselinePINN here as well for comparison

def compute_ntk_trace(model, coords):
    """Computes the empirical trace of the NTK matrix."""
    model.eval()
    traces = []
    
    # Randomly sample a small batch to avoid OOM errors
    idx = torch.randperm(coords.size(0))[:500]
    sample_coords = coords[idx].requires_grad_(True)
    
    for i in range(sample_coords.size(0)):
        model.zero_grad()
        out = model(sample_coords[i:i+1])
        # Compute gradient of output u w.r.t parameters
        u_val = out[0, 0]
        u_val.backward(retain_graph=True)
        
        # Sum squared gradients for the trace
        grad_sq_sum = sum(p.grad.pow(2).sum() for p in model.parameters() if p.grad is not None)
        traces.append(grad_sq_sum.item())
        
    return np.mean(traces)

def analyze_spectral_bias():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize networks
    fourier_model = FourierConstrainedPINN(layers=[3, 200, 200, 3], fourier_features=64).to(device)
    # baseline_model = BaselinePINN(layers=[3, 200, 200, 3]).to(device)
    
    # 2. Generate test coordinates
    coords = torch.rand((2000, 3), device=device) * 2.0 - 1.0
    
    # 3. Compute Traces
    fourier_trace = compute_ntk_trace(fourier_model, coords)
    # baseline_trace = compute_ntk_trace(baseline_model, coords)
    
    print(f"[*] NTK Trace (Fourier Model): {fourier_trace:.4e}")
    # print(f"[*] NTK Trace (Baseline Model): {baseline_trace:.4e}")
    print("[+] Higher NTK trace indicates faster convergence for high-frequency functions (alleviating spectral bias).")

if __name__ == "__main__":
    analyze_spectral_bias()