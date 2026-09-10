import torch
import numpy as np
from src.models.baseline_network import BaselinePINN
from src.models.fourier_network import FourierConstrainedPINN

def compute_empirical_ntk(model: torch.nn.Module, x: torch.Tensor):
    """Computes empirical NTK matrix for velocity output u w.r.t model parameters."""
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    N = x.shape[0]
    
    jacobians = []
    for i in range(N):
        model.zero_grad()
        out = model(x[i:i+1])
        u_out = out[0, 0] # Output u
        u_out.backward(retain_graph=True)
        
        grad_vec = torch.cat([p.grad.view(-1) for p in params if p.grad is not None])
        jacobians.append(grad_vec)
        
    J = torch.stack(jacobians) # [N, Num_Params]
    K = J @ J.T              # [N, N] Empirical NTK
    return K

def run_spectral_analysis():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_coords = torch.rand((128, 3), device=device) * 2.0 - 1.0
    
    models = {
        "Baseline PINN": BaselinePINN(layers=[3] + [200] * 8 + [3]).to(device),
        "Fourier PINN": FourierConstrainedPINN(layers=[3] + [200] * 8 + [3], fourier_features=64).to(device)
    }
    
    print("\n" + "="*60)
    print("NEURAL TANGENT KERNEL (NTK) EIGENVALUE ANALYSIS")
    print("="*60)
    
    for name, model in models.items():
        K = compute_empirical_ntk(model, sample_coords)
        eigenvalues = torch.linalg.eigvalsh(K).cpu().numpy()
        
        lambda_max = eigenvalues[-1]
        lambda_min = max(eigenvalues[0], 1e-12)
        condition_number = lambda_max / lambda_min
        
        print(f"\nModel: {name}")
        print(f"  - Max Eigenvalue (λ_max) : {lambda_max:.4e}")
        print(f"  - Min Eigenvalue (λ_min) : {lambda_min:.4e}")
        print(f"  - Condition Number (κ)   : {condition_number:.4e}")
        print(f"  - Spectral Energy Ratio  : {np.sum(eigenvalues[-10:]) / np.sum(eigenvalues):.4f}")

if __name__ == "__main__":
    run_spectral_analysis()