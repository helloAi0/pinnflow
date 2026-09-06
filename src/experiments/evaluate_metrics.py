import torch
import numpy as np
import logging
from torch.utils.data import DataLoader

from src.models.mlp import NavierStokesMLP
from src.data.observation_dataset import DataPipelineManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def calculate_l2_error(exact, pred):
    """Calculates the Relative L2 error metric."""
    return np.linalg.norm(exact - pred, 2) / np.linalg.norm(exact, 2)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 1. Load Data using Pipeline Manager
    logging.info("Loading test data via DataPipelineManager...")
    manager = DataPipelineManager(seed=42)
    datasets = manager.prepare_experiment_data(train_budget=5000, noise_level=0.0)
    
    # Extract the full test dataset
    test_loader = DataLoader(datasets["test"], batch_size=len(datasets["test"]), shuffle=False)
    X_test, Y_test = next(iter(test_loader))
    
    X_test = X_test.to(device)
    
    exact_u_test = Y_test[:, 0:1].numpy()
    exact_v_test = Y_test[:, 1:2].numpy()
    exact_p_test = Y_test[:, 2:3].numpy()

    # 2. Load the Trained Model
    model_path = 'final_pinn_model.pth'
    logging.info(f"Loading trained PINN model from {model_path}...")
    
    model = NavierStokesMLP().to(device) 
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Run Inference
    logging.info(f"Running inference on {len(X_test)} test points...")
    with torch.no_grad():
        predictions = model(X_test)
        
        # Assuming output is [u, v, p]
        u_pred = predictions[:, 0:1].cpu().numpy()
        v_pred = predictions[:, 1:2].cpu().numpy()
        p_pred = predictions[:, 2:3].cpu().numpy()

    # 4. Compute Metrics
    logging.info("Calculating Relative L2 Errors...")
    
    error_u = calculate_l2_error(exact_u_test, u_pred)
    error_v = calculate_l2_error(exact_v_test, v_pred)
    error_p = calculate_l2_error(exact_p_test, p_pred)

    print("\n" + "="*40)
    print("      PHASE 13: EVALUATION METRICS      ")
    print("="*40)
    print(f"Velocity U (Relative L2): {error_u:.4e}")
    print(f"Velocity V (Relative L2): {error_v:.4e}")
    print(f"Pressure P (Relative L2): {error_p:.4e}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()