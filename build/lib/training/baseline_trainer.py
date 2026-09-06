import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.mlp import NavierStokesMLP
from src.data.observation_dataset import DataPipelineManager

logger = logging.getLogger("BaselineTrainer")

class BaselineTrainer:
    def __init__(self, model: nn.Module, lr: float = 1e-3, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        
        for coords, targets in dataloader:
            coords = coords.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            preds = self.model(coords)
            loss = self.criterion(preds, targets)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * coords.size(0)
            
        return total_loss / len(dataloader.dataset)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        
        for coords, targets in dataloader:
            coords = coords.to(self.device)
            targets = targets.to(self.device)
            
            preds = self.model(coords)
            loss = self.criterion(preds, targets)
            total_loss += loss.item() * coords.size(0)
            
        return total_loss / len(dataloader.dataset)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Verify baseline pipeline on a tiny scarcity budget (1,000 points)
    manager = DataPipelineManager(seed=42)
    datasets = manager.prepare_experiment_data(train_budget=1000, noise_level=0.0)
    
    train_loader = DataLoader(datasets["train"], batch_size=256, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=1024, shuffle=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NavierStokesMLP()
    trainer = BaselineTrainer(model, device=device)
    
    logger.info(f"Starting Baseline Model A test run on {device}...")
    for epoch in range(1, 6):
        t_loss = trainer.train_epoch(train_loader)
        v_loss = trainer.evaluate(val_loader)
        logger.info(f"Epoch {epoch:02d} | Train Data MSE: {t_loss:.6e} | Val Data MSE: {v_loss:.6e}")