import logging
import torch
from scipy.stats import qmc
from src.geometry.domain import CylinderDomain

logger = logging.getLogger("CollocationSampler")

class CollocationSampler:
    def __init__(self, domain: CylinderDomain, seed: int = 42):
        self.domain = domain
        self.seed = seed
        
    def sample_lhs(self, n_points: int, requires_grad: bool = True) -> torch.Tensor:
        """
        Generates continuous spatio-temporal collocation points using LHS,
        rejecting points that fall inside the cylinder geometry.
        """
        logger.info(f"Generating {n_points} collocation points using LHS...")
        engine = qmc.LatinHypercube(d=3, seed=self.seed)
        valid_points = []
        points_needed = n_points
        
        while points_needed > 0:
            # Oversample to compensate for points discarded inside the cylinder
            raw_samples = engine.random(n=int(points_needed * 1.2) + 100)
            
            x = self.domain.x_min + raw_samples[:, 0] * (self.domain.x_max - self.domain.x_min)
            y = self.domain.y_min + raw_samples[:, 1] * (self.domain.y_max - self.domain.y_min)
            t = self.domain.t_min + raw_samples[:, 2] * (self.domain.t_max - self.domain.t_min)
            
            x_t = torch.tensor(x, dtype=torch.float32)
            y_t = torch.tensor(y, dtype=torch.float32)
            t_t = torch.tensor(t, dtype=torch.float32)
            
            mask = self.domain.is_inside_fluid(x_t, y_t)
            
            valid_x = x_t[mask].unsqueeze(1)
            valid_y = y_t[mask].unsqueeze(1)
            valid_t = t_t[mask].unsqueeze(1)
            
            batch_valid = torch.cat([valid_x, valid_y, valid_t], dim=1)
            valid_points.append(batch_valid)
            points_needed -= batch_valid.shape[0]
        
        final_points = torch.cat(valid_points, dim=0)[:n_points, :]
        final_points.requires_grad_(requires_grad)
        
        logger.info(f"Successfully sampled {final_points.shape[0]} valid collocation points.")
        return final_points
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    domain = CylinderDomain()
    sampler = CollocationSampler(domain)
    # Test generation of 50,000 PDE collocation points for physics loss
    pts = sampler.sample_lhs(50000)