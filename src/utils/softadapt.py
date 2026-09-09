import torch

class SoftAdapt:
    def __init__(self, num_losses=2, beta=0.1):
        """
        beta: hyperparameter controlling the sharpness of the softmax.
              Higher beta = more aggressive weight updates.
        """
        self.beta = beta
        self.num_losses = num_losses
        self.weights = torch.ones(num_losses)
        self.loss_history = []

    def get_weights(self):
        return self.weights.clone().detach()

    def update(self, current_losses):
        self.loss_history.append([l.item() for l in current_losses])
        
        # Need at least two epochs to calculate rate of change
        if len(self.loss_history) < 2:
            return self.get_weights()
            
        l_t = torch.tensor(self.loss_history[-1])
        l_t_minus_1 = torch.tensor(self.loss_history[-2])
        
        # Calculate rates of change (handling potential division by zero)
        rates_of_change = l_t / (l_t_minus_1 + 1e-8)
        
        # Compute Softmax weights
        # We want to assign HIGHER weights to losses that are decreasing SLOWEST (or increasing)
        numerator = torch.exp(self.beta * rates_of_change)
        denominator = torch.sum(numerator)
        
        self.weights = (numerator / denominator) * self.num_losses
        return self.get_weights()