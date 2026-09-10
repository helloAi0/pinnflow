import torch
import torch.nn as nn

class AdaptivePINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        # Adaptive activation parameters
        self.a = nn.Parameter(torch.ones(len(layers)-1))

    def forward(self, x):
        for i in range(len(self.linears)-1):
            # Multiply by learnable parameter 'a' to dynamically scale the activation
            x = torch.tanh(self.a[i] * self.linears[i](x))
        return self.linears[-1](x)