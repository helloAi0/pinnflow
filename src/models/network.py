import torch
import torch.nn as nn

class NavierStokesPINN(nn.Module):
    def __init__(self, layers: list[int], activation_type: str = "tanh"):
        super().__init__()
        self.layers = layers
        
        if activation_type.lower() == "tanh":
            self.activation = nn.Tanh()
        elif activation_type.lower() == "sin":
            self.activation = lambda x: torch.sin(x)
        else:
            raise ValueError(f"Activation type '{activation_type}' not supported.")
            
        net_layers = []
        for i in range(len(layers) - 2):
            net_layers.append(nn.Linear(layers[i], layers[i+1]))
            net_layers.append(self.activation)
        
        net_layers.append(nn.Linear(layers[-2], layers[-1]))
        self.dnn = nn.Sequential(*net_layers)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        print(f"[+] PINN architecture initialized on device: {self.device}")

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.device != self.device:
            coords = coords.to(self.device)
        return self.dnn(coords)
