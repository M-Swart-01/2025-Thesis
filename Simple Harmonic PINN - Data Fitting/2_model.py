import torch
import torch.nn as nn
import Config as cfg

# --- MODEL INITIALISATION ---

class MLP(nn.Module):                                           
    def __init__(self, cfg, x0, v0, device, dtype):             # define number of NEURONS per hidden layer
        super().__init__()
        layers = [nn.Linear(1, cfg.h_width), nn.Tanh()]
        for _ in range(cfg.h_depth-1):
            layers += [nn.Linear(cfg.h_width, cfg.h_width), nn.Tanh()]  # hidden layers
        layers += [nn.Linear(cfg.h_width, 1)]                           # output layer
        self.net = nn.Sequential(*layers)                               # put layers together in sequence


    def forward(self, t):
        """
        Forward pass: input shape (batch_size, 1) -> output shape (batch_size, 1)
        """
        return self.net(t)

# Initialise 
model = MLP(n_hidden=20) #****************************************************** HELP

# --- END OF MODEL INITIALISATION ---
# --- AUTOMATIC DIFFERENTIATION FUNCTION ---

def derivative(x, t):
    return torch.autograd.grad(
        x, t,
        grad_outputs=torch.ones_like(x),            #This is a trick for handling when y is a vector (batch of outputs). By default, PyTorch needs to know how to combine multiple outputs into one scalar before differentiating. Multiplying by a tensor of ones is like saying: “just add them all up and differentiate that.” Effectively: you get element-wise derivatives back.
        create_graph=True
    )[0]

# --- END AUTOMATIC DIFFERENTIATION FUNCTION ---
