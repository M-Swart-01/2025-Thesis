import torch
import torch.nn as nn

# function - get activation function by name
def get_act(name):
    acts = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "silu": nn.SiLU(),
        "linear": nn.Identity()
    }
    key = name.strip().lower()
    if key not in acts:
        raise ValueError(f"Unknown activation '{name}'. Valid: {', '.join(sorted(acts))}")  # ERROR MESSAGE
    return acts[key]

# function - initialization specific to activation function
def init_weight_by_act(m: nn.Module, act_name: str) -> None:
    if not isinstance(m, nn.Linear):
        return
    key = act_name.strip().lower()
    if key in {"tanh", "sigmoid", "linear"}:
        gain = nn.init.calculate_gain("tanh") if key == "tanh" else 1.0     # gain for tanh, 1.0 otherwise
        nn.init.xavier_uniform_(m.weight, gain=gain)                        # Xavier/Glorot
    elif key in {"relu", "silu"}:
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")             # He/Kaiming
    else:
        raise ValueError(f"Unknown activation '{act_name}'. Expected one of: relu, silu, tanh, sigmoid, linear.")   # ERROR MESSAGE
    # initialise biases to zero - see:"C:\Users\moniq\OneDrive - Queensland University of Technology\2025 Thesis\Thesis Vault\bias.md"
    if m.bias is not None:
        nn.init.zeros_(m.bias)

# class - multi-layer perceptron
class MLP(nn.Module):
    # function - initialization
    def __init__(self, cfg):
        super().__init__()
        
        # self.hard_ics = cfg.hard_ics                                        # boolean, enforching hard ics or not, not coded yet
    #     self.x0 = torch.tensor([[x0]], device=device, dtype=dtype)            # ics not used yet
    #     self.v0 = torch.tensor([[v0]], device=device, dtype=dtype)

        act = get_act(cfg.activation)                                        # get activation function
        # build layers
        layers = [nn.Linear(1, cfg.hidden_width), act]                      # input layer
        for _ in range(cfg.hidden_depth - 1):                               # hidden layers
            layers += [nn.Linear(cfg.hidden_width, cfg.hidden_width), act]
        layers += [nn.Linear(cfg.hidden_width, 1)]                          # output layer
        self.net = nn.Sequential(*layers)         
        # initialise hidden layers for specified activation function
        self.net.apply(lambda m: init_weight_by_act(m, cfg.activation))
        # re-initialize the final/output layer as 'linear' for regression
        last_linear = None
        for mod in self.net.modules():
            if isinstance(mod, nn.Linear):
                last_linear = mod
        if last_linear is not None:
            init_weight_by_act(last_linear, "linear")
    # function - forward pass
    def forward(self, t):
        return self.net(t)
    
# function - build model
def build_model(cfg):
    return MLP(cfg)
