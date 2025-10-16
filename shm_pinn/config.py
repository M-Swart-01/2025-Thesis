# config.py
from dataclasses import dataclass

@dataclass
class Config:
    # runtime
    device: str = "cpu"          # "cpu" or "cuda"
    dtype_str: str = "float32"   # "float32" or "float64"

    # data
    A: float = 1.0               # amplitude
    k: float = 1.0               # spring constant (N/m)
    m: float = 1.0               # mass (kg)
    phi: float = 0.0             # phase (rad)
    N_data: int = 200
    seed: int = 1
    t_start: float = 0.0
    t_end: float = 10.0
    noise_std: float = 0.1       # noise level

    # model
    hidden_depth: int = 3        # number of hidden layers
    hidden_width: int = 32       # neurons per hidden layer
    activation: str = "tanh"     # "tanh" | "silu" | "relu" | "sigmoid" | "linear"

    # training
    n_phys_per_step: int = 1000
    lr: float = 1e-3             # learning rate
    epochs: int = 15000

    # training flags / weights (add to Config)
    print_every: int = 1000
    use_data_loss: bool = True
    w_phys: float = 1.0
    w_ic: float   = 1.0   
    w_data: float = 1.0

    # evaluation
    steps: int = 50000            # steps between evaluations/prints


