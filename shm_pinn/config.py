from dataclasses import dataclass

@dataclass
class Config:
    # general
    device: str = "cpu"          # "cpu" or "cuda"
    dtype_str: str = "float32"   # "float32" or "float64"

    # data
    A: float = 1.0               # amplitude
    k: float = 1.0               # spring constant (N/m)
    m: float = 1.0               # mass (kg)
    phi: float = 0.0             # phase (rad)
    N_data: int = 200            # number of data points
    seed: int = 1                # randomisation seed
    t_start: float = 0.0         # start time (s)
    t_end: float = 60.0          # end time (s)
    noise_std: float = 0.1       # noise level

    # model
    hidden_depth: int = 2        # number of hidden layers
    hidden_width: int = 10       # neurons per hidden layer
    activation: str = "tanh"     # "tanh", "relu", "sigmoid"
    # training
    n_phys_per_step: int = 100   # collocation points per training step
    lr: float = 1e-3             # learning rate
    epochs: int = 5000           # training epochs

    # training flags 
    print_every: int = epochs*0.1   # print frequency
    use_data_loss: bool = True
    w_phys: float = 1.0         # physics loss weights
    w_ic: float   = 1.0         # initial condition loss weights
    w_data: float = 1.0         # data loss weights



