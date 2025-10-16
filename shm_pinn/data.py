import math
import numpy as np
import torch

# function - calculation of omega for exact solution - use: data.py
def omega(cfg):
    return math.sqrt(cfg.k / cfg.m)

# function - exact solution - use: data.py
def exact_solution(cfg, t_np: np.ndarray) -> np.ndarray:
    return cfg.A * np.cos(omega(cfg) * t_np + cfg.phi)

# function - generate dataset - use: main.py
def data_gen(cfg, device, dtype):
    N = getattr(cfg, "N_data", 200)
    seed = getattr(cfg, "seed", None)
    if seed is not None:
        np.random.seed(seed)     # !!!!!!!!!!! controls NumPy's random numbers (noise)
        torch.manual_seed(seed)  # !!!!!!!!!!! controls PyTorch RNG (not strictly needed here, but good practice)
    else:
        print("Warning: No seed set in config - results may not be reproducible")

    t_vec = np.linspace(cfg.t_start, cfg.t_end, N)  # !!!!!!!!!!! NumPy time vector
    noise_vec = cfg.noise_std * np.random.randn(len(t_vec)) # ... if cfg.use_data_loss else 0.0 !!!!!! is this a vector?

    # exact solution
    x_exact = exact_solution(cfg, t_vec)    # numpy array?
    # add noise
    x_noise = x_exact + noise_vec   # numpy array?
    # convert to tensors
    t_data = torch.tensor(t_vec, device=device, dtype=dtype).view(-1,1)
    x_noisy = torch.tensor(x_noise, device=device, dtype=dtype).view(-1,1)

    return t_data, x_noisy, x_exact