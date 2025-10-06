"""Data generation and preprocessing for mass-spring experiments."""

import torch                            # for tensor operations
import numpy as np                      # for math functions
import Config as cfg                    # configuration parameters

# --- DATA PREPARATION ---

# SHOULD MAYBE GO IN MAIN
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

# Analytical solution
def x_true(t):
    return cfg.A*np.cos(cfg.omega*t + cfg.phi)

# Synthetic data generation
def x_synth(cfg, t):
    t_data = np.linspace(cfg.t_start, cfg.t_end, cfg.N_data)        # time data points
    x_exact = x_true(t_data)
    x_noisy = x_exact + (cfg.noise_std*np.random.randn(len(t_data)))    # MAYBE LATER add- if cfg.use_data_loss else 0)
    tens_t = torch.tensor(t_data, dtype=torch.float32).view(-1, 1)   # time data tensor form
    tens_x = torch.tensor(x_noisy, dtype=torch.float32).view(-1, 1)  # position data tensor form
    return tens_t, tens_x

# Physics setup

# import math, numpy as np, torch

# def omega(cfg): return math.sqrt(cfg.k/cfg.m)
# def true_x(cfg, t_np: np.ndarray): return cfg.A * np.cos(omega(cfg)*t_np + cfg.phi)

# def implied_ics(cfg):
#     w = omega(cfg)
#     x0 = cfg.A*math.cos(cfg.phi)
#     v0 = -cfg.A*w*math.sin(cfg.phi)
#     return x0, v0

# def make_data_tensors(cfg, device, dtype):
    # t = np.linspace(cfg.t_start, cfg.t_end, 200)
    # x_clean = true_x(cfg, t)
    # x_noisy = x_clean + (cfg.noise_std*np.random.randn(len(t)) if cfg.use_data_loss else 0)
    # t_t = torch.tensor(t, device=device, dtype=dtype).view(-1,1)
    # x_t = torch.tensor(x_noisy, device=device, dtype=dtype).view(-1,1)
    # return t_t, x_t, x_clean


# --- END OF DATA PREPARATION ---