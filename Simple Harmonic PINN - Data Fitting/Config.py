import numpy as np

class Config:
    # 1_data.py parameters
    # Physics Parameters
    A = 1                       # amplitude
    k = 1                       # spring constant (N/m)
    m = 1                       # mass (kg)
    phi = 0                     # phase
    omega = np.sqrt(k/m)        # angular frequency (rad/s)

    x0 = 1.0                    # initial displacement (m)
    v0 = 0.0                    # initial velocity (m/s)

    # Data Parameters
    t_start, t_end = 0.0, 10.0  # time window
    N_data = 50                 # number of data points
    noise_level = 0.1           # noise level for synthetic data

    # Model / training
    n_hidden = 20
    hidden_layers = 2
    lr = 1e-2
    num_epochs = 1000
    lambda_data = 0.5
    lambda_ode = 1.0
    lambda_ic = 1.0
    print_frac = 0.1
