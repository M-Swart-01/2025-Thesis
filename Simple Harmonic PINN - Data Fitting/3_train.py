import torch
import torch.nn as nn  
import Config as cfg

# 1) Data Loss
# 2) Boundary Condition Loss
# 3) Initial Condition Loss

# 1) Data Loss - Standard for NNs
def data_loss(model, t_data, h_data): 
    """
    MSE between predicted h(t_i) and noisy measurements h_data.
    """
    h_pred = model(t_data)
    return torch.mean((h_pred - h_data)**2)

# 2) Physics Loss - Specific to PINNs
def physics_loss(model, t): 
    """
    Compare d(h_pred)/dt with the known expression.
    """
    # t must have requires_grad = True for autograd to work
    t.requires_grad_(True)

    x_pred = model(t)
    dx_dt_pred = derivative(x_pred, t)

    # For each t, physics says dh/dt = A*np.cos(omega*t + phi)
    dh_dt_true = A * torch.cos(omega * t + phi)

    loss_ode = torch.mean((dx_dt_pred - dh_dt_true)**2)
    return loss_ode

# 3) Initial Condition Loss - Specific to PINNs
def initial_condition_loss(model): 
    """
    Enforce h(0) = h0.
    """
    # Evaluate at t=0
    t0 = torch.zeros(1, 1, dtype=torch.float32, requires_grad=False)
    x0_pred = model(t0)
    return (x0_pred - x0).pow(2).mean()


# --- END OF LOSS FUNCTIONS ---
# --- TRAINING SETUP ---

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)   # define type of gradient descent ****** 
#   n this case, Adam optimisation 
#   also defines learning rate ("lr=0.01")

# Hyperparameters for weighting the loss terms
lambda_data = 0.5     # training data (Standard for NNs)
lambda_ode  = 1     # physics equations (Specific to PINNs)
lambda_ic   = 1     # initial conditions (Specific to PINNs)

# For logging
num_epochs = 15000               # number of forward propagation and gradient descent runs
print_every = num_epochs*0.1    # number of iterations where results are printed

# --- END OF TRAINING SETUP ---
model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Compute losses
    l_data = data_loss(model, t_data_tensor, x_data_tensor)
    l_ode  = physics_loss(model, t_data_tensor)
    l_ic   = initial_condition_loss(model)

    # Combined loss
    # loss = lambda_data * l_data
    loss = lambda_data * l_data + lambda_ode * l_ode + lambda_ic * l_ic

    # Backprop
    loss.backward() # backpropagation ******
    optimizer.step() # using previously defined optimiser (Adam)

    # Print progress
    if (epoch+1) % print_every == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Total Loss = {loss.item():.6f}, "
              f"Data Loss = {l_data.item():.6f}, "
              f"ODE Loss = {l_ode.item():.6f}, "
              f"IC Loss = {l_ic.item():.6f}"
            )

# --- END OF TRAINING LOOP ---