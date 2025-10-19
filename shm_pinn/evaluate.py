import numpy as np
import torch
from data import exact_solution
import matplotlib.pyplot as plt

@torch.no_grad()
def evaluate_and_plot(net, cfg, t_data, x_data):

    t_eval = torch.linspace(
    cfg.t_start, cfg.t_end, 801,
    device=cfg.device,
    dtype={"float32": torch.float32, "float64": torch.float64}[cfg.dtype_str]
    ).unsqueeze(1)
        
    x_pred = net(t_eval).cpu().numpy().squeeze()
    x_true = exact_solution(cfg, t_eval.cpu().numpy().squeeze())
    l2 = float(np.sqrt(np.mean((x_pred - x_true)**2)))
    print(f"[Eval] L2 error = {l2:.3e}")
    
    fig = None
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(t_data.cpu(), x_data.cpu(), s=10, c='tab:red', label='data')
    plt.plot(t_eval.cpu(), torch.tensor(x_true), 'k--', label='exact')
    plt.plot(t_eval.cpu(), torch.tensor(x_pred), label='PINN')
    plt.xlabel('Time (s)')
    plt.ylabel('X Position (m)')
    plt.title('Prediction of Simple Harmonic Motion')
    plt.legend(); plt.tight_layout()
    plt.show()

    return l2, fig
