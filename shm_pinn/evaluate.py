import numpy as np, torch
from data import exact_solution

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
    try:
        import matplotlib.pyplot as plt
        plt.figure(); plt.scatter(t_data.cpu(), x_data.cpu(), s=10, c='tab:red', label='data')
        plt.plot(t_eval.cpu(), torch.tensor(x_true), 'k--', label='exact')
        plt.plot(t_eval.cpu(), torch.tensor(x_pred), label='PINN')
        plt.legend(); plt.tight_layout(); plt.show()
    except Exception:
        pass
    return l2
