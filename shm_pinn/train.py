import time
import math
import torch
import torch.nn as nn
from data import omega

def ddt(x, t):      # first derivative of x wrt t, returns gradient, creates graph for higher derivatives
    return torch.autograd.grad(x, t, torch.ones_like(x), create_graph=True)[0]

def d2dt2(x, t):    # second derivative of x wrt t (repeat ddt())
    return ddt(ddt(x, t), t)

_MSE = nn.MSELoss() 

# function - sample times for physics loss
def sample_times(cfg, device, dtype):
    t = torch.rand(cfg.n_phys_per_step,1, device=device, dtype=dtype)*(cfg.t_end-cfg.t_start)+cfg.t_start
    t.requires_grad_(True)
    return t

# function - physics loss
def physics_loss(net, cfg, t_phys):
    x = net(t_phys)
    x_tt = d2dt2(x, t_phys)
    return _MSE(x_tt + (cfg.k/cfg.m)*x, torch.zeros_like(x))

# function - initial condition loss
def ic_loss(net, cfg, device, dtype):
    t0 = torch.zeros(1, 1, device=device, dtype=dtype).requires_grad_(True)
    x0 = torch.tensor([[cfg.A * math.cos(cfg.phi)]], device=device, dtype=dtype)
    v0 = torch.tensor([[-cfg.A * omega(cfg) * math.sin(cfg.phi)]], device=device, dtype=dtype)

    x_pred = net(t0)
    v_pred = ddt(x_pred, t0)
    return _MSE(x_pred, x0) + _MSE(v_pred, v0)

# function - data loss
def data_loss(net, t_data, x_data):
    return _MSE(net(t_data), x_data)

# function - training loop
def train(net, cfg, t_data, x_data, device, dtype):
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    curve_total, curve_phys, curve_ic, curve_data = [], [], [], []
    t0 = time.time()  # start timer

    for step in range(1, cfg.epochs + 1):
        opt.zero_grad(set_to_none=True)

        t_phys = sample_times(cfg, device, dtype)
        l_phys = physics_loss(net, cfg, t_phys)
        l_ic   = ic_loss(net, cfg, device, dtype)
        l_data = data_loss(net, t_data, x_data) if cfg.use_data_loss else torch.tensor(0.0, device=device, dtype=dtype)
        
        w_phys = getattr(cfg, "w_phys", 1.0)
        w_ic   = getattr(cfg, "w_ic", 0.0)
        w_data = getattr(cfg, "w_data", 1.0)
        
        loss = w_phys*l_phys + w_ic*l_ic + w_data*l_data 

        loss.backward(); opt.step()
        
        # logging
        curve_total.append(loss.item())
        curve_phys.append(l_phys.item())
        curve_ic.append(l_ic.item())
        curve_data.append(l_data.item())


        if step % cfg.print_every == 0:
            print(f"[{step:4d}] total={loss.item():.3e} phys={l_phys.item():.3e} ic={l_ic.item():.3e} data={l_data.item():.3e} current time={time.time()-t0:.1f}s")
        # End stats
        train_seconds = (time.time() - t0)
        stats = {
            "final_total": curve_total[-1],
            "final_L2"
            "final_phys":  curve_phys[-1],
            "final_ic":    curve_ic[-1],
            "final_data":  curve_data[-1],
            "curve_total": curve_total,
            "curve_phys":  curve_phys,
            "curve_ic":    curve_ic,
            "curve_data":  curve_data,
            "train_time_sec": train_seconds,
        }
    return net, stats