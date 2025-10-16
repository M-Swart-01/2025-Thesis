import torch
import data
import model
import train
import evaluate
from config import Config

def to_dtype(s): return {"float32": torch.float32, "float64": torch.float64}[s] # !!!!!! def

if __name__ == "__main__":
    cfg = Config()
    device = cfg.device
    dtype  = to_dtype(cfg.dtype_str)

    # data generation
    t_data, x_data, _ = data.data_gen(cfg, device, dtype) # !!!!!!! what is _ and what about X_clean

    # compute hard ics here

    # build and initialise model
    net = model.build_model(cfg)

    # train
    net, stats = train.train(net, cfg, t_data, x_data, device, dtype)

    # evaluation / plotting
    evaluate.evaluate_and_plot(net, cfg, t_data, x_data)
    