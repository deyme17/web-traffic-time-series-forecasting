from .register_optimizers import OPTIMIZERS
from torch.optim.optimizer import ParamsT
import torch.optim as opt
from utils import Config


def get_optimizer(config: Config, params: ParamsT) -> opt.Optimizer: 
    opt_factory = OPTIMIZERS.get(config.optimizer.name)
    if opt_factory is None:
        raise ValueError(f"Unknown optimizer: {config.optimizer.name}")
    print(f"Optimizer: {config.optimizer.name}")
    return opt_factory(params, **config.optimizer.parameters)