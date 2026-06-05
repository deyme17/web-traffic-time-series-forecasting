from .register_schedulers import SCHEDULERS
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
from utils import Config


def get_scheduler(config: Config, optim: Optimizer) -> LRScheduler: 
    lrs_factory = SCHEDULERS.get(config.scheduler.name)
    if lrs_factory is None:
        raise ValueError(f"Unknown scheduler: {config.scheduler.name}")
    print(f"LR Scheduler: {config.scheduler.name}")
    return lrs_factory(optim, **config.scheduler.parameters)