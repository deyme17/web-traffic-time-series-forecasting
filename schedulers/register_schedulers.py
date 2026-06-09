import torch.optim.lr_scheduler as lrs
from torch.optim import Optimizer
from utils import Registry

SCHEDULERS = Registry()


@SCHEDULERS.register("Step")
def step(optim: Optimizer,
         step_size: int,
         gamma: float = 0.1,
         last_epoch: int = -1,
         *args):
    return lrs.StepLR(
        optimizer=optim,
        step_size=step_size,
        gamma=gamma,
        last_epoch=last_epoch,
    )

@SCHEDULERS.register("CosineAnnealingWarmRestarts")
def cosine_annealing_warm_restarts(optim: Optimizer,
                                   T_0: int,
                                   T_mult: int = 1,
                                   eta_min: float = 0.,
                                   last_epoch: int = -1,
                                   *args):
    return lrs.CosineAnnealingWarmRestarts(
        optimizer=optim,
        T_0=T_0,
        T_mult=T_mult,
        eta_min=eta_min,
        last_epoch=last_epoch,
    )

@SCHEDULERS.register("CosineAnnealingLR")
def cosine_annealing(optim: Optimizer,
                                   T_max: int,
                                   eta_min: float = 0.,
                                   last_epoch: int = -1,
                                   *args):
    return lrs.CosineAnnealingLR(
        optimizer=optim,
        T_max=T_max,
        eta_min=eta_min,
        last_epoch=last_epoch,
    )

@SCHEDULERS.register("ReduceLROnPlateau")
def reduce_lr_on_plateau(optim: Optimizer,
                         mode: str = 'min',
                         factor: float = 0.1,
                         patience: int = 10,
                         threshold: float = 1e-4,
                         min_lr: float = 0.,
                         *args):
    return lrs.ReduceLROnPlateau(
        optimizer=optim,
        mode=mode,
        factor=factor,
        patience=patience,
        threshold=threshold,
        min_lr=min_lr,
    )

@SCHEDULERS.register("Exponential")
def exponential(optim: Optimizer,
                gamma: float,
                last_epoch: int = -1,
                *args):
    return lrs.ExponentialLR(
        optimizer=optim,
        gamma=gamma,
        last_epoch=last_epoch,
    )

@SCHEDULERS.register("OneCycle")
def one_cycle(optim: Optimizer,
              max_lr: float,
              total_steps: int,
              pct_start: float = 0.3,
              div_factor: float = 25.,
              final_div_factor: float = 10000.,
              *args):
    return lrs.OneCycleLR(
        optimizer=optim,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
    )