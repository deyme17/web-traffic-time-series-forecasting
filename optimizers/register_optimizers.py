from typing import Tuple
import torch.optim as opt
from torch.optim.optimizer import ParamsT
from utils import Registry

OPTIMIZERS = Registry()


@OPTIMIZERS.register("Adam")
def adam(params: ParamsT,
         lr: float = 0.001,
         betas: Tuple[float, float] = (0.9, 0.999),
         weight_decay: float = 0,
         amsgrad: bool = False, **_) -> opt.Adam:
    return opt.Adam(params,
                    lr=lr, 
                    betas=betas, 
                    weight_decay=weight_decay,
                    amsgrad=amsgrad
                    )

@OPTIMIZERS.register("AdamW")
def adamw(params: ParamsT,
          lr: float = 0.001,
          betas: Tuple[float, float] = (0.9, 0.999),
          weight_decay: float = 0,
          amsgrad: bool = False, **_) -> opt.AdamW:
    return opt.AdamW(params,
                    lr=lr, 
                    betas=betas, 
                    weight_decay=weight_decay,
                    amsgrad=amsgrad
                    )

@OPTIMIZERS.register("RMSprop")
def rmsprop(params: ParamsT,
            lr: float = 0.001,
            alpha: float = 0.99,
            weight_decay: float = 0,
            momentum: float = 0) -> opt.RMSprop:
    return opt.RMSprop(params,
                       lr=lr, 
                       alpha=alpha, 
                       weight_decay=weight_decay,
                       momentum=momentum
                       )     

@OPTIMIZERS.register("SGD")
def sgd(params: ParamsT,
        lr: float = 0.001,
        momentum: float = 0,
        weight_decay: float = 0,
        nesterov: bool = True) -> opt.SGD:
    return opt.SGD(params,
                   lr=lr, 
                   momentum=momentum,
                   weight_decay=weight_decay,
                   nesterov=nesterov
                   )