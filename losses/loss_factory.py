from .register_losses import LOSSES
import torch.nn as nn
from utils import Config


def get_loss(config: Config) -> nn.Module:
    loss_cfg = config.loss
    loss_factory = LOSSES.get(loss_cfg.name)
    if loss_factory is None:
        raise ValueError(f"Unknown loss function: {loss_cfg.name}")
    print(f"Loss function: {loss_cfg.name}")
    return loss_factory(**loss_cfg.parameters)