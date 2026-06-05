from .model import MODELS
from utils import Config
import torch.nn as nn


def get_model(config: Config) -> nn.Module:
    cfg = config.model
    model_factory = MODELS.get(cfg.name)
    if model_factory is None:
        raise ValueError(f"Unknown Model: {cfg.name}")
    print(f"Model: {cfg.name}")
    return model_factory(**cfg.parameters)