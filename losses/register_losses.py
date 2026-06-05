from utils import Registry
from torch.nn import HuberLoss, L1Loss
from .smape_loss import SMAPELoss

LOSSES = Registry()

LOSSES.register("Huber")(HuberLoss)
LOSSES.register("MAE")(L1Loss)
LOSSES.register("SMAPE")(SMAPELoss)