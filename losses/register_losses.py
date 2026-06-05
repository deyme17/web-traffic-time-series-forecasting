from utils import Registry
from .mae_loss import MAELoss
from .smape_loss import SMAPELoss
from .huber_loss import HuberLoss

LOSSES = Registry()

LOSSES.register("MAE")(MAELoss)
LOSSES.register("SMAPE")(SMAPELoss)
LOSSES.register("Huber")(HuberLoss)