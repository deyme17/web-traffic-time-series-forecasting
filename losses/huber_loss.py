import torch.nn as nn
import torch


class HuberLoss(nn.Module):
    """Huber loss. It combines the best features of MSE and MAE."""
    def __init__(self, delta: float = 0.1):
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                mask: torch.Tensor|None = None) -> torch.Tensor:
        error = (target - pred).abs()

        loss = torch.where(
            error <= self.delta,
            0.5 * error**2,
            self.delta * error - (0.5 * self.delta**2)
        )

        if mask is not None:
            loss = loss.masked_fill(~mask, 0.)
            return loss.sum() / mask.sum().clamp(min=1)

        return loss.mean()