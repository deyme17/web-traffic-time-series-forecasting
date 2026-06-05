import torch.nn as nn
import torch


class SMAPELoss(nn.Module):
    """Symmetric Mean Absolute Percentage Error with smoothing."""
    def __init__(self, epsilon: float = 0.1):
        super().__init__()
        self.eps = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                mask: torch.Tensor|None = None) -> torch.Tensor:
        denom = (pred.abs() + target.abs() + self.eps).clamp(min=0.5 + self.eps)
        loss = (pred - target).abs() / denom * 2.
        if mask is not None:
            loss = loss[mask]
        return loss.mean()