import torch.nn as nn
import torch


class MAELoss(nn.Module):
    """Mean Absolute Error (L1) loss."""
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                mask: torch.Tensor|None = None) -> torch.Tensor:
        loss = (target - pred).abs()
        if mask is not None:
            loss = loss[mask]
        return loss.mean() 