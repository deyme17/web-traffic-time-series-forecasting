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
            loss = loss.masked_fill(~mask, 0.)
            return loss.sum() / mask.sum().clamp(min=1)
        
        return loss.mean() 