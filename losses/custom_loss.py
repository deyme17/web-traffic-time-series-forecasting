import torch.nn as nn
import torch


class CustomLoss(nn.Module):
    """Class-interface for custom loss function implementations."""
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                mask: torch.Tensor|None = None) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement this method!")