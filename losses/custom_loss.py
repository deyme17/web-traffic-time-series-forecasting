from abc import ABC, abstractmethod
import torch.nn as nn
import torch


class CustomLoss(nn.Module, ABC):
    """Base interface for custom loss functions."""
    @abstractmethod
    def forward(self,pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor|None = None) -> torch.Tensor:
        pass