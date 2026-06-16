import torch


def temporal_smoothness_penalty(h: torch.Tensor) -> torch.Tensor:
    diff = h[:, 1:, :] - h[:, :-1, :]
    return (diff**2).mean()


def state_energy_penalty(h: torch.Tensor) -> torch.Tensor:
    return (h**2).mean()