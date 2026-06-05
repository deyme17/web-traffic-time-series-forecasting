from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RegistryConfig:
    name: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Config:
    # paths
    data_dir: Path = Path("data").resolve()
    checkpoints_dir: Path = Path("checkpoints").resolve()

    # data
    nan_threshold: float = 0.3      # drop page if fraction of NaNs > nan_threshold
    max_gap_interpolate: int = 7    # if gaps <= max_gap_interpolate -> linear interpolation
    winsor_k: float = 4.            # spike threshold: median +- winsor_k * MAD

    lookback: int = 90
    horizon: int = 62
    add_days: int = 63
    first_date: str = "2015-07-01"

    # dataloader
    batch_size: int = 512

    n_workers: int = 4
    persistent_workers: bool = False
    prefetch_factor: int|None = None

    shuffle: bool = True
    drop_last: bool = False
    pin_memory: bool = True

    # training
    seed: int = 17
    epochs: int = 50
    max_norm: float|None = None

    # registries
    model: RegistryConfig = field(default_factory=RegistryConfig)
    optimizer: RegistryConfig = field(default_factory=RegistryConfig)
    loss: RegistryConfig = field(default_factory=RegistryConfig)

    def __post_init__(self) -> None:
        """
        Derived/default logic after initialization.
        """
        self.data_dir = Path(self.data_dir).resolve()
        self.checkpoints_dir = Path(self.checkpoints_dir).resolve()
        self.persistent_workers = self.n_workers > 0
        self.prefetch_factor = 2 if self.n_workers > 0 else None

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "Config":
        cfg = cfg.copy()
        cfg["model"] = RegistryConfig(**cfg.get("model", {}))
        cfg["optimizer"] = RegistryConfig(**cfg.get("optimizer", {}))
        cfg["loss"] = RegistryConfig(**cfg.get("loss", {}))
        return cls(**cfg)