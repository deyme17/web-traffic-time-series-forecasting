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
    max_gap_interpolate: int|None = 7   # if gaps <= max_gap_interpolate -> linear interpolation
    winsor_k: float|None = None         # spike threshold: median +- winsor_k * MAD
    dead_check_window: int = 365        # remove pages where's no traffic in last `check_window` days
    max_zero_ratio: float = 0.3         # exclude pages where the fraction of zeros in the series (in dataset sampling) > threshold.

    lookback: int = 365
    horizon: int = 62
    add_days: int = 63
    first_date: str = "2015-07-01"

    # dataloader
    train_batch: int = 256
    valid_batch: int = 512
    test_batch: int = 512

    n_workers: int = 4
    persistent_workers: bool = False
    prefetch_factor: int|None = None

    shuffle: bool = True
    drop_last: bool = False
    pin_memory: bool = True

    # training
    seed: int = 17
    epochs: int = 50
    warmup_epochs: int = 0
    max_norm: float|None = None
    patience: int|None = None
    ema_decay: float|None = None

    tsp_h: float = 0.   # temporal_smoothness_penalty for hidden state
    tsp_c: float = 0.   # temporal_smoothness_penalty for cell (LSTM) state
    sep_h: float = 0.   # state_energy_penalty for hidden state
    sep_c: float = 0.   # state_energy_penalty for cell (LSTM) state

    # registries
    model: RegistryConfig = field(default_factory=RegistryConfig)
    loss: RegistryConfig = field(default_factory=RegistryConfig)
    optimizer: RegistryConfig = field(default_factory=RegistryConfig)
    scheduler: RegistryConfig = field(default_factory=RegistryConfig)

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
        cfg["loss"] = RegistryConfig(**cfg.get("loss", {}))
        cfg["optimizer"] = RegistryConfig(**cfg.get("optimizer", {}))
        cfg["scheduler"] = RegistryConfig(**cfg.get("scheduler", {}))
        return cls(**cfg)