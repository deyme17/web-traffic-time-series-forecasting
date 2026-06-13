from __future__ import annotations

from pathlib import Path
from torch.utils.data import DataLoader

from .dataset import WTTSF_Dataset
from utils import Config



def get_dataset(data_root: Path,
                config: Config,
                split: str = "train",
                back_offset: int = 0,
                transforms = None) -> WTTSF_Dataset:
    return WTTSF_Dataset(
        data_root=data_root,
        lookback=config.lookback,
        horizon=config.horizon,
        split=split,
        back_offset=back_offset,
        transforms=transforms,
        seed=config.seed if split == "train" else None,
    )


def get_dataloader(config: Config,
                   split: str = "train",
                   back_offset: int = 0,
                   transforms = None) -> DataLoader:
    dataset = get_dataset(
        data_root=config.data_dir,
        config=config,
        split=split,
        back_offset=back_offset,
        transforms=transforms,
    )
    shuffle = config.shuffle and split == "train"
    drop_last = config.drop_last and split == "train"
    b_size = config.train_batch if split == "train" \
             else config.valid_batch if split == "valid" \
             else config.test_batch

    dl = DataLoader(
        dataset=dataset,
        batch_size=b_size,
        shuffle=shuffle,
        num_workers=config.n_workers,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers,
        drop_last=drop_last,
        pin_memory=config.pin_memory,
    )
    print(f"[{split}] Dataloader initialized with: num_workers={config.n_workers}, batch_size={b_size}.")
    return dl