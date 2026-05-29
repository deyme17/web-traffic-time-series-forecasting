from pathlib import Path
from torch.utils.data import DataLoader
from .dataset import WTTSF_Dataset
from utils import Config


def get_dataset(data_root: Path, 
                transforms=None) -> WTTSF_Dataset:
    return WTTSF_Dataset(
        data_root=data_root,
        transforms=transforms
    )


def get_dataloader(config: Config,
                   transforms=None) -> DataLoader:
    dl = DataLoader(
        dataset=get_dataset(
            data_root=config.data_dir,
            transforms=transforms
        ),
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.n_workers,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers,
        drop_last=config.drop_last,
        pin_memory=config.pin_memory,
    )
    print(f"Dataloader initialized with: num_workers={config.n_workers}, batch_size={config.batch_size}.")
    return dl