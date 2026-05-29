import torch
from torch.utils.data.dataset import Dataset
from pathlib import Path


class WTTSF_Dataset(Dataset):
    def __init__(self, data_root: Path, transforms=None):
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, idx: int):
        pass