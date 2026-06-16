from typing import Optional, Tuple, Dict, List, Any
from pathlib import Path
import random as rnd
import numpy as np
import csv
import zipfile
import matplotlib.pyplot as plt

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch_ema import ExponentialMovingAverage
import torch




def set_seed(seed: int) -> None:
    rnd.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def _open_csv(path: Path):
    if path.suffix == ".zip":
        zf = zipfile.ZipFile(path)
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        return zf.open(name)
    return open(path, "rb")

def read_csv(path: Path) -> Tuple[List[str], np.ndarray]:
    """Row-by-row read. Returns (page_names, float32 [n_pages, n_days], NaN=missing)."""
    print(f"Reading {path} ...")
    with _open_csv(path) as fh:
        reader = csv.reader(l.decode() for l in fh)
        header = next(reader)
        n_days = len(header) - 1
        n_pages = sum(1 for _ in reader)

    pages = []
    data = np.full((n_pages, n_days), np.nan, dtype=np.float32)

    with _open_csv(path) as fh:
        reader = csv.reader(l.decode() for l in fh)
        next(reader)
        for i, row in enumerate(reader):
            pages.append(row[0])
            for j, v in enumerate(row[1:]):
                if v:
                    data[i, j] = float(v)
            if i % 10_000 == 0:
                print(f"  {i}/{n_pages}", end="\r", flush=True)

    print(f"\n\tshape: {data.shape}")
    return pages, data



def save_checkpoint(model: nn.Module,
                    optim: Optimizer, 
                    scheduler: Optional[LRScheduler],
                    ema: Optional[ExponentialMovingAverage],
                    train_loss: List[float], 
                    val_loss: Optional[List[float]], 
                    epoch: int, save_path: Path|str) -> None:
    """Save checkpoint dict with epoch, model, optimizer, losses."""
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "ema": ema.state_dict(),
        "scheduler": scheduler.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }, Path(save_path))



def load_checkpoint(path: Path|str,
                    model: nn.Module,
                    map_location: str = None) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load checkpoint and return (model, whole_checkpoint)."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=map_location or device)
    if "model" not in checkpoint:
        raise KeyError(f"`model` key is not found in checkpoint: {path}")
    
    model.to(device)
    try:
        model.load_state_dict(checkpoint["model"], strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load state_dict for {type(model).__name__}: {e}"
        )
    model.eval()

    return model, checkpoint



def visualize_training(train_loss: list[float], 
                       val_loss: Optional[list[float]], 
                       title: str = "Training plot",
                       save: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_loss, label='Train Loss')
    if val_loss is not None:
        ax.plot(val_loss, label='Validation Loss')
    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.tight_layout()
    if save:
        fig.savefig(f"{title}.png")
        plt.close(fig)
    else:
        plt.show()



def normalize(arr: np.ndarray) -> np.ndarray:
    mu = arr.mean(axis=0)
    std = arr.std(axis=0)

    if np.isscalar(std):
        std = 1. if std == 0 else std
    else:
        std[std == 0] = 1.

    return ((arr - mu) / std).astype(np.float32)



def denormalize_tensor(input: torch.Tensor,
                       mean: torch.Tensor,
                       std: torch.Tensor) -> torch.Tensor:
    return input * std + mean