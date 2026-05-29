from typing import Optional, Tuple, Dict, List, Any
from pathlib import Path
import random as rnd
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.optim import Optimizer
import torch



def set_seed(seed: int) -> None:
    rnd.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def save_checkpoint(model: nn.Module,
                    optim: Optimizer, 
                    train_loss: List[float], 
                    val_loss: Optional[List[float]], 
                    epoch: int, save_path: Path|str) -> None:
    """Save checkpoint dict with epoch, model, optimizer, losses."""
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
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