from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torch_ema import ExponentialMovingAverage

from tqdm import tqdm
import yaml
from pathlib import Path
from contextlib import nullcontext

from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from utils import (Config,
    save_checkpoint, load_checkpoint, set_seed, 
    visualize_training, denormalize_tensor,
    temporal_smoothness_penalty, state_energy_penalty
)

from models import get_model
from optimizers import get_optimizer
from schedulers import get_scheduler
from dataset import get_dataloader
from losses import get_loss



def train_rnn(model: nn.Module, 
              optimizer: optim.Optimizer, 
              criterion: nn.Module, 
              config: Config,
              train_loader: DataLoader, 
              valid_loader: Optional[DataLoader], 
              scheduler: Optional[lrs.LRScheduler], 
              ema: Optional[ExponentialMovingAverage],
              curr_epoch: int = 0, 
              experiment_tag: str = "experiment",
              train_losses: Optional[List[float]] = None, 
              val_losses: Optional[List[float]] = None,
              device: torch.device = torch.device("cpu")) -> Tuple[List[float], Optional[List[float]]]:
    """
    RNN training loop. Return: (train_losses, val_losses|None).
    """
    if device.type == "cuda":
        print("[INFO] CUDA is used for training.")
        torch.backends.cudnn.benchmark = True
    else:
        print("[WARNING] CUDA is not available.")

    if train_losses is None: train_losses = []
    if val_losses is None and valid_loader is not None: 
        val_losses = []

    best_val_loss = float('inf')
    patient_level = 0

    for epoch in range(curr_epoch, config.epochs):
        train_loss = val_loss = 0

        # train
        model.train()

        for X in tqdm(train_loader, desc="Train", leave=False):
            enc_in = X["enc_input"].to(device)
            dec_in = X["dec_input"].to(device)
            target = X["target"].to(device)
            target_mask = X["target_mask"].to(device)
            mean = X["series_mean"].to(device).unsqueeze(-1)
            std = X["series_std"].to(device).unsqueeze(-1)

            optimizer.zero_grad(set_to_none=True)
            
            out = model(enc_in, dec_in)

            out = denormalize_tensor(out, mean, std)
            target = denormalize_tensor(target, mean, std)
            loss = criterion(out, target, target_mask)

            # regularization
            if hasattr(model, "h_states") and model.h_states is not None:
                if config.tsp_h > 0:
                    loss += config.tsp_h * temporal_smoothness_penalty(model.h_states)
                if config.sep_h > 0:
                    loss += config.sep_h * state_energy_penalty(model.h_states)
            if hasattr(model, "c_states") and model.c_states is not None:
                if config.tsp_c > 0:
                    loss += config.tsp_c * temporal_smoothness_penalty(model.c_states)
                if config.sep_c > 0:
                    loss += config.sep_c * state_energy_penalty(model.c_states)

            loss.backward()

            # grad clipping
            if config.max_norm is not None:
                clip_grad_norm_(model.parameters(), config.max_norm)

            optimizer.step()
            if ema is not None: 
                ema.update()
            
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # eval
        if valid_loader is not None:
            model.eval()
            with (
                ema.average_parameters() if ema is not None else nullcontext(),
                torch.no_grad()
            ):
                for X in tqdm(valid_loader, desc="Valid", leave=False):
                    enc_in = X["enc_input"].to(device)
                    dec_in = X["dec_input"].to(device)
                    target = X["target"].to(device)
                    target_mask = X["target_mask"].to(device)
                    mean = X["series_mean"].to(device).unsqueeze(-1)
                    std = X["series_std"].to(device).unsqueeze(-1)

                    out = model(enc_in, dec_in)

                    out = denormalize_tensor(out, mean, std)
                    target = denormalize_tensor(target, mean, std)
                    loss = criterion(out, target, target_mask)

                    val_loss += loss.item()

            val_loss /= len(valid_loader)
            val_losses.append(val_loss)
        else:
            val_loss = train_loss

        # schedule lr
        if scheduler is not None:
            if isinstance(scheduler, lrs.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # log
        lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
        val_str = "NaN" if valid_loader is None else f"{val_loss:.3f}"
        print(f"[Epoch: {epoch + 1}] Train Loss: {train_loss:.3f} | Val Loss: {val_str} | lr: {lr}")
        
        # checkpoint
        if val_loss < best_val_loss:
            if epoch >= config.warmup_epochs:
                with ema.average_parameters() if ema is not None else nullcontext():
                    save_checkpoint(
                        model=model, 
                        optim=optimizer,
                        scheduler=scheduler,
                        ema=ema,
                        train_loss=train_losses, 
                        val_loss=val_losses, 
                        epoch=epoch,
                        save_path=config.checkpoints_dir / f"{experiment_tag}_e{epoch + 1}_checkpoint.pt"
                    )
            best_val_loss = val_loss
            patient_level = 0
        else:
            patient_level += 1

        # early stopping
        if config.patience is not None and patient_level >= config.patience:
            break

    return train_losses, val_losses


# main section
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="WTTSF competition training pipeline.")
    parser.add_argument("--config", type=str, required=True, default=None, help="Path to an experiment global config (.yml).")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--tag", type=str, default="experiment", help="Current train loop launch tag (used for saving).")
    parser.add_argument("--use-valid", action="store_true", help="Use validation set for model evaluation while training.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config path not found: {config_path}") 
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint_path and not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}") 

    # config
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = Config.from_dict(config_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config.seed)

    # data
    if args.use_valid:
        train_loader = get_dataloader(config, split="train", back_offset=config.horizon)
        valid_loader = get_dataloader(config, split="valid", back_offset=0)
    else:
        train_loader = get_dataloader(config, split="train", back_offset=0)
        valid_loader = None

    # model / optimizer / scheduler / loss
    model = get_model(
        config,
        enc_in_size=train_loader.dataset.enc_dim,
        dec_in_size=train_loader.dataset.dec_dim,
        horizon=train_loader.dataset.horizon,
        lookback=train_loader.dataset.lookback,
    ).to(device)
    optimizer = get_optimizer(config, model.parameters())
    scheduler = get_scheduler(config, optimizer)
    criterion = get_loss(config)

    # exp moving avg 
    ema = ExponentialMovingAverage( 
        model.parameters(), decay=config.ema_decay 
    ) if config.ema_decay else None
    print(f"ExponentialMovingAverage is used: {ema is not None}")

    # load checkpoint
    curr_epoch = 0
    train_losses = []
    val_losses = []
    if args.checkpoint is not None:
        model, state_dict = load_checkpoint(
            path=checkpoint_path, 
            model=model,
            map_location=device,
        )
        optimizer.load_state_dict(state_dict["optim"])
        if state_dict.get("scheduler") is not None:
            scheduler.load_state_dict(state_dict["scheduler"])
        if ema is not None and state_dict.get("ema") is not None:
            ema.load_state_dict(state_dict["ema"])

        curr_epoch = state_dict["epoch"] + 1
        train_losses = state_dict["train_loss"]
        val_losses = state_dict["val_loss"]
        print(f"Resumed '{checkpoint_path}' at epoch {curr_epoch}.")

    # train
    train_losses, val_losses = train_rnn(
        model=model, 
        optimizer=optimizer, 
        criterion=criterion, 
        config=config,
        train_loader=train_loader, 
        valid_loader=valid_loader, 
        scheduler=scheduler, 
        ema=ema,
        curr_epoch=curr_epoch, 
        experiment_tag=args.tag,
        train_losses=train_losses, 
        val_losses=val_losses,
        device=device
    )
    visualize_training(train_losses, val_losses, title=args.tag, save=True)