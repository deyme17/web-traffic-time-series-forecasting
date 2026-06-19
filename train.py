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




def _run_validation(model: nn.Module,
                    valid_loader: DataLoader,
                    criterion: nn.Module,
                    ema: Optional[ExponentialMovingAverage],
                    device: torch.device) -> float:
    """Run a full pass over valid_loader, return mean loss."""
    model.eval()
    val_loss = 0.
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

    return val_loss / len(valid_loader)



def train_rnn(model: nn.Module, 
              optimizer: optim.Optimizer, 
              criterion: nn.Module, 
              config: Config,
              train_loader: DataLoader, 
              valid_loader: Optional[DataLoader], 
              scheduler: Optional[lrs.LRScheduler], 
              ema: Optional[ExponentialMovingAverage],
              curr_step: int = 0, 
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
    step_count = curr_step
    update_steps = config.update_steps or config.steps
    window_train_loss = 0.
    window_n = 0

    # TRAIN
    model.train()
    train_iter = iter(train_loader)
    pbar = tqdm(total=config.steps, initial=step_count, desc="Train")

    while step_count < config.steps:
        try:
            X = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            X = next(train_iter)

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

        step_count += 1
        pbar.update(1)
        window_train_loss += loss.item()
        window_n += 1

        # update
        is_update_step = (step_count % update_steps == 0)
        is_last_step = (step_count >= config.steps)

        if is_update_step or is_last_step:
            train_loss = window_train_loss / max(window_n, 1)
            train_losses.append(train_loss)
            window_train_loss, window_n = 0., 0

            # EVAL
            if valid_loader is not None:
                val_loss = _run_validation(model, valid_loader, criterion, ema, device)
                val_losses.append(val_loss)
                model.train()
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
            pbar.clear()
            print(f"[Step: {step_count}/{config.steps}] Train Loss: {train_loss:.3f} | Val Loss: {val_str} | lr: {lr:.5f}")
            pbar.refresh()

            # checkpoint
            if val_loss < best_val_loss or valid_loader is None:
                if step_count >= config.warmup_steps:
                    with ema.average_parameters() if ema is not None else nullcontext():
                        save_checkpoint(
                            model=model, 
                            optim=optimizer,
                            scheduler=scheduler,
                            ema=ema,
                            train_loss=train_losses, 
                            val_loss=val_losses, 
                            step=step_count,
                            save_path=config.checkpoints_dir / f"{experiment_tag}_s{step_count}_checkpoint.pt"
                        )
                best_val_loss = val_loss
                patient_level = 0
            else:
                patient_level += 1

            # early stopping
            if config.patience is not None and patient_level >= config.patience:
                break

    pbar.close()
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
    curr_step = 0
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

        curr_step = state_dict["step"]
        train_losses = state_dict["train_loss"]
        val_losses = state_dict["val_loss"]
        print(f"Resumed '{checkpoint_path}' at step {curr_step}.")

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
        curr_step=curr_step, 
        experiment_tag=args.tag,
        train_losses=train_losses, 
        val_losses=val_losses,
        device=device
    )
    visualize_training(train_losses, val_losses, title=args.tag, save=True)