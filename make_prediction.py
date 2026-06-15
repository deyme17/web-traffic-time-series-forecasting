from typing import List, Tuple, Optional
from contextlib import nullcontext
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch_ema import ExponentialMovingAverage
from torch.utils.data import DataLoader

from utils import Config, load_checkpoint, set_seed
from models import get_model
from dataset import get_dataloader



def _predict(model: nn.Module,
             loader: DataLoader,
             ema: Optional[ExponentialMovingAverage] = None,
             device: torch.device = torch.device("cpu")) -> Tuple[np.ndarray, List[str]]:
    """Run inference over all pages."""
    model.eval()
    all_preds = []

    with (
        ema.average_parameters() if ema is not None else nullcontext(),
        torch.no_grad()
    ):
        for X in tqdm(loader, desc="Predicting"):
            enc_in = X["enc_input"].to(device)
            dec_in = X["dec_input"].to(device)
            mean = X["series_mean"].to(device)
            std = X["series_std"].to(device)

            out = model(enc_in, dec_in)

            log1p_pred = out * std[:, None] + mean[:, None]
            raw_pred = torch.expm1(log1p_pred).clamp(min=0.)

            all_preds.append(raw_pred.cpu().numpy())

    return np.concatenate(all_preds, axis=0)



def predict_checkpoint(checkpoint: Path, 
                       config: Config, 
                       loader: DataLoader, 
                       device: torch.device) -> np.ndarray:
    """Make prediction for a specific checkpoint."""
    model = get_model(
        config,
        enc_in_size=loader.dataset.enc_dim,
        dec_in_size=loader.dataset.dec_dim,
        horizon=loader.dataset.horizon,
        lookback=loader.dataset.lookback,
    ).to(device)
    model, state_dict = load_checkpoint(
        checkpoint,
        model,
        map_location=device
    )
    print(f"Loaded checkpoint: {checkpoint}")

    ema = None
    if config.ema_decay is not None and state_dict.get("ema") is not None:
        ema = ExponentialMovingAverage(model.parameters(), decay=config.ema_decay)
        ema.load_state_dict(state_dict["ema"])
        print("ExponentialMovingAverage is used for inferance.")

    return _predict(model, loader, ema, device)



def build_submission(preds: np.ndarray,
                     pages: List[str],
                     first_pred_date: str,
                     key_path: Path,
                     out_path: Path) -> None:
    """Merge predictions with key file and write submission CSV."""
    horizon = preds.shape[1]

    # dates
    dates = pd.date_range(start=first_pred_date, periods=horizon, freq="D")
    page_idx = {p: i for i, p in enumerate(pages)}

    # keys
    key_df = pd.read_csv(key_path)
    key_df[["Page", "date"]] = key_df["Page"].str.rsplit("_", n=1, expand=True)
    key_df["date"] = pd.to_datetime(key_df["date"])

    # pred
    rows = []
    for page, idx in page_idx.items():
        for t, d in enumerate(dates):
            rows.append((page, d, preds[idx, t]))
    pred_df = pd.DataFrame(rows, columns=["Page", "date", "Visits"])

    # merge
    merged = key_df.merge(pred_df, on=["Page", "date"], how="left")
    merged["Visits"] = merged["Visits"].fillna(0).clip(lower=0).round().astype(int)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged[["Id", "Visits"]].to_csv(out_path, index=False)

    missing = merged["Visits"].eq(0).sum()
    print(f"Submission written: {out_path} ({len(merged)} rows, {missing} zeros)")


# main section
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="WTTSF competition inference script.")
    parser.add_argument("--configs", nargs="+", required=True, help="Path to experiment config(s) (.yaml).")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Path to model checkpoint(s) (.pt).")
    parser.add_argument("--weights", nargs="+", help="Weight for each checkpoint.")
    parser.add_argument("--key", type=str, default="data/key_2.csv", help="Path to key_2.csv.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--out", type=str, default="submissions/submission.csv", help="Output CSV path.")
    args = parser.parse_args()

    config_paths = [Path(p) for p in args.configs]
    checkpoint_paths = [Path(p) for p in args.checkpoints]
    weights = np.array(list(map(float, args.weights))) if args.weights else None
    key_path = Path(args.key)
    out_path = Path(args.out)

    # validation
    if len(config_paths) != len(checkpoint_paths):
        raise ValueError(
            f"Expected equal number of configs and checkpoints "
            f"(got {len(config_paths)} configs and "
            f"{len(checkpoint_paths)} checkpoints)."
        )
    if weights is not None and len(weights) != len(config_paths):
        raise ValueError(
            f"Weights must match number of models "
            f"({len(weights)} vs {len(config_paths)})"
        )
    for p in (*config_paths, *checkpoint_paths, Path(args.key)):
        if not p.exists():
            raise FileNotFoundError(p)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    set_seed(args.seed)
        
    all_preds = []
    for config_path, checkpoint_path in zip(config_paths, checkpoint_paths):
        # config
        with open(config_path) as f:
            config = Config.from_dict(yaml.safe_load(f))

        # data
        predict_loader = get_dataloader(config, split="predict")

        # prediction
        pred = predict_checkpoint(checkpoint_path, config, predict_loader, device)
        all_preds.append(pred)
    
    # combine preds
    preds = np.stack(all_preds)
    if weights is None:
        preds = preds.mean(axis=0)
    else:
        preds = np.average(preds, axis=0, weights=weights)
    print(f"Predictions shape: {preds.shape} min={preds.min():.1f} max={preds.max():.1f}")

    # page names (same order as the dataset)
    pages = list(np.load(Path(config.data_dir) / "processed" / "page_meta.npz")["pages"])

    # first prediction date
    tmp = pd.read_csv(key_path)
    first_pred_date = tmp["Page"].str[-10:].min()

    build_submission(preds, pages, first_pred_date, key_path, out_path)