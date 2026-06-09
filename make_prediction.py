from typing import List
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import Config, load_checkpoint, set_seed
from models import get_model
from dataset import get_dataloader



def predict(model: nn.Module,
            loader: DataLoader,
            device: torch.device) -> tuple[np.ndarray, List[str]]:
    """Run inference over all pages."""
    model.eval()
    all_preds = []

    with torch.no_grad():
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
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config (.yaml).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument("--key", type=str, default="data/key_2.csv", help="Path to key_2.csv.")
    parser.add_argument("--out", type=str, default="submissions/submission.csv", help="Output CSV path.")
    args = parser.parse_args()

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    key_path = Path(args.key)
    out_path = Path(args.out)

    for p in (config_path, checkpoint_path, key_path):
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p}")

    with open(config_path) as f:
        config = Config.from_dict(yaml.safe_load(f))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    set_seed(config.seed)

    # data
    predict_loader = get_dataloader(config, split="predict")

    # model
    model = get_model(
        config,
        enc_in_size=predict_loader.dataset.enc_dim,
        dec_in_size=predict_loader.dataset.dec_dim,
        horizon=predict_loader.dataset.horizon,
    ).to(device)
    model, _ = load_checkpoint(path=checkpoint_path, model=model, map_location=device)
    print(f"Loaded checkpoint: {checkpoint_path}")

    # inference
    preds = predict(model, predict_loader, device)
    print(f"Predictions shape: {preds.shape} min={preds.min():.1f} max={preds.max():.1f}")

    # page names (same order as the dataset)
    pages = list(np.load(Path(config.data_dir) / "processed" / "page_meta.npz")["pages"])

    # first prediction date
    tmp = pd.read_csv(key_path)
    first_pred_date = tmp["Page"].str[-10:].min()

    build_submission(preds, pages, first_pred_date, key_path, out_path)