from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from utils.constants import ENC_DIM, DEC_DIM



class WTTSF_Dataset(Dataset):
    """Efficient Web Traffic Time-Series Forecasting dataset."""
    def __init__(self, 
                 data_root: Path, 
                 lookback: int = 500, 
                 horizon: int = 62, 
                 split: str = "train",
                 back_offset: int = 0, 
                 transforms= None, 
                 seed: Optional[int] = None):
        """    
        Args:
            data_root: directory that contains the processed data.
            lookback: encoder window length in days (default 500)
            horizon: decoder / prediction window (default 62)
            split: splitting strategy 'train' | 'val' | 'predict'
                - train: random start in [0, free_space)
                - val: fixed start = data_days - lookback - back_offset
                - predict: same as val but y_hits are future (NaN pad)
            back_offset: days to leave at the end of training window for validation
            transforms: optional callable applied to the output dict.
            seed: random seed for reproducibility.
        """
        super().__init__()
        assert split in ("train", "val", "predict")
        self.lookback = lookback
        self.horizon = horizon
        self.split = split
        self.back_offset = back_offset
        self.transforms = transforms

        proc = Path(data_root) / "processed"
        meta = json.loads((proc / "meta.json").read_text())
        self.n_pages = meta["n_pages"]
        self.n_days = meta["n_days"]
        self.n_days_full = meta["n_days_full"]

        # memory-mapped arrays
        self._hits = np.load(proc / "hits.npy", mmap_mode="r")
        self._nan_mask = np.load(proc / "nan_mask.npy", mmap_mode="r")

        # time-dependent arrays [n_days_full, N_LAGS/N_TEMPORAL] 
        self._lagged_ix = np.load(proc / "lagged_ix.npy")
        self._lag_valid = np.load(proc / "lag_valid.npy")
        self._temporal = np.load(proc / "temporal.npy")

        # per-series features [n_pages, *]
        pm = np.load(proc / "page_meta.npz")
        self._starts = pm["starts"].astype(np.int32)
        self._ends = pm["ends"].astype(np.int32)

        self._page_features = np.concatenate([
            pm["year_autocorr"][:, None],
            pm["quarter_autocorr"][:, None],
            pm["page_mean"][:, None],
            pm["page_std"][:, None],
            pm["page_vc"][:, None],
            pm["cat_lang"],
            pm["cat_access"],
            pm["cat_agent"],
            pm["cat_site"],
        ], axis=1).astype(np.float32)

        # min valid start for train
        self._min_start = 0 # max(meta.get("lag_days", [365])) # <- believe to _nan_mask

        self._rng = np.random.default_rng(seed)

        if split in ("val", "predict"):
            self._fixed_start = self.n_days - self.lookback - back_offset
            assert self._fixed_start >= 0

    @property
    def enc_dim(self) -> int:
        return ENC_DIM

    @property
    def dec_dim(self) -> int:
        return DEC_DIM
    
    @property
    def lookback(self) -> int:
        return self.lookback
    
    @property
    def horizon(self) -> int:
        return self.horizon

    def __len__(self) -> int:
        return self.n_pages

    def __getitem__(self, idx: int) -> dict:
        n_window = self.lookback + self.horizon
        if self.split == "train":
            lo = self._min_start
            hi = self.n_days - n_window - self.back_offset
            assert hi > lo, "lookback + horizon + back_offset too large"
            start = int(self._rng.integers(lo, hi))
        else:
            start = self._fixed_start
        end = start + n_window

        # hits [n_days]
        full_hits = np.array(self._hits[idx], dtype=np.float32)

        raw_window = full_hits[start : end]
        x_raw = raw_window[:self.lookback]
        y_raw = raw_window[self.lookback:]

        # per-series normalization
        mean = float(x_raw.mean())
        std = float(x_raw.std())
        if std < 1e-6: std = 1.

        x_hits = (x_raw - mean) / std  # [lookback]
        y_hits = (y_raw - mean) / std  # [horizon]

        # lags [n_window, N_LAGS]
        lags_ix = self._lagged_ix[start:end]
        lags_val = self._lag_valid[start:end]

        padded = np.concatenate([full_hits, np.zeros(self.horizon, dtype=np.float32)])
        lagged = padded[np.maximum(lags_ix, 0)]
        lagged = np.where(lags_val, (lagged - mean) / std, 0.).astype(np.float32)

        x_lagged = lagged[:self.lookback]
        x_lag_valid = lags_val[:self.lookback].astype(np.float32)
        y_lagged = lagged[self.lookback:]
        y_lag_valid = lags_val[self.lookback:].astype(np.float32)

        # temporal features [n_window, N_TEMPORAL]
        temp = self._temporal[start : end]
        x_temp = temp[:self.lookback]
        y_temp = temp[self.lookback:]

        # page features
        pf = self._page_features[idx]           # [N_PAGE]
        pf_x = np.tile(pf, (self.lookback, 1))  # [lookback, N_PAGE]
        pf_y = np.tile(pf, (self.horizon, 1))   # [horizon,  N_PAGE]

        # assemble
        enc_input = np.concatenate([
            x_hits[:, None],    # [lookback, 1]
            x_lagged,           # [lookback, N_LAGS]
            x_lag_valid,        # [lookback, N_LAGS]
            x_temp,             # [lookback, N_TEMPORAL]
            pf_x,               # [lookback, N_PAGE]
        ], axis=1).astype(np.float32)   # [lookback, ENC_DIM]

        dec_input = np.concatenate([
            y_lagged,           # [horizon, N_LAGS]
            y_lag_valid,        # [horizon, N_LAGS]
            y_temp,             # [horizon, N_TEMPORAL]
            pf_y,               # [horizon, N_PAGE]
        ], axis=1).astype(np.float32)   # [horizon, DEC_DIM]

        # target mask [horizon]
        target_mask = ~self._nan_mask[idx, start + self.lookback : end]

        sample = {
            "enc_input": torch.from_numpy(enc_input),
            "dec_input": torch.from_numpy(dec_input),
            "target": torch.from_numpy(y_hits),
            "target_mask": torch.from_numpy(target_mask),
            "series_mean": torch.tensor(mean, dtype=torch.float32),
            "series_std": torch.tensor(std, dtype=torch.float32),
        }
        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample