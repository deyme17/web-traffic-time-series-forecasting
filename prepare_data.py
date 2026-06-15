from typing import List, Dict, Tuple
import argparse
import json
import yaml
import math
import numpy as np
import datetime as dt
from pathlib import Path

from utils import (
    read_csv, normalize, set_seed, Config
)
from utils.constants import (
    LANGS, ACCESS, AGENTS, SITES, RE_PAGE, LAG_DAYS
)



# ======================================== PARSE PAGE NAME ========================================


def parse_pages(pages: List[str]) -> Dict[str, np.ndarray]:
    """
    Returns one-hot arrays for lang [n,8], access [n,4], agent [n,2], site [n,3].
    All normalised to zero-mean unit-variance across pages.
    """
    n = len(pages)
    lang = np.zeros((n, len(LANGS)), dtype=np.float32)
    access = np.zeros((n, len(ACCESS)), dtype=np.float32)
    agent = np.zeros((n, len(AGENTS)), dtype=np.float32)
    site = np.zeros((n, len(SITES)), dtype=np.float32)

    for i, p in enumerate(pages):
        m = RE_PAGE.match(p)
        if m:
            _, lg, wp, cm, _, acc, ag = m.groups()
            li = LANGS.index(lg) if lg in LANGS else LANGS.index("other")
            ai = ACCESS.index(acc) if acc in ACCESS else len(ACCESS) - 1
            gi = 0 if ag == "spider" else 1
            si = 0 if wp else (1 if cm else 2)
        else:
            li, ai, gi, si = LANGS.index("other"), len(ACCESS) - 1, 1, 0

        lang[i, li] = 1.
        access[i, ai] = 1.
        agent[i, gi] = 1.
        site[i, si] = 1.

    return {
        "cat_lang": normalize(lang),
        "cat_access": normalize(access),
        "cat_agent": normalize(agent),
        "cat_site": normalize(site),
    }



# ============================================ CLEANING ============================================


def winsorize(data: np.ndarray, winsor_k: float) -> np.ndarray:
    """
    Per-series spike capping: values > median + winsor_k * MAD -> capped.
    Operates on raw (pre-log) data in-place copy. NaN ignored.
    """
    out = data.copy()
    for i in range(len(out)):
        row = out[i]
        vals = row[~np.isnan(row)]
        if len(vals) < 10:
            continue
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        if mad == 0:
            continue
        hi = med + winsor_k * mad
        row[row > hi] = hi
    return out


def interpolate_gaps(data: np.ndarray, max_gap_interpolate: int) -> np.ndarray:
    """
    For each series interpolate missing days if gaps <= max_gap_interpolate.
    Returns cleaned data (NaN filled).
    """
    out = data.copy()
    n_pages, n_days = out.shape

    for i in range(n_pages):
        row = out[i]
        nans = np.where(np.isnan(row))[0]
        if len(nans) == 0:
            continue

        # group consecutive NaN runs
        groups = []
        cur = [nans[0]]
        for idx in nans[1:]:
            if idx == cur[-1] + 1:
                cur.append(idx)
            else:
                groups.append(cur)
                cur = [idx]
        groups.append(cur)

        for g in groups:
            left, right = g[0] - 1, g[-1] + 1
            if len(g) <= max_gap_interpolate and left >= 0 and right < n_days:
                lv, rv = float(row[left]), float(row[right])
                for k, idx in enumerate(g):
                    t = (k + 1) / (len(g) + 1)
                    row[idx] = lv + t * (rv - lv)
            else:
                row[g] = 0.

    return out


def drop_dead_pages(data: np.ndarray,
                    pages: List[str],
                    check_window: int = 180,
                    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Drops pages that are truly dead:
    - never had any traffic (all NaN or zero across entire series), OR
    - no traffic in last `check_window` days
    """
    never_alive = np.all(np.isnan(data) | (data == 0), axis=1)
    window = data[:, -check_window:]
    dead_recently = np.all(np.isnan(window) | (window == 0), axis=1)
    keep_mask = ~(never_alive | dead_recently)
    print(f"\tnever alive: {never_alive.sum()}, dead in last {check_window}d: {(dead_recently & ~never_alive).sum()}")
    print(f"\tdropped {(~keep_mask).sum()} pages, kept {keep_mask.sum()}")
    return data[keep_mask], [p for p, k in zip(pages, keep_mask) if k], keep_mask



# ============================================ TEMPORAL ============================================


def make_temporal(n_days_full: int, first_date_str: str = "2015-07-01") -> np.ndarray:
    """
    Returns float32 [n_days_full, 4]: sin_dow, cos_dow, sin_month, cos_month.
    first_date_str: date of column index 0 in the original CSV.
    """
    start = dt.date.fromisoformat(first_date_str)
    out = np.zeros((n_days_full, 4), dtype=np.float32)
    for i in range(n_days_full):
        d = start + dt.timedelta(days=i)
        dow_rad = d.weekday() / 7. * 2 * math.pi
        month_rad = (d.month - 1) / 12. * 2 * math.pi
        out[i, 0] = math.sin(dow_rad)
        out[i, 1] = math.cos(dow_rad)
        out[i, 2] = math.sin(month_rad)
        out[i, 3] = math.cos(month_rad)
    return out



# ============================================== LAGS ==============================================


def make_lag_indexes(n_days_full: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        lagged_ix (int32) [n_days_full, 4]: source day index per lag, 0 when invalid
        lag_valid (bool) [n_days_full, 4]: True = lag is within range
    """
    ix = np.zeros((n_days_full, len(LAG_DAYS)), dtype=np.int32)
    valid = np.zeros((n_days_full, len(LAG_DAYS)), dtype=bool)
    for j, ld in enumerate(LAG_DAYS):
        for i in range(n_days_full):
            src = i - ld
            if src >= 0:
                ix[i, j] = src
                valid[i, j] = True
    return ix, valid



# ========================================= PAGE STATISTIC =========================================


def page_stats(hits: np.ndarray, horizon: int) -> Dict[str, np.ndarray]:
    """
    Computed on hits[:, :-horizon] (train portion only).
    Returns normalised: page_mean, page_std, variation_coeff.
    """
    train = hits[:, :-horizon] if horizon > 0 else hits
    mean = train.mean(axis=1)
    std = train.std(axis=1)
    vc = np.where(mean > 1e-6, std / mean, 0.) # Pearson's coefficient of variation
    return {"page_mean": normalize(mean), "page_std": normalize(std), "page_vc": normalize(vc)}



# ========================================= AUTOCORRELATION =========================================


def _single_ac(s: np.ndarray, lag: int) -> float:
    a, b = s[lag:], s[:-lag]
    da, db = a - a.mean(), b - b.mean()
    denom = math.sqrt((da**2).sum() * (db**2).sum())
    return float((da * db).sum() / denom) if denom > 0 else 0.


def batch_autocorr(hits: np.ndarray, lag: int,
                   starts: np.ndarray, ends: np.ndarray,
                   min_ratio: float = 1.5) -> np.ndarray:
    n = len(hits)
    corr = np.zeros(n, dtype=np.float32)
    for i in range(n):
        s, e = int(starts[i]), int(ends[i])
        if (e - s) / lag > min_ratio:
            row = hits[i, s : e]
            # !!! SMOOTH AUTOCORRELATION !!!
            corr[i] = (0.50 * _single_ac(row, lag) +
                       0.25 * _single_ac(row, max(lag - 1, 1)) +
                       0.25 * _single_ac(row, lag + 1))
        if i % 20_000 == 0:
            print(f"\tautocorr lag={lag}: {i}/{n}", end="\r", flush=True)
    print()
    return corr


def find_starts_ends(hits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid = hits > 0
    starts = np.argmax(valid, axis=1).astype(np.int32)
    # argmax returns 0 if all False - fix those
    starts[~valid.any(axis=1)] = hits.shape[1]
    # last valid day
    ends = (hits.shape[1] - 1 - np.argmax(valid[:, ::-1], axis=1)).astype(np.int32)
    ends[~valid.any(axis=1)] = 0
    return starts, ends



# =========================================== MAIN SECTION ===========================================


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, default=None, 
                        help="Path to an experiment global config (.yml)")
    args = parser.parse_args()
    # config
    config_path = Path(args.config) if args.config else None
    if config_path is not None and config_path.exists():
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        config = Config.from_dict(config_dict)
        set_seed(config.seed)
    else:
        config = Config()

    out_dir = Path(config.data_dir) / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = config.data_dir / "train_2.csv" \
        if (config.data_dir / "train_2.csv").exists() else None
    assert csv_path, "train_2.csv not found"

    # read data
    pages, raw = read_csv(csv_path)

    # dead pages
    raw, pages, _ = drop_dead_pages(raw, pages)

    # anomalies
    print("Winsorizing...")
    raw = winsorize(raw, config.winsor_k)

    # handle missings
    print("Interpolating gaps...")
    raw = interpolate_gaps(raw, config.max_gap_interpolate)
    nan_mask = np.isnan(raw)

    # log1p
    hits = np.log1p(np.where(nan_mask, 0., raw)).astype(np.float32)
    del raw

    n_pages, n_days = hits.shape
    n_days_full = n_days + config.add_days

    # save hits + nan_mask
    print("Saving hits.npy...")
    np.save(out_dir / "hits.npy", hits)
    np.save(out_dir / "nan_mask.npy", nan_mask)

    # lag indexes
    print("Building lag indexes...")
    lagged_ix, lag_valid = make_lag_indexes(n_days_full)
    np.save(out_dir / "lagged_ix.npy", lagged_ix)
    np.save(out_dir / "lag_valid.npy", lag_valid)

    # temporal
    print("Building temporal features...")
    temporal = make_temporal(n_days_full, config.first_date)
    np.save(out_dir / "temporal.npy", temporal)

    # page stats + categorical + autocorr
    print("Computing page stats...")
    stats = page_stats(hits, config.horizon)

    print("Parsing page names...")
    cats = parse_pages(pages)

    starts, ends = find_starts_ends(hits)
    print("Computing year autocorr...")
    yr_ac = batch_autocorr(hits, 365, starts, ends, 1.5)
    print("Computing quarter autocorr...")
    qt_ac = batch_autocorr(hits, 91,  starts, ends, 2.0)

    # saving
    np.savez(
        out_dir / "page_meta.npz",
        pages = np.array(pages),
        starts = starts,
        ends = ends,
        year_autocorr = normalize(yr_ac),
        quarter_autocorr = normalize(qt_ac),
        **stats,
        **cats,
    )
    # metadata
    meta = dict(
        n_pages = n_pages,
        n_days = n_days,
        n_days_full = n_days_full,
        horizon = config.horizon,
        lag_days = LAG_DAYS,
        first_date = config.first_date,
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nDone -> {out_dir}/")
    print(f"\thits: {hits.shape} {hits.nbytes/1e6:.1f} MB")
    print(f"\ttemporal: {temporal.shape}")
    print(f"\tlagged_ix: {lagged_ix.shape} valid from day {LAG_DAYS[-1]}")


if __name__ == "__main__":
    main()