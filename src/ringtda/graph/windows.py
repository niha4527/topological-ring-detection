"""Sliding windows and per-pair aggregation (proposal v3 §3.1, "Aggregation").

Input transaction tables need columns: timestamp, src, dst, amount.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


def iter_windows(
    tx: pd.DataFrame, length: float, stride: float, start: float | None = None, end: float | None = None
) -> Iterator[tuple[float, float, pd.DataFrame]]:
    """Yield (window_start, window_end, transactions in [start, end))."""
    if length <= 0 or stride <= 0:
        raise ValueError("length and stride must be positive")
    t = tx["timestamp"].to_numpy()
    lo = t.min() if start is None else start
    hi = t.max() if end is None else end
    w = lo
    while w < hi:
        mask = (t >= w) & (t < w + length)
        yield w, w + length, tx.loc[mask]
        w += stride


def _gap_cv(ts: np.ndarray) -> float:
    """Coefficient of variation of inter-transaction gaps.

    Needs at least 2 gaps (3 transactions); otherwise NaN. With a single gap
    the std is trivially 0, which would falsely look perfectly regular.
    """
    if len(ts) < 3:
        return np.nan
    gaps = np.diff(np.sort(ts))
    mean = gaps.mean()
    if mean <= 0:
        return 0.0
    return float(gaps.std() / mean)


def aggregate_pairs(tx: pd.DataFrame) -> pd.DataFrame:
    """One row per ordered pair (src, dst): volume, count, gap_cv, first/last time."""
    if tx.empty:
        return pd.DataFrame(columns=["src", "dst", "volume", "count", "gap_cv", "t_first", "t_last"])
    g = tx.groupby(["src", "dst"], sort=False)
    out = g.agg(
        volume=("amount", "sum"),
        count=("amount", "size"),
        t_first=("timestamp", "min"),
        t_last=("timestamp", "max"),
    )
    out["gap_cv"] = g["timestamp"].apply(lambda s: _gap_cv(s.to_numpy()))
    return out.reset_index()
