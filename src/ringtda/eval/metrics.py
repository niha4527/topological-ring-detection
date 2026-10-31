"""Metrics for the classification harness: average precision and grouped bootstrap (proposal v3 §3.8).

Average precision (PR-AUC) follows scikit-learn's definition (step-wise, tied scores form one threshold).
The grouped bootstrap resamples GROUPS with replacement (a candidate's weight is its group's multiplicity),
because candidates of one group (rings sharing an edge or attempt) are not independent. The same resamples
are reused across feature sets and models, so paired differences are computed inside the same resamples.
"""

from __future__ import annotations

import numpy as np


def average_precision(y, score, weight=None) -> float:
    """AP = sum_k (R_k - R_{k-1}) P_k over the distinct score thresholds; weight = integer multiplicities."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(score, dtype=float)
    w = np.ones(len(y)) if weight is None else np.asarray(weight, dtype=float)
    o = np.argsort(-s, kind="stable")
    y, s, w = y[o], s[o], w[o]
    ends = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    tp, seen = np.cumsum(w * y)[ends], np.cumsum(w)[ends]
    if tp[-1] == 0:
        return float("nan")
    prec = np.divide(tp, seen, out=np.zeros_like(tp), where=seen > 0)  # zero-weight head: no recall gained there
    rec = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


class GroupedBootstrap:
    """`n_boot` resamples of the groups; `ap(y, score)` returns the AP of every resample at once."""

    def __init__(self, groups, n_boot: int = 1000, seed: int = 0):
        self.groups_, self.inv = np.unique(np.asarray(groups), return_inverse=True)
        rng = np.random.default_rng(seed)
        g = len(self.groups_)
        self.counts = np.stack([np.bincount(rng.integers(0, g, g), minlength=g) for _ in range(n_boot)]).astype(np.float32)
        self.n_boot = n_boot

    def ap(self, y, score) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        s = np.asarray(score, dtype=float)
        o = np.argsort(-s, kind="stable")
        ys, ss = y[o], s[o]
        W = self.counts[:, self.inv[o]]
        ends = np.r_[np.flatnonzero(np.diff(ss) != 0), len(ss) - 1]
        tp = np.cumsum(W * ys, axis=1)[:, ends]
        seen = np.cumsum(W, axis=1)[:, ends]
        total = tp[:, -1:]
        # a resample may miss the top-ranked candidates' groups: seen == 0 there, and no recall is gained
        prec = np.divide(tp, seen, out=np.zeros_like(tp), where=seen > 0)
        rec = np.divide(tp, total, out=np.zeros_like(tp), where=total > 0)
        ap = np.sum(np.diff(np.concatenate([np.zeros((len(W), 1)), rec], axis=1), axis=1) * prec, axis=1)
        ap[total[:, 0] == 0] = np.nan  # only a resample without any positive is undefined
        return ap


def ci(values, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    lo, hi = np.nanpercentile(values, [100 * a, 100 * (1 - a)])
    return float(lo), float(hi)


def mean_ap_with_ci(y, scores_by_repeat, bootstrap: GroupedBootstrap) -> dict:
    """Mean over repeats of the pooled out-of-fold AP, with the grouped-bootstrap CI of that mean."""
    pts = [average_precision(y, s) for s in scores_by_repeat]
    boot = np.mean([bootstrap.ap(y, s) for s in scores_by_repeat], axis=0)
    lo, hi = ci(boot)
    return {"ap": float(np.mean(pts)), "ci_low": lo, "ci_high": hi, "boot": boot, "ap_by_repeat": pts, "n_boot_nan": int(np.isnan(boot).sum())}


def paired_difference(res_a: dict, res_b: dict) -> dict:
    """A minus B inside the same resamples (both results must use the same GroupedBootstrap)."""
    d = res_a["boot"] - res_b["boot"]
    lo, hi = ci(d)
    return {"diff": res_a["ap"] - res_b["ap"], "ci_low": lo, "ci_high": hi, "share_boot_positive": float(np.nanmean(d > 0))}
