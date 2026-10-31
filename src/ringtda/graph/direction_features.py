"""D features: direction-aware and behavioural features of a candidate ring (proposal v3 §3.4).

Computed from the ring's own edges and members on the window graph. Inputs are only the weighted
window graph (edge attributes written by `weights.to_digraph`) and a Candidate: no planted
structures, labels, attempt ids or scenario names, so leakage is impossible by signature.

Ring birth (`D_birth`, the max edge weight on the ring) lives here, not in T (§3.4).

Fallbacks (no NaN / inf):
    gap_cv is undefined for a pair with < 3 transactions; such an edge takes the window's median
    defined gap_cv (`gap_cv_fill`), or 0.0 if no pair of the window has one.
    volume ratio / CV: a ring whose maximum volume is 0 has ratio 1.0 and CV 0.0.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from ringtda.topology.candidates import Candidate

D_NAMES = [
    "D_length", "D_birth", "D_mean_w", "D_min_w",
    "D_rec_mean", "D_rec_min", "D_bal_mean", "D_bal_min", "D_reg_mean", "D_reg_min",
    "D_deg_imb_mean", "D_deg_imb_max",
    "D_log_volume", "D_vol_ratio", "D_vol_cv",
    "D_gap_cv_mean", "D_log_count_mean",
]  # fmt: skip


def gap_cv_fill(G: nx.DiGraph) -> float:
    """Median defined gap_cv over the window's edges (0.0 if none): the fallback for short pairs."""
    v = np.array([d["gap_cv"] for _, _, d in G.edges(data=True)], dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else 0.0


def direction_features(G: nx.DiGraph, candidate: Candidate, fill: float | None = None) -> dict[str, float]:
    """Flat dict of D features for one candidate (fixed key set `D_NAMES`).

    fill: gap_cv fallback for pairs with < 3 transactions; pass `gap_cv_fill(G)` computed once per
    window (computed here if omitted).
    """
    fill = gap_cv_fill(G) if fill is None else fill
    ed = [G.edges[e] for e in candidate.edges]
    w = np.array([d["weight"] for d in ed], dtype=float)
    vol = np.array([d["volume"] for d in ed], dtype=float)
    cnt = np.array([d["count"] for d in ed], dtype=float)
    gap = np.array([d["gap_cv"] for d in ed], dtype=float)
    gap = np.where(np.isfinite(gap), gap, fill)

    imb = []
    for v in candidate.nodes:
        i, o = G.in_degree(v), G.out_degree(v)
        imb.append(abs(i - o) / (i + o) if i + o else 0.0)

    vmax = vol.max()
    out = {
        "D_length": float(candidate.length),
        "D_birth": float(w.max()),
        "D_mean_w": float(w.mean()),
        "D_min_w": float(w.min()),
        "D_deg_imb_mean": float(np.mean(imb)),
        "D_deg_imb_max": float(np.max(imb)),
        "D_log_volume": float(np.log1p(vol.sum())),
        "D_vol_ratio": float(vol.min() / vmax) if vmax > 0 else 1.0,
        "D_vol_cv": float(vol.std() / vol.mean()) if vol.mean() > 0 else 0.0,
        "D_gap_cv_mean": float(gap.mean()),
        "D_log_count_mean": float(np.log(np.maximum(cnt, 1.0)).mean()),
    }
    for key, col in (("rec", "n_s_rec"), ("bal", "n_s_bal"), ("reg", "n_s_reg")):
        z = np.array([d[col] for d in ed], dtype=float)
        out[f"D_{key}_mean"], out[f"D_{key}_min"] = float(z.mean()), float(z.min())
    return {k: out[k] for k in D_NAMES}
