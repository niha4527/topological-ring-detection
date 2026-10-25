"""Edge suspiciousness scores and filtration weights (proposal v3 §3.1).

All scores are in [0, 1], higher = more suspicious:
    s_rec  pair reciprocity      1 - |V_ab - V_ba| / (V_ab + V_ba)
    s_bal  node flow balance     min(beta(a), beta(b)),  beta(x) = 1 - |in-out|/(in+out)
    s_reg  timing regularity     1 / (1 + CV of inter-transaction gaps)

Each score is rank-normalised within the window, then
    weight = 1 - mean(normalised scores)
so LOWER weight = MORE suspicious (sublevel-set filtration: suspicious edges enter first).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

SCORES = ("s_rec", "s_bal", "s_reg")


def rank_normalize(x: pd.Series) -> pd.Series:
    """Map values to [0, 1] by rank. Ties take the LOWEST rank of their group.

    Using the lowest rank matters: most edges have s_rec == 0 (no reverse
    trades). With average ranks, that big tied block would land mid-scale and
    look moderately suspicious; with lowest ranks it maps to 0.
    """
    n = len(x)
    if n <= 1:
        return x.astype(float).clip(0, 1)
    r = x.rank(method="min")
    return (r - 1) / (n - 1)


def compute_scores(pairs: pd.DataFrame) -> pd.DataFrame:
    """Add raw s_rec, s_bal, s_reg columns to an aggregate_pairs() table."""
    p = pairs.copy()

    # pair reciprocity
    rev = p[["src", "dst", "volume"]].rename(columns={"src": "dst", "dst": "src", "volume": "rev_volume"})
    p = p.merge(rev, on=["src", "dst"], how="left")
    p["rev_volume"] = p["rev_volume"].fillna(0.0)
    tot = p["volume"] + p["rev_volume"]
    p["s_rec"] = 1 - (p["volume"] - p["rev_volume"]).abs() / tot

    # node flow balance
    out_v = p.groupby("src")["volume"].sum()
    in_v = p.groupby("dst")["volume"].sum()
    nodes = out_v.index.union(in_v.index)
    out_v = out_v.reindex(nodes, fill_value=0.0)
    in_v = in_v.reindex(nodes, fill_value=0.0)
    beta = 1 - (in_v - out_v).abs() / (in_v + out_v)
    p["s_bal"] = np.minimum(p["src"].map(beta).to_numpy(), p["dst"].map(beta).to_numpy())

    # timing regularity; pairs with < 3 transactions get the window median
    reg = 1 / (1 + p["gap_cv"])
    med = reg.median()
    p["s_reg"] = reg.fillna(0.5 if np.isnan(med) else med)
    return p


def edge_weights(pairs: pd.DataFrame, scores: tuple[str, ...] = SCORES) -> pd.DataFrame:
    """Scores + rank-normalised scores (n_*) + combined `weight`.

    `scores` selects which components enter the weight (for ablations,
    e.g. scores=("s_bal",)).
    """
    p = compute_scores(pairs)
    for s in SCORES:
        p["n_" + s] = rank_normalize(p[s])
    p["weight"] = 1 - p[["n_" + s for s in scores]].mean(axis=1)
    return p


def to_digraph(weighted_pairs: pd.DataFrame) -> nx.DiGraph:
    """Directed graph with all pair columns as edge attributes."""
    G = nx.DiGraph()
    attrs = [c for c in weighted_pairs.columns if c not in ("src", "dst")]
    for row in weighted_pairs.itertuples(index=False):
        d = row._asdict()
        G.add_edge(d["src"], d["dst"], **{k: d[k] for k in attrs})
    return G
