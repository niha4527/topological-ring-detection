"""IBM AML pattern-behaviour audit (proposal v3 §3.7, §4.10). Measurement only.

For the edges of a window (ordered pairs), do the wash-trading scores s_rec, s_bal, s_reg and the combined
weight separate pairs backed by laundering transactions from background pairs, per laundering pattern type?

Groups of pairs (a pair can be in several):
    <pattern type>   pairs carrying >= 1 transaction of an attempt of that type
    unattributed     pairs carrying >= 1 laundering row that belongs to no attempt
    background       pairs with no laundering transaction at all
Direction: a higher score or a LOWER weight is more suspicious.
Ties: the weights tie near 1.0, so besides the quantiles each summary gives the share of the group's values
exactly equal to the window median and to the window's most common value.
Decile share: share of the group beyond the 10% most suspicious tail of the window's pairs, strict
(value > q90 for scores, value < q10 for weights), so ties never inflate it; `decile_share_all` is the same
share over all pairs of the window.
`weight_bal_only` = 1 - n_s_bal, the balance-only variant named in v3 (equal to
edge_weights(pairs, scores=("s_bal",)).weight); it is reported, not used by the pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

COLUMNS = ["s_rec", "s_bal", "s_reg", "n_s_rec", "n_s_bal", "n_s_reg", "weight", "weight_bal_only"]
LOWER_IS_SUSPICIOUS = {"weight", "weight_bal_only"}
SPECIAL = ("unattributed",)


def add_balance_only_weight(pairs: pd.DataFrame) -> pd.DataFrame:
    return pairs.assign(weight_bal_only=1.0 - pairs["n_s_bal"])


def _key(src, dst, n: int) -> np.ndarray:
    return np.asarray(src, dtype=np.int64) * n + np.asarray(dst, dtype=np.int64)


def edge_membership(pairs: pd.DataFrame, tx: pd.DataFrame) -> dict[str, np.ndarray]:
    """{group: boolean mask over `pairs` rows}. tx needs src, dst, pattern, laundering (IBM loader output)."""
    n = int(max(pairs["src"].max(), pairs["dst"].max(), tx["src"].max(), tx["dst"].max())) + 1
    pk = _key(pairs["src"], pairs["dst"], n)
    L = tx[tx["laundering"]]
    lk = _key(L["src"], L["dst"], n)
    out = {}
    for t in sorted(L["pattern"].unique()):
        rows = L[L["pattern"] == t]
        out[t] = np.isin(pk, _key(rows["src"], rows["dst"], n))
    out["background"] = ~np.isin(pk, lk)
    return out


def auc(values, in_group, in_background, lower_is_suspicious: bool = False) -> float:
    """P(group value more suspicious than background value) + 0.5 P(tie): Mann-Whitney with average ranks."""
    v = np.asarray(values, dtype=float)
    v = -v if lower_is_suspicious else v
    g, b = np.asarray(in_group, bool), np.asarray(in_background, bool)
    n1, n0 = int(g.sum()), int(b.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(np.concatenate([v[g], v[b]]))
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def summarize(pairs: pd.DataFrame, groups: dict[str, np.ndarray], window: str, columns=COLUMNS) -> pd.DataFrame:
    """One row per (group, column). `pairs` has the score columns and weight_bal_only; `groups` from edge_membership."""
    bg = groups["background"]
    rows = []
    for col in columns:
        x = pairs[col].to_numpy(float)
        lower = col in LOWER_IS_SUSPICIOUS
        med = float(np.median(x))
        vc = pd.Series(x).value_counts()
        mode = float(vc.index[0])
        thr = np.quantile(x, 0.10 if lower else 0.90)
        beyond = (x < thr) if lower else (x > thr)
        for name, m in groups.items():
            v = x[m]
            if len(v) == 0:
                continue
            q = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
            rows.append(
                {
                    "window": window, "group": name, "score": col, "n_pairs": int(m.sum()),
                    "p5": q[0], "p25": q[1], "median": q[2], "p75": q[3], "p95": q[4],
                    "tie_window_median": float((v == med).mean()), "tie_window_mode": float((v == mode).mean()),
                    "auc_vs_background": float("nan") if name == "background" else auc(x, m, bg, lower),
                    "decile_share": float(beyond[m].mean()), "decile_share_all": float(beyond.mean()),
                }
            )  # fmt: skip
    return pd.DataFrame(rows)


def imputation_counts(pairs: pd.DataFrame, groups: dict[str, np.ndarray], window: str) -> pd.DataFrame:
    """Pairs whose gap_cv is undefined (< 3 transactions) and whose s_reg therefore uses the window median."""
    imp = pairs["gap_cv"].isna().to_numpy()
    return pd.DataFrame(
        [{"window": window, "group": g, "n_pairs": int(m.sum()), "n_imputed": int(imp[m].sum()), "share_imputed": float(imp[m].mean()) if m.any() else float("nan")} for g, m in groups.items()]
    )


def ring_scores(pairs: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Per candidate (columns edges, kind, y): birth (max edge weight), mean/min of n_s_bal, n_s_rec, n_s_reg
    over the ring's edges, from the window's pairs table (same quantities as the D columns, no feature code)."""
    idx = pairs.set_index(["src", "dst"])[["weight", "n_s_rec", "n_s_bal", "n_s_reg"]]
    rows = []
    for edges, kind, y in zip(candidates["edges"], candidates["kind"], candidates["y"]):
        e = idx.loc[list(edges)]
        rows.append({"kind": kind, "y": y, "birth": e["weight"].max(), **{f"{c}_mean": e[c].mean() for c in ("n_s_rec", "n_s_bal", "n_s_reg")}, **{f"{c}_min": e[c].min() for c in ("n_s_rec", "n_s_bal", "n_s_reg")}})
    return pd.DataFrame(rows)
