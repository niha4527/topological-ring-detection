"""Groups and fold assignments (README "Phase 3 design decisions", decision 1; proposal v3 §3.8).

A group is the unit that must never be split across folds:
  edge variant    union-find over candidates that share any edge, merged with the attempt id of every
                  positive candidate (so near-duplicate rings and a ring seen in overlapping windows
                  stay together; dedup is effectively off, see README)
  strict variant  as above, but candidates sharing any account are merged (sensitivity check)
Schemes: stratified group k-fold (primary), time-ordered check, HI<->LI transfer.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def assign_groups(edges, attempts, nodes=None, strict: bool = False) -> np.ndarray:
    """Group id (0..G-1) per candidate. edges: per-candidate edge tuples; attempts: attempt id per
    candidate (-1 = none); nodes: per-candidate account tuples, used when strict."""
    n = len(edges)
    uf = _UF(n)
    owner: dict = {}
    for i in range(n):
        keys = [("e", e) for e in edges[i]]
        if strict:
            keys += [("n", v) for v in nodes[i]]
        if attempts[i] >= 0:
            keys.append(("a", int(attempts[i])))
        for k in keys:
            if k in owner:
                uf.union(i, owner[k])
            else:
                owner[k] = i
    roots = np.array([uf.find(i) for i in range(n)])
    return pd.factorize(roots)[0]


def group_summary(groups: np.ndarray, y: np.ndarray) -> dict:
    g = pd.Series(groups)
    size = g.value_counts()
    pos_groups = set(groups[y == 1])
    return {"n_groups": int(g.nunique()), "largest_group_share": float(size.iloc[0] / len(g)) if len(g) else 0.0, "positive_groups": len(pos_groups)}


def stratified_group_cv(groups: np.ndarray, y: np.ndarray, k: int = 5, fallback_k: int = 3, repeats: int = 5, seed: int = 0) -> list[dict]:
    """Stratified group k-fold, `repeats` times with different seeds. Per repeat returns
    {repeat, k, fold (fold id per candidate), fold_without_positive}. If any fold has no positive
    candidate, that repeat is redone with fallback_k. Ambiguous (NaN) candidates count as non-positive
    for stratification but stay in their group's fold."""
    pos = (np.nan_to_num(y, nan=0.0) == 1).astype(int)
    out = []
    for r in range(repeats):
        for kk in (k, fallback_k):
            fold = np.full(len(groups), -1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv = StratifiedGroupKFold(n_splits=kk, shuffle=True, random_state=seed + r)
                for f, (_, te) in enumerate(cv.split(np.zeros(len(groups)), pos, groups)):
                    fold[te] = f
            empty = [f for f in range(kk) if pos[fold == f].sum() == 0]
            if not empty or kk == fallback_k:
                break
        out.append({"repeat": r, "k": kk, "fold": fold, "fold_without_positive": len(empty)})
    return out


def time_ordered_split(df: pd.DataFrame, groups: np.ndarray, train_max_day: float = 5.0, test_min_day: float = 7.0) -> pd.DataFrame:
    """Time-ordered check. A group's first-seen day is the earliest `t_seen` of its candidates:
    train if <= train_max_day, test if >= test_min_day, else gap (2 days by default). A window that
    holds both train-side and test-side candidates is dropped. Returns df with `group`, `side` in
    {train, test, gap, dropped}."""
    d = df.assign(group=groups)
    first = d.groupby("group")["t_seen"].transform("min")
    d["side"] = np.where(first <= train_max_day, "train", np.where(first >= test_min_day, "test", "gap"))
    sides = d.groupby("window_id")["side"].agg(lambda s: {"train", "test"} <= set(s))
    both = sides[sides].index
    d.loc[d["window_id"].isin(both), "side"] = "dropped"
    return d


def transfer_split(a: pd.DataFrame, b: pd.DataFrame, accounts_a: np.ndarray, accounts_b: np.ndarray) -> dict:
    """Train on `a`, test on `b` (or the reverse by swapping). Logs how many candidates of each side
    use any account that exists in both datasets (matched by bank_account key)."""
    shared = set(accounts_a) & set(accounts_b)
    def uses(df, accounts):
        ids = {i for i, k in enumerate(accounts) if k in shared}
        return int(sum(any(v in ids for v in nodes) for nodes in df["nodes"]))
    return {"n_shared_accounts": len(shared), "train_candidates_using_shared": uses(a, accounts_a), "test_candidates_using_shared": uses(b, accounts_b)}
