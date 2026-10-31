"""Next-window severity regression on SYNTHETIC rings (proposal v3 §3.5). Exploratory; nothing here is a gate.

Target: log1p of the planted ring's own wash volume in window t+1. A candidate of window t is matched to the same ring
in window t+1 by account set (the candidate's node set equals a positive candidate's node set of window t+1, same
planted structure). The generator decides persistence: the same planted rings run for the whole week.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ringtda.data.synthetic import SyntheticConfig, generate
from ringtda.eval.labels import label_candidates
from ringtda.features import full_feature_table

MODELS = ("ridge", "rf", "gbr")


def ring_volume(tx: pd.DataFrame, pid: int) -> float:
    """Planted wash volume of structure `pid` in the transactions `tx` (its own wash-ring rows only; decoys excluded)."""
    t = tx[(tx["pattern_id"] == pid) & (tx["pattern"] == "wash_ring")]
    return float(t["amount"].sum())


def match_next(nodes_t, pid_t, next_cands: list[tuple[frozenset, int]]):
    """True if a positive candidate of window t+1 has the same account set and the same planted structure."""
    return any(n == frozenset(nodes_t) and p == pid_t for n, p in next_cands)


def severity_pairs(cfg: SyntheticConfig, length: float = 86400.0, n_windows: int = 7, tau_quantile=None, max_len: int = 6, group_offset: int = 0) -> pd.DataFrame:
    """One row per (positive candidate of window t present again in window t+1): B/D/T features of window t, the target
    `y_next` = log1p(volume in t+1), `vol_next`, `vol_now`, ring id and a group id (seed-prefixed ring identity)."""
    tx, planted = generate(cfg)
    wins = [tx[(tx["timestamp"] >= w * length) & (tx["timestamp"] < (w + 1) * length)] for w in range(n_windows)]
    feats, cands = full_feature_table(wins, tau_quantile=tau_quantile, max_len=max_len, return_candidates=True)
    labs = {w: label_candidates(cands[w], wins[w], planted) for w in range(n_windows)}
    scen = {p.pattern_id: p.scenario for p in planted}
    pos = {w: [(frozenset(c.nodes), int(pid)) for c, y, pid in zip(cands[w], labs[w].y, labs[w].pattern_id) if y == 1] for w in range(n_windows)}
    rows = []
    for w in range(n_windows - 1):
        for cid, (c, y, pid) in enumerate(zip(cands[w], labs[w].y, labs[w].pattern_id)):
            if y != 1 or not match_next(c.nodes, int(pid), pos[w + 1]):
                continue
            v_next = ring_volume(wins[w + 1], int(pid))
            rows.append({"window_id": w, "candidate_id": cid, "pattern_id": int(pid), "scenario": scen[int(pid)], "seed": cfg.seed, "group": group_offset + int(pid),
                         "vol_now": ring_volume(wins[w], int(pid)), "vol_next": v_next, "y_next": float(np.log1p(v_next))})  # fmt: skip
    df = pd.DataFrame(rows)
    return df.merge(feats, on=["window_id", "candidate_id"], how="left", validate="one_to_one") if len(df) else df


def make_model(name: str, cfg: dict, seed: int = 0) -> Pipeline:
    p = cfg["models"][name]
    if name == "ridge":
        m = Ridge(alpha=p["alpha"])
    elif name == "rf":
        m = RandomForestRegressor(n_estimators=p["n_estimators"], min_samples_leaf=p["min_samples_leaf"], random_state=seed, n_jobs=p["n_jobs"])
    elif name == "gbr":
        m = GradientBoostingRegressor(n_estimators=p["n_estimators"], max_depth=p["max_depth"], learning_rate=p["learning_rate"], subsample=p["subsample"], random_state=seed)
    else:
        raise ValueError(name)
    return Pipeline([("scale", StandardScaler()), ("m", m)])


def assert_no_excluded(columns, excluded) -> None:
    bad = sorted(set(columns) & set(excluded))
    if bad:
        raise AssertionError(f"excluded volume columns in the feature set: {bad}")


def grouped_folds(groups: np.ndarray, k: int, repeats: int, seed: int = 0) -> list[np.ndarray]:
    """Per repeat a fold id per row; every group sits in exactly one fold (groups shuffled, dealt round-robin)."""
    out = []
    ug = np.unique(groups)
    for r in range(repeats):
        perm = np.random.default_rng(seed + r).permutation(ug)
        fold_of = {g: i % k for i, g in enumerate(perm)}
        out.append(np.array([fold_of[g] for g in groups]))
    return out


def oof_predict(X, y, fold, name, cfg, seed=0) -> np.ndarray:
    out = np.full(len(y), np.nan)
    for f in np.unique(fold):
        te = fold == f
        if name == "mean":
            out[te] = y[~te].mean()
            continue
        m = make_model(name, cfg, seed)
        m.fit(X[~te], y[~te])
        out[te] = m.predict(X[te])
    return out


def reg_metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    rho = spearmanr(y, p).statistic if len(np.unique(p)) > 1 and len(np.unique(y)) > 1 else float("nan")
    return {"rmse": float(np.sqrt(np.mean((y - p) ** 2))), "mae": float(np.mean(np.abs(y - p))), "spearman": float(rho)}


def grouped_bootstrap_metrics(y, preds_by_repeat: dict[str, list[np.ndarray]], groups, n_boot=1000, seed=0) -> dict[str, dict]:
    """Resample groups (same resamples for every predictor); per predictor the metrics of every resample, mean over repeats."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_of = {g: np.flatnonzero(groups == g) for g in ug}
    res = {k: {"rmse": [], "mae": [], "spearman": []} for k in preds_by_repeat}
    for _ in range(n_boot):
        idx = np.concatenate([idx_of[g] for g in rng.choice(ug, size=len(ug), replace=True)])
        for k, reps in preds_by_repeat.items():
            ms = [reg_metrics(y[idx], p[idx]) for p in reps]
            for m in ("rmse", "mae", "spearman"):
                res[k][m].append(float(np.nanmean([x[m] for x in ms])))
    return {k: {m: np.array(v) for m, v in d.items()} for k, d in res.items()}


def ci(a, level=0.95):
    lo, hi = np.nanpercentile(a, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(lo), float(hi)
