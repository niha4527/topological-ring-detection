"""Simulator: score user-entered transactions with a classifier trained on the synthetic generator.

The user's transactions are merged into a background-only synthetic week (same generator and settings
as the training data, no planted structures), so the within-window rank normalisation (§3.1), the
sparsification cutoff (§3.3) and the feature scales mean what they meant in training. Every
intermediate quantity is returned, so a front end can show the whole calculation.

The model is logistic regression (fixed hyperparameters of configs/classify.yaml) on B+D or B+D+T_local
(the window-level T_win columns are left out: constant within a window, they would only identify the window),
trained on the cached synthetic tables (hard negatives on, ambiguous candidates excluded). Its score is
the decision function, not a calibrated probability (class_weight is balanced). The flag threshold is
the F1-optimal score on grouped out-of-fold training scores. This is a demonstration of the pipeline on
synthetic data: it says nothing about real-world or IBM AML performance.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ringtda.data.synthetic import SyntheticConfig, generate
from ringtda.eval.splits import stratified_group_cv
from ringtda.features import window_graphs
from ringtda.graph.baseline_features import WindowBaseline, baseline_features
from ringtda.graph.baseline_features import load_config as load_features_config
from ringtda.graph.direction_features import direction_features, gap_cv_fill
from ringtda.models.classify import load_config as load_classify_config
from ringtda.models.classify import make_model, named_feature_sets, oof_scores
from ringtda.synthetic_table import synthetic_candidate_table
from ringtda.topology.candidates import candidate_rings
from ringtda.topology.vectorize import (
    load_config as load_topology_config,
)
from ringtda.topology.vectorize import (
    local_diagrams,
    local_graph,
    local_topology_features,
)

DAY = 86400.0
WEEK_H = 7 * 24.0
MAX_ROWS = 500
USER_ID_BASE = 100_000  # user accounts get ids above any background account
TRAIN_CONDITION = "hn-on_chords-default"
TRAIN_SEEDS = range(6)
FEATURE_SETS = ("B+D", "B+D+T_local")  # T_win is constant within a window: with 6 training windows it is a window id


# ---------------------------------------------------------------- input


def parse_transactions(text: str) -> pd.DataFrame:
    """Lines `hour from to amount` (whitespace or comma separated; `#` starts a comment).

    hour in [0, 168) is the time in hours since the start of the simulated week.
    """
    rows = []
    for n, line in enumerate(text.splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        p = [x for x in re.split(r"[\s,]+", line) if x]
        if len(p) != 4:
            raise ValueError(f"line {n}: expected `hour from to amount`, got {len(p)} fields")
        try:
            hour, amount = float(p[0]), float(p[3])
        except ValueError:
            raise ValueError(f"line {n}: hour and amount must be numbers") from None
        if not 0 <= hour < WEEK_H:
            raise ValueError(f"line {n}: hour must be in [0, {WEEK_H:g})")
        if not amount > 0:
            raise ValueError(f"line {n}: amount must be positive")
        if p[1] == p[2]:
            raise ValueError(f"line {n}: self-transfer {p[1]} -> {p[2]} (self-loops are dropped by the pipeline)")
        rows.append((hour, p[1], p[2], amount))
        if len(rows) > MAX_ROWS:
            raise ValueError(f"at most {MAX_ROWS} transactions")
    if not rows:
        raise ValueError("no transactions")
    return pd.DataFrame(rows, columns=["hour", "from", "to", "amount"])


def background_config(seed: int = 0) -> SyntheticConfig:
    """Generator settings with nothing planted: background traffic only."""
    return SyntheticConfig(
        ring_sizes=(), dense_ring_sizes=(), n_transitive_decoys=0, n_chorded_rings=0,
        n_shortcut_filled=0, hard_negatives=False, seed=seed,
    )  # fmt: skip


def merge_window(user: pd.DataFrame, seed: int = 0) -> tuple[pd.DataFrame, dict[str, int]]:
    """Background week + user transactions. Returns (tx, label -> account id)."""
    bg, _ = generate(background_config(seed))
    names = sorted(set(user["from"]) | set(user["to"]))
    ids = {n: USER_ID_BASE + i for i, n in enumerate(names)}
    u = pd.DataFrame(
        {
            "timestamp": user["hour"].to_numpy() * 3600.0,
            "src": user["from"].map(ids).to_numpy(),
            "dst": user["to"].map(ids).to_numpy(),
            "amount": user["amount"].to_numpy(),
            "pattern_id": -2,
            "pattern": "user",
        }
    )
    tx = pd.concat([bg, u], ignore_index=True).sort_values("timestamp", kind="stable").reset_index(drop=True)
    return tx, ids


# ---------------------------------------------------------------- model


@dataclass
class SimModel:
    feature_set: str
    columns: list[str]
    pipe: object  # sklearn Pipeline (StandardScaler + LogisticRegression)
    threshold: float
    train_scores: np.ndarray  # out-of-fold scores of the training candidates
    train_y: np.ndarray
    info: dict = field(default_factory=dict)

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.pipe.decision_function(X)

    def contributions(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """(intercept, per-feature contribution to the logit) for one standardised row."""
        sc, clf = self.pipe.named_steps["scale"], self.pipe.named_steps["clf"]
        z = (x - sc.mean_) / sc.scale_
        return float(clf.intercept_[0]), clf.coef_[0] * z


def training_table(cache_dir: str | Path | None = None) -> pd.DataFrame:
    """Cached per-seed tables of experiments/synth_classify.py if present, else built (slow)."""
    parts = []
    for s in TRAIN_SEEDS:
        f = Path(cache_dir) / f"{TRAIN_CONDITION}_seed{s}.pkl" if cache_dir else None
        if f is not None and f.exists():
            parts.append(pd.read_pickle(f))
        else:
            parts.append(synthetic_candidate_table(SyntheticConfig(seed=s), group_offset=s * 1_000_000))
    return pd.concat(parts, ignore_index=True)


def _best_f1_threshold(score: np.ndarray, y: np.ndarray) -> float:
    o = np.argsort(-score, kind="stable")
    s, yy = score[o], y[o]
    tp = np.cumsum(yy)
    k = np.arange(1, len(yy) + 1)
    f1 = 2 * tp / (k + yy.sum())
    ends = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]  # only cut between distinct scores
    best = ends[np.argmax(f1[ends])]
    return float((s[best] + s[best + 1]) / 2) if best + 1 < len(s) else float(s[best])


def train(df: pd.DataFrame, feature_set: str) -> SimModel:
    cols = named_feature_sets()[feature_set]
    cfg = load_classify_config()
    d = df[df.y.notna()].reset_index(drop=True)
    X, y, groups = d[cols].to_numpy(float), d.y.to_numpy(int), d.group.to_numpy()
    run = stratified_group_cv(groups, y.astype(float), cfg["cv"]["k"], cfg["cv"]["fallback_k"], 1, cfg["cv"]["seed"])[0]
    oof = oof_scores(X, y, groups, run["fold"], "logreg", cfg, cfg["model_seed"])
    pipe = make_model("logreg", cfg, cfg["model_seed"]).fit(X, y)
    info = {
        "n_train": int(len(y)), "n_pos": int(y.sum()), "n_pos_groups": int(len(set(groups[y == 1]))),
        "n_features": len(cols), "cv_k": int(run["k"]),
    }
    return SimModel(feature_set, cols, pipe, _best_f1_threshold(oof, y), oof, y, info)


def load_models(cache_dir: str | Path | None = None, pkl: str | Path | None = None) -> dict[str, SimModel]:
    """Both models; read from `pkl` if it exists, else trained (and written to `pkl`)."""
    if pkl is not None and Path(pkl).exists():
        with open(pkl, "rb") as f:
            return pickle.load(f)
    df = training_table(cache_dir)
    models = {fs: train(df, fs) for fs in FEATURE_SETS}
    if pkl is not None:
        Path(pkl).parent.mkdir(parents=True, exist_ok=True)
        with open(pkl, "wb") as f:
            pickle.dump(models, f)
    return models


# ---------------------------------------------------------------- simulate


def _bars(dgm) -> list[list[float | None]]:
    a = np.asarray(dgm, dtype=float).reshape(-1, 2)
    return [[float(b), None if not np.isfinite(e) else float(e)] for b, e in a]


def _r(x, n=4):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def simulate(text: str, models: dict[str, SimModel], feature_set: str = "B+D", seed: int = 0) -> dict:
    """Run the full pipeline on the user's transactions and return every step."""
    user = parse_transactions(text)
    model = models[feature_set]
    tcfg = load_topology_config()
    tau_q, max_len = tcfg["candidates"]["tau_quantile"], tcfg["candidates"]["max_len"]

    tx, ids = merge_window(user, seed)
    inv = {v: k for k, v in ids.items()}
    name = lambda a: inv.get(a, f"bg{a}")  # noqa: E731
    is_user = lambda a: a in inv  # noqa: E731

    # 1-2: window, aggregation, scores, weights
    pairs, G, Gs, tau = window_graphs(tx, tau_q)
    n_bg = int((tx.pattern == "background").sum())
    steps: dict = {
        "input": user.round(4).to_dict("records"),
        "window": {
            "user_tx": len(user), "background_tx": n_bg, "accounts": G.number_of_nodes(), "pairs": len(pairs),
            "user_accounts": sorted(ids), "seed": seed,
        },
    }
    up = pairs[pairs.src.map(is_user) & pairs.dst.map(is_user)].sort_values(["src", "dst"])
    steps["pairs"] = [
        {
            "from": name(r.src), "to": name(r.dst), "volume": _r(r.volume, 2), "count": int(r.count),
            "gap_cv": _r(r.gap_cv), "rev_volume": _r(r.rev_volume, 2),
            "s_rec": _r(r.s_rec), "s_bal": _r(r.s_bal), "s_reg": _r(r.s_reg),
            "n_s_rec": _r(r.n_s_rec), "n_s_bal": _r(r.n_s_bal), "n_s_reg": _r(r.n_s_reg),
            "weight": _r(r.weight), "kept": bool(r.weight <= tau),
        }
        for r in up.itertuples()
    ]
    steps["scores_note"] = {
        "n_pairs_in_window": len(pairs),
        "median_s_reg": _r(float(pairs["s_reg"].median())),
        "gap_cv_fill": _r(gap_cv_fill(G)),
    }
    steps["sparsify"] = {
        "quantile": tau_q, "tau": _r(tau), "edges_total": G.number_of_edges(), "edges_kept": Gs.number_of_edges(),
        "user_edges": len(up), "user_edges_kept": int((up.weight <= tau).sum()),
    }

    # 3: candidate rings; keep those that touch a user account
    res = candidate_rings(G, tau=tau, max_len=max_len)
    mine = [c for c in res.candidates if any(is_user(n) for n in c.nodes)]
    steps["candidates"] = {
        "sccs": res.n_sccs, "all_candidates": len(res.candidates), "user_candidates": len(mine), "max_len": max_len,
        "truncated_sccs": res.truncated_sccs,
    }
    out = {"model": _model_info(model), "steps": steps, "rings": []}
    out["viz"] = _viz(user, pairs, G, tau, ids, max_len, seed)
    if not mine:
        out["verdict"] = {
            "flagged": False,
            "reason": "no candidate ring: no directed cycle of length 2-%d survives the sparsification cutoff among these accounts" % max_len,
        }
        return out

    # 4: features + score for each user ring (window context computed once)
    wmax = float(max(d["weight"] for _, _, d in G.edges(data=True)))
    ffcfg = load_features_config()
    use_t = any(c.startswith("T_local") for c in model.columns)
    LG = local_graph(G, Gs, tcfg)
    ctx = WindowBaseline(G, ffcfg)
    fill = gap_cv_fill(G)

    rows, rings = [], []
    for c in mine:
        f = {**baseline_features(G, c, ctx, tcfg["local"]["hops"]), **direction_features(G, c, fill)}
        if use_t:
            f.update(local_topology_features(LG, c, wmax, tcfg))
        rows.append([f[k] for k in model.columns])
        H, dg = local_diagrams(LG, c, tcfg)
        rings.append((c, f, H, dg))
    X = np.array(rows, dtype=float)
    sc = model.score(X)

    for (c, f, H, dg), x, s in zip(rings, X, sc):
        icpt, contrib = model.contributions(x)
        grp = {}
        for col, v in zip(model.columns, contrib):
            g = "T_local" if col.startswith("T_local") else col[0]
            grp[g] = grp.get(g, 0.0) + float(v)
        order = np.argsort(-np.abs(contrib))[:10]
        sca = model.pipe.named_steps["scale"]
        ring_edges = [
            {"from": name(u), "to": name(v), "weight": _r(G.edges[u, v]["weight"]), "kept": G.edges[u, v]["weight"] <= tau,
             "n_s_rec": _r(G.edges[u, v]["n_s_rec"]), "n_s_bal": _r(G.edges[u, v]["n_s_bal"]), "n_s_reg": _r(G.edges[u, v]["n_s_reg"])}
            for u, v in c.edges
        ]  # fmt: skip
        out["rings"].append(
            {
                "nodes": [name(n) for n in c.nodes], "length": c.length, "birth": _r(c.birth),
                "is_pure_user": all(is_user(n) for n in c.nodes),
                "edges": ring_edges,
                "neighbourhood": {"accounts": H.number_of_nodes(), "edges": H.number_of_edges()},
                "h0": _bars(dg[0]), "h1": _bars(dg[1]),
                "open_loop": _r(f.get("T_local_ring_inf")) if use_t else None,
                "features": {k: _r(v, 5) for k, v in f.items() if k[0] in "BD" and k[1] == "_"},
                "contrib_groups": {k: _r(v) for k, v in grp.items()},
                "contrib_top": [
                    {"feature": model.columns[i], "value": _r(x[i], 5), "mean": _r(sca.mean_[i], 5), "scale": _r(sca.scale_[i], 5), "contribution": _r(contrib[i])}
                    for i in order
                ],
                "intercept": _r(icpt), "score": _r(s), "threshold": _r(model.threshold),
                "flagged": bool(s >= model.threshold),
                "percentile": _r(float((model.train_scores < s).mean() * 100), 1),
            }
        )
    out["rings"].sort(key=lambda r: -r["score"])
    by_nodes = {tuple(r["nodes"]): r for r in out["rings"]}
    for vr in out["viz"]["rings"]:  # attach the model's score to the rings it scored (same account cycle)
        m = by_nodes.get(tuple(vr["nodes"]))
        if m is not None:
            vr["score"], vr["flagged"] = m["score"], m["flagged"]
    top = out["rings"][0]
    out["verdict"] = {"flagged": any(r["flagged"] for r in out["rings"]), "top_ring": top["nodes"], "top_score": top["score"], "threshold": top["threshold"]}
    return out


def _model_info(m: SimModel) -> dict:
    tr = m.train_y == 1
    return {
        "feature_set": m.feature_set, "threshold": _r(m.threshold), **m.info,
        "oof_score_median_pos": _r(float(np.median(m.train_scores[tr]))),
        "oof_score_median_neg": _r(float(np.median(m.train_scores[~tr]))),
        "flag_precision_oof": _r(float((m.train_y[m.train_scores >= m.threshold]).mean())),
        "flag_recall_oof": _r(float((m.train_scores[tr] >= m.threshold).mean())),
    }


def _viz(user: pd.DataFrame, pairs: pd.DataFrame, G, tau: float, ids: dict[str, int], max_len: int, seed: int) -> dict:
    """What the page draws for the user's own accounts: the graph, every edge's scores and weight, and every ring among the user's
    accounts. A ring exists at threshold t exactly when its birth (largest edge weight) is <= t, so the page filters these rings by birth.
    Weights are ranks inside the WHOLE window (user plus background), so the window range and the pipeline cutoff `tau` are included."""
    import networkx as nx

    inv = {v: k for k, v in ids.items()}
    mine = set(ids.values())
    up = pairs[pairs.src.isin(mine) & pairs.dst.isin(mine)]
    Gu = G.subgraph(mine).copy()
    names = lambda a: inv[a]  # noqa: E731
    times = {(a, b): (g["hour"].to_numpy()).round(2).tolist() for (a, b), g in user.assign(a=user["from"].map(ids), b=user["to"].map(ids)).groupby(["a", "b"])}
    und = Gu.to_undirected()
    pos = nx.spring_layout(und, seed=seed, k=1.6 / max(1, und.number_of_nodes()) ** 0.5, iterations=200) if und.number_of_nodes() > 2 else {n: (i * 2.0 - 1.0, 0.0) for i, n in enumerate(sorted(und.nodes))}
    res = candidate_rings(Gu, tau=1.0, max_len=max_len, max_cycles_per_scc=2000)
    rings = [
        {"nodes": [names(n) for n in c.nodes], "edges": [[names(u), names(v)] for u, v in c.edges], "birth": _r(c.birth), "in_pipeline": bool(c.birth <= tau)}
        for c in res.candidates
    ]
    return {
        "window": {"wmin": _r(float(pairs["weight"].min())), "wmax": _r(float(pairs["weight"].max())), "n_pairs": int(len(pairs)), "tau": _r(tau), "tau_quantile": load_topology_config()["candidates"]["tau_quantile"]},
        "max_hour": float(user["hour"].max()),
        "nodes": [{"id": names(n), "x": round(float(pos[n][0]), 4), "y": round(float(pos[n][1]), 4)} for n in Gu.nodes],
        "edges": [
            {"src": names(r.src), "dst": names(r.dst), "times": times[(r.src, r.dst)], "volume": _r(r.volume, 2), "count": int(r.count), "rev_volume": _r(r.rev_volume, 2),
             "gap_cv": _r(r.gap_cv), "s_rec": _r(r.s_rec), "s_bal": _r(r.s_bal), "s_reg": _r(r.s_reg), "n_s_rec": _r(r.n_s_rec), "n_s_bal": _r(r.n_s_bal),
             "n_s_reg": _r(r.n_s_reg), "weight": _r(r.weight)}
            for r in up.itertuples()
        ],
        "rings": sorted(rings, key=lambda r: r["birth"]), "rings_truncated": bool(res.truncated_sccs),
    }  # fmt: skip
