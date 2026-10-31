"""Candidate-ring table for one window: labels, behavioural (D) and topological (T) features.

Candidates are generated once per window on a loose cutoff of the equal-weight filtration,
so that the candidate set does not depend on the alpha being learned.

Labels follow the §3.7 ring-labeling rule (ringtda.eval.labels): kind is the planted
pattern backing >= label_x of the ring's edges, "background" if none are backed, else
"ambiguous" with y = NaN; the caller decides whether to drop ambiguous rows or count
them as negatives.

Topological features come from ringtda.topology.vectorize. The legacy scalar columns
(`topo_*`, used by the optimization report) are derived from the same local diagram with the
cap fixed at 1.0, so the report's numbers are unchanged; `t_feature_table` builds the full
T feature table (landscapes + scalars, local and window level) with the configured cap rule.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ringtda.eval.labels import label_candidates
from ringtda.graph.baseline_features import B_NAMES, WindowBaseline, baseline_features
from ringtda.graph.baseline_features import load_config as load_features_config
from ringtda.graph.direction_features import D_NAMES, direction_features, gap_cv_fill
from ringtda.graph.weights import edge_weights, to_digraph
from ringtda.graph.windows import aggregate_pairs
from ringtda.topology.candidates import candidate_rings, sparsify
from ringtda.topology.vectorize import (
    cap_diagram,
    feature_names,
    load_config,
    local_diagrams,
    local_graph,
    local_topology_features,
    window_topology_features,
)

ZCOLS = ["n_s_rec", "n_s_bal", "n_s_reg"]
# Legacy `topo_*` columns: report-only, fixed cap 1.0, superseded by the window-max cap in
# vectorize.py. Frozen so that `gd_report.py --config report` keeps reproducing the report: they
# ignore configs/topology.yaml (hops, backend, local_graph, cap rule). New code must not use them;
# use t_feature_table.
LEGACY_CAP = 1.0
LEGACY_LOCAL = {"local": {"hops": 1, "backend": "auto"}}


def window_graphs(tx: pd.DataFrame, tau_quantile: float = 0.25):
    """(pairs, full window graph G, sparsified graph Gs, tau_s) for one window."""
    pairs = edge_weights(aggregate_pairs(tx))
    G = to_digraph(pairs)
    tau = float(pairs["weight"].quantile(tau_quantile))
    return pairs, G, sparsify(G, tau), tau


def window_candidates(
    tx: pd.DataFrame, planted, tau_quantile: float = 0.25, max_len: int = 5, topo: bool = True, label_x: float = 0.8
):
    """Return (table, Z): one row per candidate ring, and the list of its (|R|, 3) score arrays.

    The `topo_*` columns (topo=True) are LEGACY: report-only, fixed cap 1.0, superseded by the
    window-max cap of vectorize.py / t_feature_table. They are computed on the sparsified graph
    with 1 hop regardless of configs/topology.yaml, so the optimization report stays reproducible.
    """
    pairs, G, Gs, tau = window_graphs(tx, tau_quantile)
    res = candidate_rings(G, tau=tau, max_len=max_len)

    labels = label_candidates(res.candidates, tx, planted, x=label_x)
    rows, Z = [], []
    for c, kind, y in zip(res.candidates, labels["kind"], labels["y"]):
        z = np.array([[G.edges[e][k] for k in ZCOLS] for e in c.edges])
        row = {
            "kind": kind,
            "y": y,
            "length": c.length,
            "birth_eq": c.birth,  # ring birth under equal weights (1/3, 1/3, 1/3)
            **{f"mean_{k[4:]}": z[:, i].mean() for i, k in enumerate(ZCOLS)},
            **{f"min_{k[4:]}": z[:, i].min() for i, k in enumerate(ZCOLS)},
        }
        if topo:
            H, dgms = local_diagrams(Gs, c, LEGACY_LOCAL)  # legacy: always sparsified, 1 hop
            capped, n_inf = cap_diagram(dgms[1], LEGACY_CAP)
            pers = capped[:, 1] - capped[:, 0] if len(capped) else np.zeros(0)
            row.update(
                topo_h1_count=len(dgms[1]),
                topo_total_pers=float(pers.sum()),
                topo_max_pers=float(pers.max()) if len(pers) else 0.0,
                topo_n_infinite=n_inf,
                topo_nbhd_edges=H.number_of_edges(),
            )
        rows.append(row)
        Z.append(z)
    return pd.DataFrame(rows), Z


KEYS = ["window_id", "candidate_id"]


def _feature_table(windows, parts, tau_quantile, max_len, cfg, backend, return_candidates, betweenness_cache=None, timings=None):
    """Shared loop of the B / D / T tables: same windows, graphs, candidates and (window_id,
    candidate_id) keys for every part. parts: subset of {"B", "D", "T"}; columns are
    keys + B + D + T_local + T_win in that order.

    betweenness_cache: optional file path for the (single) window's cached betweenness (B only).
    timings: optional dict filled with accumulated seconds per part ("graph_candidates", "B", "D", "T").
    """
    if tau_quantile is None:
        tau_quantile = cfg["candidates"]["tau_quantile"]
    names = (
        (B_NAMES if "B" in parts else [])
        + (D_NAMES if "D" in parts else [])
        + (feature_names(cfg, "T_local") + feature_names(cfg, "T_win") if "T" in parts else [])
    )
    tm = timings if timings is not None else {}
    for k in ("graph_candidates", "B", "D", "T"):
        tm.setdefault(k, 0.0)
    rows, cands = [], {}
    for wid, tx in enumerate(windows):
        t0 = time.perf_counter()
        _, G, Gs, tau = window_graphs(tx, tau_quantile)
        res = candidate_rings(G, tau=tau, max_len=max_len)
        cands[wid] = res.candidates
        tm["graph_candidates"] += time.perf_counter() - t0
        if not res.candidates:
            continue
        if "T" in parts:
            t0 = time.perf_counter()
            wmax = float(max(d["weight"] for _, _, d in G.edges(data=True)))
            win = window_topology_features(Gs, wmax, cfg, backend)  # once per window
            LG = local_graph(G, Gs, cfg)
            tm["T"] += time.perf_counter() - t0
        if "B" in parts:
            t0 = time.perf_counter()
            ctx = WindowBaseline(G, load_features_config(), betweenness_cache)  # betweenness once per window
            tm["B"] += time.perf_counter() - t0
        if "D" in parts:
            fill = gap_cv_fill(G)
        for cid, c in enumerate(res.candidates):
            row = {"window_id": wid, "candidate_id": cid}
            if "B" in parts:
                t0 = time.perf_counter()
                row.update(baseline_features(G, c, ctx, cfg["local"]["hops"]))
                tm["B"] += time.perf_counter() - t0
            if "D" in parts:
                t0 = time.perf_counter()
                row.update(direction_features(G, c, fill))
                tm["D"] += time.perf_counter() - t0
            if "T" in parts:
                t0 = time.perf_counter()
                row.update(local_topology_features(LG, c, wmax, cfg, backend))
                row.update(win)
                tm["T"] += time.perf_counter() - t0
            rows.append(row)
    df = pd.DataFrame(rows, columns=KEYS + names)
    return (df, cands) if return_candidates else df


def t_feature_table(
    windows, tau_quantile: float | None = None, max_len: int = 6, config: dict | None = None, backend: str | None = None,
    return_candidates: bool = False,
):
    """T feature table: one row per candidate, keyed by (window_id, candidate_id).

    windows: iterable of transaction tables (window_id = position). Columns: the two keys,
    the local T features (`T_local_*`) and the window T features (`T_win_*`, computed once per
    window on the sparsified graph and repeated on each of its rows). T only: no ring birth,
    degree, centrality or behavioural columns (those belong to D / B, §3.4). Fixed width, no NaNs.

    tau_quantile (None = configs/topology.yaml `candidates.tau_quantile`) sets the sparsification
    cutoff tau_s; max_len=6 follows §3.3 (window_candidates keeps 5 for the optimization report).
    """
    return _feature_table(windows, {"T"}, tau_quantile, max_len, config or load_config(), backend, return_candidates)


def bd_feature_table(
    windows, tau_quantile: float | None = None, max_len: int = 6, config: dict | None = None, return_candidates: bool = False,
    betweenness_cache=None, timings=None,
):
    """B + D table with the same keys, windows and candidates as `t_feature_table` (B: graph baseline
    on the full window graph; D: direction/behaviour incl. ring birth). Label-free by signature."""
    return _feature_table(windows, {"B", "D"}, tau_quantile, max_len, config or load_config(), None, return_candidates, betweenness_cache, timings)


def full_feature_table(
    windows, tau_quantile: float | None = None, max_len: int = 6, config: dict | None = None, backend: str | None = None,
    return_candidates: bool = False, betweenness_cache=None, timings=None,
):
    """B + D + T joined on (window_id, candidate_id), computing windows and candidates once."""
    return _feature_table(windows, {"B", "D", "T"}, tau_quantile, max_len, config or load_config(), backend, return_candidates, betweenness_cache, timings)


def feature_sets(config: dict | None = None) -> dict[str, list[str]]:
    """Explicit column lists of the classification feature sets, built from the prefixes B_ / D_ / T_."""
    cfg = config or load_config()
    T = feature_names(cfg, "T_local") + feature_names(cfg, "T_win")
    return {"B": list(B_NAMES), "B+D": list(B_NAMES) + list(D_NAMES), "B+D+T": list(B_NAMES) + list(D_NAMES) + T, "T": T}
