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

import numpy as np
import pandas as pd

from ringtda.eval.labels import label_candidates
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
    cfg = config or load_config()
    if tau_quantile is None:
        tau_quantile = cfg["candidates"]["tau_quantile"]
    names = feature_names(cfg, "T_local") + feature_names(cfg, "T_win")
    rows, cands = [], {}
    for wid, tx in enumerate(windows):
        _, G, Gs, tau = window_graphs(tx, tau_quantile)
        res = candidate_rings(G, tau=tau, max_len=max_len)
        cands[wid] = res.candidates
        if not res.candidates:
            continue
        wmax = float(max(d["weight"] for _, _, d in G.edges(data=True)))
        win = window_topology_features(Gs, wmax, cfg, backend)  # once per window
        LG = local_graph(G, Gs, cfg)
        for cid, c in enumerate(res.candidates):
            rows.append({"window_id": wid, "candidate_id": cid, **local_topology_features(LG, c, wmax, cfg, backend), **win})
    df = pd.DataFrame(rows, columns=["window_id", "candidate_id", *names])
    return (df, cands) if return_candidates else df
