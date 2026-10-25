"""Candidate-ring table for one window: labels, behavioural (D) and topological (T) features.

Candidates are generated once per window on a loose cutoff of the equal-weight filtration,
so that the candidate set does not depend on the alpha being learned.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from ringtda.graph.weights import edge_weights, to_digraph
from ringtda.graph.windows import aggregate_pairs
from ringtda.topology.candidates import candidate_rings, neighborhood
from ringtda.topology.persistence import directed_flag_persistence

ZCOLS = ["n_s_rec", "n_s_bal", "n_s_reg"]


def window_candidates(tx: pd.DataFrame, planted, tau_quantile: float = 0.25, max_len: int = 5, topo: bool = True):
    """Return (table, Z): one row per candidate ring, and the list of its (|R|, 3) score arrays."""
    pairs = edge_weights(aggregate_pairs(tx))
    G = to_digraph(pairs)
    tau = float(pairs["weight"].quantile(tau_quantile))
    res = candidate_rings(G, tau=tau, max_len=max_len)
    Gs = nx.DiGraph()
    Gs.add_edges_from((u, v, d) for u, v, d in G.edges(data=True) if d["weight"] <= tau)

    kinds = {frozenset(p.edges): p.pattern for p in planted if p.pattern in ("wash_ring", "arbitrage_loop")}
    mm_edges = [set(p.edges) for p in planted if p.pattern == "market_maker"]
    rows, Z = [], []
    for c in res.candidates:
        z = np.array([[G.edges[e][k] for k in ZCOLS] for e in c.edges])
        kind = kinds.get(frozenset(c.edges), "background")
        if kind == "background" and any(set(c.edges) <= m for m in mm_edges):
            kind = "market_maker"
        row = {
            "kind": kind,
            "y": int(kind == "wash_ring"),
            "length": c.length,
            "birth_eq": c.birth,  # ring birth under equal weights (1/3, 1/3, 1/3)
            **{f"mean_{k[4:]}": z[:, i].mean() for i, k in enumerate(ZCOLS)},
            **{f"min_{k[4:]}": z[:, i].min() for i, k in enumerate(ZCOLS)},
        }
        if topo:
            H = neighborhood(Gs, c.nodes, hops=1)
            d1 = directed_flag_persistence(H)[1]
            fin = np.where(np.isfinite(d1[:, 1]), d1[:, 1], 1.0) if len(d1) else np.zeros(0)  # cap inf at 1
            pers = fin - d1[:, 0] if len(d1) else np.zeros(0)
            row.update(
                topo_h1_count=len(d1),
                topo_total_pers=float(pers.sum()),
                topo_max_pers=float(pers.max()) if len(pers) else 0.0,
                topo_n_infinite=int(np.isinf(d1[:, 1]).sum()) if len(d1) else 0,
                topo_nbhd_edges=H.number_of_edges(),
            )
        rows.append(row)
        Z.append(z)
    return pd.DataFrame(rows), Z
