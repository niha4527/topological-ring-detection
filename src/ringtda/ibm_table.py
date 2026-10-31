"""Candidate table for IBM AML windows: candidates, ring labels (§3.7) and first-seen times.

Reuses the existing window aggregation, weights and candidate enumeration (no new graph logic).
One row per (window, tau_quantile, candidate); no features yet (D, B, T come later).
"""

from __future__ import annotations

import logging

import pandas as pd

from ringtda.data.ibm_aml import IBMData, iter_ibm_windows, load_config
from ringtda.eval.labels import label_candidates_ibm, unattributed_edges
from ringtda.graph.weights import edge_weights, to_digraph
from ringtda.graph.windows import aggregate_pairs
from ringtda.topology.candidates import candidate_rings

log = logging.getLogger(__name__)


def ibm_candidate_tables(data: IBMData, cfg: dict | None = None, tau_quantiles=None, max_len: int | None = None) -> dict[float, pd.DataFrame]:
    """{tau_quantile: table}. Columns: dataset, tau_quantile, window_id, window_start, cand_id, nodes,
    edges, length, kind, attempt (attempt id for positives, else -1), y, t_seen (day the ring completes:
    the max over its edges of the edge's first transaction in the window)."""
    cfg = cfg or load_config()
    taus = list(tau_quantiles if tau_quantiles is not None else cfg["candidates"]["tau_quantiles"])
    max_len = max_len or cfg["candidates"]["max_len"]
    x = cfg["labels"]["label_x"]
    attempts = dict(zip(data.attempts["attempt_id"], data.attempts["pattern"]))
    rows = {q: [] for q in taus}
    for wid, start, _, w in iter_ibm_windows(data, cfg["window"]["length_days"], cfg["window"]["stride_days"]):
        pairs = edge_weights(aggregate_pairs(w))
        G = to_digraph(pairs)
        first = dict(zip(zip(pairs["src"], pairs["dst"]), pairs["t_first"]))
        flagged = unattributed_edges(w)
        for q in taus:
            res = candidate_rings(G, tau=float(pairs["weight"].quantile(q)), max_len=max_len)
            if res.truncated_sccs:
                log.warning("%s window %d q=%s: %d SCCs hit the enumeration cap", data.name, wid, q, len(res.truncated_sccs))
            lab = label_candidates_ibm(res.candidates, w, attempts, x, flagged)
            for cid, (c, kind, pid, y) in enumerate(zip(res.candidates, lab["kind"], lab["pattern_id"], lab["y"])):
                rows[q].append(
                    (data.name, q, wid, start, cid, c.nodes, c.edges, c.length, kind, int(pid) if y == 1 else -1, y, max(first[e] for e in c.edges))
                )
        log.info("%s window %d done", data.name, wid)
    cols = ["dataset", "tau_quantile", "window_id", "window_start", "cand_id", "nodes", "edges", "length", "kind", "attempt", "y", "t_seen"]
    return {q: pd.DataFrame(r, columns=cols) for q, r in rows.items()}
