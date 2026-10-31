"""Engineering gates on synthetic data (proposal v3 §3.7, Phases 1-2).

Gates run against the "easy" background (hard_negatives=False). With hard
negatives on, how many rings stay in the suspicious decile / become candidates
is candidate-generation recall, a reported result (§3.8), not a gate.
"""

import numpy as np

from ringtda.data.synthetic import SyntheticConfig, generate
from ringtda.graph.weights import edge_weights, to_digraph
from ringtda.graph.windows import aggregate_pairs
from ringtda.topology.candidates import candidate_rings


def _one_window(seed=0, **kw):
    tx, planted = generate(SyntheticConfig(seed=seed, **kw))
    pairs = edge_weights(aggregate_pairs(tx))
    return tx, planted, pairs, to_digraph(pairs)


def test_planted_ring_edges_in_lowest_weight_decile():
    """Phase 1 gate, for every positive scenario (isolated, dense, chorded)."""
    for seed in range(3):
        _, planted, pairs, G = _one_window(seed, hard_negatives=False)
        q10 = pairs["weight"].quantile(0.10)
        for p in planted:
            if not p.is_positive:
                continue
            ws = [G.edges[e]["weight"] for e in p.edges]
            assert max(ws) <= q10, f"seed {seed}: {p.scenario} ring {p.accounts} weights {np.round(ws, 3)} > q10 {q10:.3f}"


def test_every_planted_ring_is_a_candidate():
    """Phase 2 gate (candidate recall = 1 at tau = lowest decile), for every
    positive scenario, including dense rings whose neighbours offer near-duplicate
    detours (de-duplication keeps the shorter cycle, see candidates.py)."""
    for seed in range(3):
        _, planted, pairs, G = _one_window(seed, hard_negatives=False)
        found = {frozenset(c.edges) for c in candidate_rings(G, tau=pairs["weight"].quantile(0.10)).candidates}
        for p in planted:
            if p.is_positive or p.pattern == "shortcut_filled":  # shortcut-filled: ring-like under enumeration
                assert frozenset(p.edges) in found, f"seed {seed}: missed {p.scenario} ring {p.accounts}"


def test_transitive_decoys_are_not_candidates():
    _, planted, pairs, G = _one_window(0)
    res = candidate_rings(G, tau=pairs["weight"].quantile(0.10))
    for p in planted:
        if p.pattern == "transitive_decoy":
            # one-way flow must never come out as a ring on the decoy's own accounts
            assert not any(set(c.nodes) == set(p.accounts) for c in res.candidates)
