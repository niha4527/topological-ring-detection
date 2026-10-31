"""The generator plants what proposal v3 §3.7 says it plants.

Topology checks use only each structure's own planted edges (with their real
window weights), so incidental background edges between planted accounts
cannot change the answer.
"""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from ringtda.data.synthetic import HARD_NEGATIVES, SyntheticConfig, generate
from ringtda.graph.weights import edge_weights, to_digraph
from ringtda.graph.windows import _gap_cv, aggregate_pairs
from ringtda.topology.persistence import directed_flag_persistence

SEEDS = range(3)


@pytest.fixture(scope="module", params=SEEDS)
def window(request):
    tx, planted = generate(SyntheticConfig(seed=request.param))
    G = to_digraph(edge_weights(aggregate_pairs(tx)))
    return tx, planted, G


def _planted_graph(G, p) -> nx.DiGraph:
    H = nx.DiGraph()
    for e in [*p.edges, *p.context_edges]:
        H.add_edge(*e, weight=G.edges[e]["weight"])
    return H


def _h1(G, p):
    return directed_flag_persistence(_planted_graph(G, p))[1]


def _own_rows(tx, p):
    return tx[(tx.pattern_id == p.pattern_id) & (tx.pattern == p.pattern)]


def _of(planted, pattern, scenario=None):
    out = [p for p in planted if p.pattern == pattern and (scenario is None or p.scenario == scenario)]
    assert out, f"no {pattern}/{scenario} planted"
    return out


def test_all_scenarios_present(window):
    _, planted, _ = window
    kinds = {(p.pattern, p.scenario) for p in planted}
    for scen in ("isolated", "dense", "chorded"):
        assert ("wash_ring", scen) in kinds
    for pat in ("transitive_decoy", "shortcut_filled", *HARD_NEGATIVES):
        assert any(k[0] == pat for k in kinds)


def test_planted_accounts_disjoint(window):
    _, planted, _ = window
    seen = set()
    for p in planted:
        if p.pattern in HARD_NEGATIVES:
            continue  # hard-negative counterparties are ordinary accounts and may be shared
        mine = set(p.accounts) | set(p.context_accounts)
        assert not mine & seen
        seen |= mine


def test_without_hard_negatives():
    tx, planted = generate(SyntheticConfig(seed=0, hard_negatives=False))
    assert not any(p.pattern in HARD_NEGATIVES for p in planted)
    assert not tx.pattern.isin(HARD_NEGATIVES).any()


def test_hard_negatives_do_not_change_other_structures():
    """Hard negatives are drawn last, so the with/without runs share everything else."""
    a, pa = generate(SyntheticConfig(seed=1))
    b, pb = generate(SyntheticConfig(seed=1, hard_negatives=False))
    keep = {p.pattern_id for p in pb}
    assert [p for p in pa if p.pattern_id in keep] == pb
    a = a[a.pattern_id.isin(keep) | (a.pattern == "background")].reset_index(drop=True)
    assert a.equals(b)


# --- positives -------------------------------------------------------------


def test_isolated_ring_is_one_lasting_loop(window):
    _, planted, G = window
    for p in _of(planted, "wash_ring", "isolated"):
        d = _h1(G, p)
        assert len(d) == 1 and np.isinf(d[0, 1])


def test_ring_members_net_to_zero(window):
    """Wash rings, plain or chorded, close every round: all members balanced."""
    tx, planted, _ = window
    for p in _of(planted, "wash_ring"):
        t = _own_rows(tx, p)
        for a in p.accounts:
            out_v, in_v = t.loc[t.src == a, "amount"].sum(), t.loc[t.dst == a, "amount"].sum()
            # < 0.05, not 0: a round cut by the window edge leaves one hop unmatched
            assert abs(out_v - in_v) / (out_v + in_v) < 0.05


def test_chorded_ring_splits_into_shorter_loops(window):
    tx, planted, G = window
    for p in _of(planted, "wash_ring", "chorded"):
        assert p.context_edges, "chords missing"
        d = _h1(G, p)
        # 6-ring + chord (0,3): two independent loops, no transitive triangle to fill either
        assert len(d) == 2 and np.isinf(d[:, 1]).all()
        # chord traffic really happens and is part of the scheme
        t = _own_rows(tx, p)
        for u, v in p.context_edges:
            assert ((t.src == u) & (t.dst == v)).sum() > 3


def test_dense_ring_sits_among_many_loops(window):
    _, planted, G = window
    for p in _of(planted, "wash_ring", "dense"):
        assert len(p.context_accounts) == SyntheticConfig().dense_neighbours
        H = _planted_graph(G, p)
        assert len(directed_flag_persistence(H)[1]) > 3
        # the ring is one of many directed cycles in its neighbourhood
        assert sum(1 for _ in nx.simple_cycles(H, length_bound=4)) > 5


def test_chord_validation():
    with pytest.raises(ValueError):
        generate(SyntheticConfig(chords=((0, 1),)))  # adjacent: not a chord


# --- topology-sensitive negatives ------------------------------------------


def test_shortcut_filled_ring_dies(window):
    """Phase 2 gate on generated data: the cyclic triangle is filled by the
    transitive triangles through the fourth account, so no H1 class lasts."""
    _, planted, G = window
    for p in _of(planted, "shortcut_filled"):
        assert len(p.accounts) == 4 and len(p.context_edges) == 3
        d = _h1(G, p)
        assert not np.isinf(d[:, 1]).any()
        # without the filler edges the same triangle is a lasting loop
        H = nx.DiGraph()
        H.add_edges_from((u, v, {"weight": G.edges[u, v]["weight"]}) for u, v in p.edges)
        assert np.isinf(directed_flag_persistence(H)[1][:, 1]).sum() == 1


# --- hard negatives ----------------------------------------------------------


def test_market_maker_two_sided_and_regularity_knob(window):
    tx, planted, _ = window
    for p in _of(planted, "market_maker"):
        t = _own_rows(tx, p)
        hub, clients = p.accounts[0], p.accounts[1:]
        two_sided = [((t.src == hub) & (t.dst == c)).any() and ((t.src == c) & (t.dst == hub)).any() for c in clients]
        assert np.mean(two_sided) >= 0.6
    # mm_regularity: 0 = random arrivals (gap CV ~1), near 1 = scheduled quoting (CV << 1)
    def median_cv(reg):
        tx, planted = generate(SyntheticConfig(seed=0, mm_regularity=reg))
        cvs = []
        for p in _of(planted, "market_maker"):
            t = _own_rows(tx, p)
            cvs += [_gap_cv(t.loc[(t.src == c) | (t.dst == c), "timestamp"].to_numpy()) for c in p.accounts[1:]]
        return np.nanmedian(cvs)

    assert median_cv(0.0) > 0.6
    assert median_cv(0.9) < 0.3


def test_arbitrage_loop_profits_on_irregular_events(window):
    tx, planted, _ = window
    cfg = SyntheticConfig()
    for p in _of(planted, "arbitrage_loop"):
        t = _own_rows(tx, p)
        first, last = p.edges[0], p.edges[-1]
        v_first = t.loc[(t.src == first[0]) & (t.dst == first[1]), "amount"].sum()
        v_last = t.loc[(t.src == last[0]) & (t.dst == last[1]), "amount"].sum()
        assert np.isclose(v_last / v_first, (1 + cfg.arb_profit) ** 2, rtol=0.05)  # value grows, not net zero
        starts = t.loc[(t.src == first[0]) & (t.dst == first[1]), "timestamp"].to_numpy()
        assert _gap_cv(starts) > 0.6  # price-triggered, not clocked (wash rings: ~0.05)


def test_hot_wallet_is_balanced_hub(window):
    tx, planted, _ = window
    deg = (tx.groupby("src").dst.nunique()).add(tx.groupby("dst").src.nunique(), fill_value=0)
    for p in _of(planted, "hot_wallet"):
        t = _own_rows(tx, p)
        h = p.accounts[0]
        i, o = t.loc[t.dst == h, "amount"].sum(), t.loc[t.src == h, "amount"].sum()
        assert 1 - abs(i - o) / (i + o) > 0.95
        assert deg[h] >= deg.quantile(0.98)  # among the highest-degree accounts


# ---- dense_internal_chords (opt-in; the default must reproduce earlier windows exactly) ----

# sha256 prefixes of (transactions, planted) produced BEFORE the flag existed (default generator)
_PRE_FLAG_DIGESTS = {
    ((), 0): "2f5a1b7ef84558f7",
    ((), 1): "ba12666f8273253e",
    ((("hard_negatives", False),), 0): "ea381f0d7b005c74",
    ((("hard_negatives", False),), 1): "fecda2eee395edaf",
    ((("dense_neighbours", 12), ("dense_rings_per_size", 3)), 0): "8c6108209a4ee751",
    ((("dense_neighbours", 12), ("dense_rings_per_size", 3)), 7): "c1f1922a517406af",
}


def _digest(tx, planted):
    import hashlib

    h = hashlib.sha256(pd.util.hash_pandas_object(tx, index=True).values.tobytes())
    h.update(
        repr([(p.pattern_id, p.pattern, p.accounts, p.edges, p.scenario, p.context_edges, p.context_accounts) for p in planted]).encode()
    )
    return h.hexdigest()[:16]


def test_default_generator_unchanged_by_dense_internal_chords_flag():
    for (kw, seed), want in _PRE_FLAG_DIGESTS.items():
        assert SyntheticConfig().dense_internal_chords is True
        assert _digest(*generate(SyntheticConfig(seed=seed, **dict(kw)))) == want, (kw, seed)


def test_no_internal_chords_changes_only_internal_dense_context_edges():
    on = generate(SyntheticConfig(seed=0, hard_negatives=False))
    off = generate(SyntheticConfig(seed=0, hard_negatives=False, dense_internal_chords=False))
    (tx_on, pl_on), (tx_off, pl_off) = on, off
    members = {p.pattern_id: set(p.accounts) for p in pl_on if p.scenario == "dense"}
    internal = np.array(
        [p == "context" and s in members.get(i, ()) and d in members.get(i, ()) for s, d, i, p in zip(tx_on.src, tx_on.dst, tx_on.pattern_id, tx_on.pattern)]
    )
    assert internal.any()
    assert tx_on[~internal].reset_index(drop=True).equals(tx_off)  # same random stream elsewhere
    for a, b in zip(pl_on, pl_off):
        assert (a.pattern_id, a.accounts, a.edges) == (b.pattern_id, b.accounts, b.edges)
        if a.scenario == "dense":
            assert set(b.context_edges) == {e for e in a.context_edges if not (e[0] in a.accounts and e[1] in a.accounts)}
        else:
            assert a.context_edges == b.context_edges


def test_no_internal_chords_means_no_inside_only_filling_of_dense_rings():
    from ringtda.features import window_graphs
    from ringtda.topology.vectorize import ring_survives

    def inside_filled(seed, **kw):
        tx, planted = generate(SyntheticConfig(seed=seed, hard_negatives=False, **kw))
        _, _, Gs, _ = window_graphs(tx)
        out = []
        for p in planted:
            if p.scenario == "dense":
                members = set(p.accounts)
                out.append(not ring_survives(Gs.subgraph(members).copy(), p.edges))
        return out

    default = [f for s in range(4) for f in inside_filled(s)]
    off = [f for s in range(4) for f in inside_filled(s, dense_internal_chords=False)]
    assert any(default)  # the default does fill dense rings from inside (so the check is not vacuous)
    assert not any(off)  # outside fillers are still allowed, inside-only fillers are gone
