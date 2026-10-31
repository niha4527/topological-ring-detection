import networkx as nx

from ringtda.data import toy
from ringtda.topology.candidates import candidate_rings, neighborhood, sparsify


def test_finds_cyclic_triangle_not_transitive():
    G = nx.disjoint_union(toy.cyclic_triangle(0.1), toy.transitive_triangle(0.1))
    res = candidate_rings(G, tau=0.5)
    assert [c.length for c in res.candidates] == [3]


def test_two_cycles_are_candidates():
    res = candidate_rings(toy.reciprocal_pair(0.1), tau=0.5)
    assert len(res.candidates) == 1 and res.candidates[0].length == 2


def test_sparsify_removes_heavy_edges():
    G = toy.cyclic_triangle(0.1)
    G.edges[2, 0]["weight"] = 0.9
    assert sparsify(G, 0.5).number_of_edges() == 2
    assert candidate_rings(G, tau=0.5).candidates == []


def test_birth_is_max_edge_weight():
    G = toy.square(0.1)
    G.edges[1, 2]["weight"] = 0.3
    (c,) = candidate_rings(G, tau=0.5).candidates
    assert abs(c.birth - 0.3) < 1e-12


def test_length_bound():
    G = nx.DiGraph()
    nx.add_cycle(G, range(8), weight=0.1)
    assert candidate_rings(G, tau=0.5, max_len=6).candidates == []
    assert len(candidate_rings(G, tau=0.5, max_len=8).candidates) == 1


def test_neighborhood_one_hop():
    G = toy.shortcut_filled_triangle()
    H = neighborhood(G, [0, 1, 2], hops=1)
    assert set(H.nodes()) == {0, 1, 2, 3}


def test_dedupe_keeps_ring_over_lower_birth_detour():
    # ring 0->1->2->3->4->0 (birth 0.3); detour 4->5->0 replaces edge 4->0 with lower birth 0.2.
    # The detour covers 4/5 = 80% of the ring's edges but must not replace it.
    G = nx.DiGraph()
    nx.add_cycle(G, range(5), weight=0.1)
    G.edges[4, 0]["weight"] = 0.3
    G.add_edge(4, 5, weight=0.2)
    G.add_edge(5, 0, weight=0.2)
    found = {c.nodes for c in candidate_rings(G, tau=0.5).candidates}
    assert (0, 1, 2, 3, 4) in found



def test_dedupe_drops_longer_cycle_covered_by_shorter_one():
    # same graph; at overlap 0.5 the 6-cycle detour (4/6 of its edges in the ring) is dropped
    G = nx.DiGraph()
    nx.add_cycle(G, range(5), weight=0.1)
    G.edges[4, 0]["weight"] = 0.3
    G.add_edge(4, 5, weight=0.2)
    G.add_edge(5, 0, weight=0.2)
    assert [c.nodes for c in candidate_rings(G, tau=0.5, dedupe_overlap=0.5).candidates] == [(0, 1, 2, 3, 4)]
    assert len(candidate_rings(G, tau=0.5).candidates) == 2  # default 0.8: both kept


def test_dedup_removes_nothing_at_default_overlap():
    """§3.3 80%-shared-edge dedup removes nothing for max_len <= 9 and first acts at 10.

    Verified on dense synthetic windows (seeds 0-1, tau quantile 0.15 and 0.25: 0 removed up to 9,
    44-2316 removed at 10). Theory: a distinct simple cycle shares at most L-2 of a length-L ring's
    edges, so the shared fraction reaches 0.8 only at L = 10.
    """
    from ringtda.data.synthetic import SyntheticConfig, generate
    from ringtda.features import window_graphs

    tx, _ = generate(SyntheticConfig(seed=0, n_accounts=300, n_background=3000, hard_negatives=False))
    _, G, _, tau = window_graphs(tx, 0.15)

    def n(max_len, overlap):
        return len(candidate_rings(G, tau=tau, max_len=max_len, dedupe_overlap=overlap).candidates)

    for max_len in (6, 9):
        assert n(max_len, 1.01) > 100 and n(max_len, 0.8) == n(max_len, 1.01)  # 1.01 never drops
    assert n(10, 0.8) < n(10, 1.01)
    assert n(6, 0.5) < n(6, 1.01)  # the dedup does work at a lower threshold, so the test is not vacuous


def _shortcut_cycle(length):
    G = nx.DiGraph()
    for i in range(length):
        G.add_edge(i, (i + 1) % length, weight=0.1)
    G.add_edge(0, 2, weight=0.1)  # bypasses node 1: a (length-1)-cycle sharing length-2 edges
    return G


def test_dedup_first_acts_at_cycle_length_ten():
    for length, kept in ((9, 2), (10, 1)):
        G = _shortcut_cycle(length)
        assert len(candidate_rings(G, tau=1, max_len=length, dedupe_overlap=1.01).candidates) == 2
        assert len(candidate_rings(G, tau=1, max_len=length, dedupe_overlap=0.8).candidates) == kept
