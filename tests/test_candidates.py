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
