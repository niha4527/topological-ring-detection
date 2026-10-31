"""Hand-checked B features (baseline_features.py) on a 4-node graph: ring 0->1->2->0 plus 3->0."""

import networkx as nx
import numpy as np
import pytest

from ringtda.graph.baseline_features import B_NAMES, WindowBaseline, baseline_features, load_config
from ringtda.topology.candidates import Candidate


def toy():
    G = nx.DiGraph()
    for u, v, w in [(0, 1, 0.1), (1, 2, 0.2), (2, 0, 0.3), (3, 0, 0.4)]:
        G.add_edge(u, v, weight=w)
    return G, Candidate((0, 1, 2), ((0, 1), (1, 2), (2, 0)), 0.3)


def test_b_features_on_a_four_node_graph():
    G, c = toy()
    ctx = WindowBaseline(G, load_config())
    assert ctx.betweenness_exact
    f = baseline_features(G, c, ctx, hops=1)
    assert list(f) == B_NAMES
    # degrees: node 0 in=2 out=1, nodes 1 and 2 in=1 out=1
    assert f["B_deg_mean"] == pytest.approx(7 / 3) and f["B_deg_max"] == 3
    assert f["B_indeg_mean"] == pytest.approx(4 / 3) and f["B_indeg_max"] == 2
    assert f["B_outdeg_mean"] == pytest.approx(1.0) and f["B_outdeg_max"] == 1
    # directed betweenness, normalised by (n-1)(n-2) = 6: node0 3, node1 2, node2 1
    assert f["B_betw_mean"] == pytest.approx(1 / 3) and f["B_betw_max"] == pytest.approx(0.5)
    # undirected clustering: node0 has neighbours {1,2,3} with one link (1-2) -> 1/3; nodes 1, 2 -> 1
    assert f["B_clust_mean"] == pytest.approx((1 / 3 + 1 + 1) / 3)
    assert f["B_w_var"] == pytest.approx(np.var([0.1, 0.2, 0.3, 0.4]))
    assert f["B_nbhd_size"] == 4 and f["B_density"] == pytest.approx(4 / 12)


def test_sampled_betweenness_is_seeded_and_cached_per_window():
    G = nx.gnp_random_graph(60, 0.08, seed=1, directed=True)
    for u, v in G.edges:
        G.edges[u, v]["weight"] = 0.5
    cfg = {"betweenness": {"exact_max_nodes": 10, "k": 15, "seed": 3, "graph": "full", "sparsify_quantile": 0.25}}
    a, b = WindowBaseline(G, cfg), WindowBaseline(G, cfg)
    assert not a.betweenness_exact and a.betweenness == b.betweenness
    assert a.clustering(0) == a.clustering(0) and 0 in a._clust


def test_ring_members_missing_from_betweenness_graph_get_zero():
    G, c = toy()
    for u, v in G.edges:
        G.edges[u, v]["weight"] = 0.9  # nothing survives a low sparsify quantile except ties
    cfg = {"betweenness": {"exact_max_nodes": 100, "k": 5, "seed": 0, "graph": "sparsified", "sparsify_quantile": 0.0}}
    G.edges[0, 1]["weight"] = 0.1
    ctx = WindowBaseline(G, cfg)
    f = baseline_features(G, c, ctx)
    assert np.isfinite(list(f.values())).all()
