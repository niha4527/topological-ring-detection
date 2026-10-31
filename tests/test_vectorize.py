"""Engineering gate (proposal v3 §3.4): per-candidate T features are well-formed and
reflect known topology on tiny cases. Correctness only; nothing here asserts that T helps."""

import numpy as np
import pytest

from ringtda.data import toy
from ringtda.topology.candidates import Candidate, candidate_rings
from ringtda.topology.persistence import available_backends
from ringtda.topology.vectorize import (
    cap_diagram,
    feature_names,
    load_config,
    local_topology_features,
    resolve_cap,
)

BACKENDS = available_backends()
CFG = load_config()


def ring_of(G, nodes):
    nodes = tuple(nodes)
    edges = tuple((nodes[i], nodes[(i + 1) % len(nodes)]) for i in range(len(nodes)))
    return Candidate(nodes, edges, max(G.edges[e]["weight"] for e in edges))


def land(f):
    return np.array([v for k, v in f.items() if k.startswith("T_local_land_")])


def test_isolated_cyclic_triangle():
    G = toy.cyclic_triangle(0.1)
    f = local_topology_features(G, ring_of(G, (0, 1, 2)), window_max=0.9, config=CFG)
    assert f["T_local_inf_h1"] == 1 and f["T_local_h1_count"] == 1
    assert land(f).max() > 0
    # the infinite bar is born at the ring's max edge weight and capped at the window max
    assert np.isclose(f["T_local_total_pers"], 0.9 - 0.1, atol=1e-5)


def test_transitive_triangle_has_no_loop():
    G = toy.transitive_triangle(0.1)
    cyc = Candidate((0, 1, 2), ((0, 1), (1, 2), (2, 0)), 0.1)  # nominal; no cycle exists
    f = local_topology_features(G, cyc, window_max=0.9, config=CFG)
    assert f["T_local_inf_h1"] == 0 and f["T_local_h1_count"] == 0
    assert not land(f).any()


def test_shortcut_filled_triangle_has_no_infinite_bar():
    G = toy.shortcut_filled_triangle(0.1, 0.6)
    f = local_topology_features(G, ring_of(G, (0, 1, 2)), window_max=0.9, config=CFG)
    assert f["T_local_inf_h1"] == 0
    assert f["T_local_h1_count"] == 1  # born 0.1, dies 0.6
    assert np.isclose(f["T_local_total_pers"], 0.5, atol=1e-5)


def test_neighbour_outside_ring_is_included():
    G = toy.shortcut_filled_triangle()
    # without the 1-hop neighbour (node 3) the ring would look isolated and infinite
    assert local_topology_features(G, ring_of(G, (0, 1, 2)), 0.9, CFG)["T_local_inf_h1"] == 0


def test_empty_diagram_and_zero_vector_length():
    G = toy.transitive_triangle()
    f = local_topology_features(G, Candidate((0, 1, 2), (), 0.1), 0.9, CFG)
    lc = CFG["landscape"]
    assert len(land(f)) == lc["num_landscapes"] * lc["resolution"]
    assert f["T_local_entropy"] == 0.0 and f["T_local_total_pers"] == 0.0


def test_birth_at_or_above_cap_is_counted_but_not_vectorised():
    G = toy.cyclic_triangle(0.5)
    f = local_topology_features(G, ring_of(G, (0, 1, 2)), window_max=0.5, config=CFG)  # birth == cap
    assert f["T_local_inf_h1"] == 1 and f["T_local_h1_count"] == 1
    assert not land(f).any() and f["T_local_total_pers"] == 0.0


def test_cap_diagram_rules():
    h1 = np.array([[0.1, np.inf], [0.2, 0.4], [0.8, 0.95], [0.95, np.inf]])
    d, n = cap_diagram(h1, cap=0.9)
    assert n == 2  # both infinite bars counted, including the one born above the cap
    assert np.allclose(d, [[0.1, 0.9], [0.2, 0.4], [0.8, 0.9]])  # inf and 0.95 clipped; birth 0.95 dropped
    assert cap_diagram(np.zeros((0, 2)), 0.9)[1] == 0


def test_cap_rule_changes_result():
    G = toy.cyclic_triangle(0.1)
    c = ring_of(G, (0, 1, 2))
    res = {}
    for rule in ("window_max", "fixed", "window_max_x"):
        cfg = {**CFG, "cap": {"rule": rule, "fixed": 0.5, "factor": 1.1}}
        res[rule] = local_topology_features(G, c, 0.8, cfg)["T_local_total_pers"]
    assert np.isclose(res["window_max"], 0.8 - 0.1, atol=1e-5)
    assert np.isclose(res["fixed"], 0.5 - 0.1, atol=1e-5)
    assert np.isclose(res["window_max_x"], 0.88 - 0.1, atol=1e-5)
    assert resolve_cap(0.8, {"rule": "window_max_x", "factor": 1.1}) == pytest.approx(0.88)
    with pytest.raises(ValueError):
        resolve_cap(0.8, {"rule": "nope"})


@pytest.mark.parametrize("make", [toy.cyclic_triangle, toy.shortcut_filled_triangle, toy.square, toy.reciprocal_pair])
def test_backends_agree(make):
    if len(BACKENDS) < 2:
        pytest.skip("only one backend available")
    G = make()
    nodes = [0, 1] if make is toy.reciprocal_pair else (0, 1, 2) if make is not toy.square else (0, 1, 2, 3)
    c = ring_of(G, nodes)
    a = local_topology_features(G, c, 0.9, CFG, backend="flagser")
    b = local_topology_features(G, c, 0.9, CFG, backend="reference")
    assert list(a) == list(b)
    assert np.allclose(list(a.values()), list(b.values()), atol=1e-5)  # flagser is float32


def test_fixed_keys_and_no_nan_across_candidates():
    G = toy.shortcut_filled_triangle()
    G.add_edges_from([(4, 5, {"weight": 0.2}), (5, 4, {"weight": 0.3})])
    res = candidate_rings(G, tau=0.7, max_len=4)
    assert len(res.candidates) >= 2
    names = feature_names(CFG)
    for c in res.candidates:
        f = local_topology_features(G, c, 0.7, CFG)
        assert list(f) == names
        assert np.isfinite(list(f.values())).all()


def test_ring_inf_matches_the_rings_own_bar():
    G = toy.cyclic_triangle(0.1)
    assert local_topology_features(G, ring_of(G, (0, 1, 2)), 0.9, CFG)["T_local_ring_inf"] == 1.0
    G = toy.shortcut_filled_triangle(0.1, 0.6)
    assert local_topology_features(G, ring_of(G, (0, 1, 2)), 0.9, CFG)["T_local_ring_inf"] == 0.0


def test_ring_inf_ignores_other_infinite_loops_in_the_neighbourhood():
    # shortcut-filled ring (0,1,2) plus an unrelated ping-pong pair 0 <-> 5 born at 0.2
    G = toy.shortcut_filled_triangle(0.1, 0.6)
    G.add_edge(0, 5, weight=0.2)
    G.add_edge(5, 0, weight=0.2)
    f = local_topology_features(G, ring_of(G, (0, 1, 2)), 0.9, CFG)
    assert f["T_local_inf_h1"] == 1.0 and f["T_local_ring_inf"] == 0.0


def test_ring_inf_tolerates_float32_birth():
    from ringtda.topology.vectorize import ring_has_infinite_bar

    assert ring_has_infinite_bar(np.array([[np.float32(0.017), np.inf]]), 0.017)
    assert not ring_has_infinite_bar(np.array([[0.2, np.inf], [0.1, 0.5]]), 0.1)


def test_ring_survives_exact_cases():
    from ringtda.topology.vectorize import ring_survives

    G = toy.cyclic_triangle()
    assert ring_survives(G, ring_of(G, (0, 1, 2)).edges) is True
    G = toy.transitive_triangle()  # no directed cycle: the nominal ring (0,1,2) has no edge 2->0
    assert ring_survives(G, ((0, 1), (1, 2), (2, 0))) is False
    G = toy.shortcut_filled_triangle()
    assert ring_survives(G, ring_of(G, (0, 1, 2)).edges) is False
    G = toy.reciprocal_pair()  # a 2-cycle is two distinct edges
    assert ring_survives(G, ((0, 1), (1, 0))) is True
    assert ring_survives(toy.square(), ring_of(toy.square(), (0, 1, 2, 3)).edges) is True


def test_ring_survives_ignores_filtration_values():
    from ringtda.topology.vectorize import ring_survives

    G = toy.shortcut_filled_triangle(0.1, 0.99)  # filled very late, but filled
    assert ring_survives(G, ring_of(G, (0, 1, 2)).edges) is False


def test_ring_survives_size_guard_and_fallback(caplog):
    from ringtda.topology.vectorize import ring_survives

    G = toy.cyclic_triangle()
    assert ring_survives(G, ring_of(G, (0, 1, 2)).edges, max_edges=2) is None
    assert ring_survives(G, ring_of(G, (0, 1, 2)).edges, max_triangles=0) is True  # no triangles: guard not hit
    G = toy.shortcut_filled_triangle()
    assert ring_survives(G, ring_of(G, (0, 1, 2)).edges, max_triangles=1) is None
    cfg = {**CFG, "local": {**CFG["local"], "ring_inf_limits": {"max_edges": 2, "max_triangles": 10}}}
    G = toy.cyclic_triangle()
    with caplog.at_level("WARNING"):
        f = local_topology_features(G, ring_of(G, (0, 1, 2)), 0.9, cfg)
    assert f["T_local_ring_inf"] == 1.0  # fallback = birth-match proxy
    assert "fallback" in caplog.text
