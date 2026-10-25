"""Engineering gate (proposal v3 §3.7, Phase 2): the directed flag complex
behaves as §3.2 claims. If these fail, nothing downstream is meaningful."""

import networkx as nx
import numpy as np
import pytest

from ringtda.data import toy
from ringtda.topology.persistence import available_backends, directed_flag_persistence

BACKENDS = available_backends()


def h1(G, backend):
    return directed_flag_persistence(G, backend=backend)[1]


def test_flagser_is_installed():
    # Not a hard failure of the method, but you want the fast backend.
    if "flagser" not in BACKENDS:
        pytest.skip("pyflagser not installed: only the slow reference backend is available")


@pytest.mark.parametrize("backend", BACKENDS)
def test_cyclic_triangle_never_dies(backend):
    d = h1(toy.cyclic_triangle(0.1), backend)
    assert d.shape == (1, 2)
    assert np.isclose(d[0, 0], 0.1) and np.isinf(d[0, 1])


@pytest.mark.parametrize("backend", BACKENDS)
def test_transitive_triangle_is_filled(backend):
    assert len(h1(toy.transitive_triangle(0.1), backend)) == 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_reciprocal_pair_is_a_loop(backend):
    d = h1(toy.reciprocal_pair(0.1), backend)
    assert d.shape == (1, 2) and np.isinf(d[0, 1])


@pytest.mark.parametrize("backend", BACKENDS)
def test_shortcut_filled_triangle_dies(backend):
    d = h1(toy.shortcut_filled_triangle(0.1, 0.6), backend)
    assert d.shape == (1, 2)
    assert np.allclose(d[0], [0.1, 0.6])


@pytest.mark.parametrize("backend", BACKENDS)
def test_square_persists(backend):
    d = h1(toy.square(0.1), backend)
    assert d.shape == (1, 2) and np.isinf(d[0, 1])


@pytest.mark.parametrize("backend", BACKENDS)
def test_h0_counts_components(backend):
    G = nx.disjoint_union(toy.cyclic_triangle(), toy.square())
    d0 = directed_flag_persistence(G, backend=backend)[0]
    assert np.isinf(d0[:, 1]).sum() == 2


def _canon(d):
    # pyflagser stores weights as float32 (0.017 -> 0.017000001), so compare
    # to 5 decimals rather than exactly.
    d = np.round(d, 5)
    return sorted(map(tuple, d.tolist()))


@pytest.mark.skipif("flagser" not in BACKENDS, reason="needs pyflagser")
@pytest.mark.parametrize("seed", range(20))
def test_flagser_matches_reference(seed):
    """Random small weighted digraphs: both backends give identical diagrams."""
    rng = np.random.default_rng(seed)
    G = nx.gnp_random_graph(9, 0.35, seed=seed, directed=True)
    for u, v in G.edges():
        G.edges[u, v]["weight"] = float(np.round(rng.uniform(0, 1), 3))
    a = directed_flag_persistence(G, backend="flagser")
    b = directed_flag_persistence(G, backend="reference")
    for dim in (0, 1):
        assert _canon(a[dim]) == _canon(b[dim]), f"H{dim} differs for seed {seed}"
