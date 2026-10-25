"""Weighted directed-flag-complex persistence (proposal v3 §3.2).

Directed flag complex:
    0-cells  every account (filtration 0)
    1-cells  every directed edge a->b (a reciprocal pair a<->b gives TWO cells,
             so ping-pong trading forms its own H1 loop)
    2-cells  ordered triples (a, b, c) with a->b, a->c, b->c (a one-way
             "transitive" triangle). A cyclic triangle a->b->c->a has no such
             ordering, so it is never filled and its H1 class persists.
Filtration: a cell enters at the max weight of its edges (sublevel set; low
weight = suspicious = enters first).

Backends
    "flagser"    pyflagser (fast C++; Python <= 3.12)
    "reference"  pure-Python boundary-matrix reduction over Z/2. Slow, but has
                 no dependencies; used to cross-check flagser in the tests and
                 as a fallback for small neighbourhoods.
    "auto"       flagser if importable, else reference.

Returns {0: array (n,2), 1: array (m,2)} of (birth, death); death = inf for
classes that never die. Zero-length bars are dropped.

Note: flagser works in float32, so its values differ from the reference
backend by ~1e-7 (0.017 -> 0.017000001). Compare diagrams with a tolerance.
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def available_backends() -> list[str]:
    out = ["reference"]
    try:
        import pyflagser  # noqa: F401

        out.insert(0, "flagser")
    except ImportError:
        pass
    return out


def _clean(dgm) -> np.ndarray:
    a = np.asarray(dgm, dtype=float).reshape(-1, 2)
    if len(a) == 0:
        return a
    a = a[a[:, 1] > a[:, 0]]
    return a[np.lexsort((a[:, 1], a[:, 0]))]


def _flagser(G: nx.DiGraph, weight: str) -> dict[int, np.ndarray]:
    import pyflagser
    from scipy.sparse import coo_matrix

    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    if G.number_of_edges() == 0:
        return {0: _clean([[0.0, np.inf]] * n), 1: _clean([])}
    rows, cols, data = zip(*((idx[u], idx[v], float(d[weight])) for u, v, d in G.edges(data=True)))
    # sparse input: stored entries = edges (explicit zeros are zero-weight
    # edges), unstored = no edge, diagonal = vertex weights (0)
    M = coo_matrix((data, (rows, cols)), shape=(n, n))
    # max_dimension=2 so that 2-cells exist and H1 classes can die.
    out = pyflagser.flagser_weighted(M, max_dimension=2, directed=True, filtration="max")
    dgms = out["dgms"]
    return {0: _clean(dgms[0]), 1: _clean(dgms[1]) if len(dgms) > 1 else _clean([])}


def _reference(G: nx.DiGraph, weight: str) -> dict[int, np.ndarray]:
    w = {(u, v): float(d[weight]) for u, v, d in G.edges(data=True) if u != v}
    cells = []  # (filtration, dim, key)
    for v in G.nodes():
        cells.append((0.0, 0, ("v", v)))
    for (u, v), x in w.items():
        cells.append((x, 1, ("e", u, v)))
    succ = {n: set() for n in G.nodes()}
    for u, v in w:
        succ[u].add(v)
    for a in G.nodes():
        for b in succ[a]:
            for c in succ[a] & succ[b]:
                if c != a and c != b:
                    cells.append((max(w[a, b], w[a, c], w[b, c]), 2, ("t", a, b, c)))
    cells.sort(key=lambda c: (c[0], c[1]))

    index = {c[2]: i for i, c in enumerate(cells)}
    pivot: dict[int, int] = {}
    reduced: list[set] = []
    for j, (_, dim, key) in enumerate(cells):
        if dim == 0:
            col = set()
        elif dim == 1:
            _, u, v = key
            col = {index[("v", u)], index[("v", v)]}
        else:
            _, a, b, c = key
            col = {index[("e", b, c)], index[("e", a, c)], index[("e", a, b)]}
        while col:
            low = max(col)
            if low in pivot:
                col ^= reduced[pivot[low]]
            else:
                pivot[low] = j
                break
        reduced.append(col)

    bars: dict[int, list] = {0: [], 1: []}
    paired = set(pivot)
    for low, j in pivot.items():
        d = cells[low][1]
        if d in bars:
            bars[d].append((cells[low][0], cells[j][0]))
    for i, (f, d, _) in enumerate(cells):
        if d in bars and not reduced[i] and i not in paired:
            bars[d].append((f, np.inf))
    return {0: _clean(bars[0]), 1: _clean(bars[1])}


def directed_flag_persistence(G: nx.DiGraph, weight: str = "weight", backend: str = "auto") -> dict[int, np.ndarray]:
    """H0 and H1 persistence diagrams of the weighted directed flag complex of G."""
    if backend == "auto":
        backend = available_backends()[0]
    if backend == "flagser":
        return _flagser(G, weight)
    if backend == "reference":
        return _reference(G, weight)
    raise ValueError(f"unknown backend {backend!r}")
