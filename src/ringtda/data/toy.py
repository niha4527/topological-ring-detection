"""Tiny hand-built graphs with known topology, used by tests and notebooks."""

from __future__ import annotations

import networkx as nx


def _g(edges) -> nx.DiGraph:
    G = nx.DiGraph()
    for e in edges:
        u, v, w = e if len(e) == 3 else (*e, 0.1)
        G.add_edge(u, v, weight=w)
    return G


def cyclic_triangle(w: float = 0.1) -> nx.DiGraph:
    """A->B->C->A: a loop. Never filled -> one H1 bar (w, inf)."""
    return _g([(0, 1, w), (1, 2, w), (2, 0, w)])


def transitive_triangle(w: float = 0.1) -> nx.DiGraph:
    """A->B, B->C, A->C: one-way flow. Filled at once -> no H1 bar."""
    return _g([(0, 1, w), (1, 2, w), (0, 2, w)])


def reciprocal_pair(w: float = 0.1) -> nx.DiGraph:
    """A<->B ping-pong: two directed edges -> one H1 bar (w, inf)."""
    return _g([(0, 1, w), (1, 0, w)])


def shortcut_filled_triangle(w_ring: float = 0.1, w_fill: float = 0.6) -> nx.DiGraph:
    """Cyclic triangle 0->1->2->0 plus account 3 sending to all three.

    The triangles (3,0,1), (3,1,2), (3,2,0) are transitive, so they enter the
    complex and their boundaries add up to the ring: the loop is born at
    w_ring and dies at w_fill -> one H1 bar (w_ring, w_fill).
    """
    return _g([(0, 1, w_ring), (1, 2, w_ring), (2, 0, w_ring), (3, 0, w_fill), (3, 1, w_fill), (3, 2, w_fill)])


def square(w: float = 0.1) -> nx.DiGraph:
    """Chordless directed 4-cycle -> one H1 bar (w, inf)."""
    return _g([(0, 1, w), (1, 2, w), (2, 3, w), (3, 0, w)])
