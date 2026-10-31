"""Candidate ring generation (proposal v3 §3.3).

sparsify -> strongly connected components -> bounded simple directed cycles
-> de-duplicate. Each surviving cycle is an explicit, account-level candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import islice

import networkx as nx


@dataclass
class Candidate:
    nodes: tuple  # cycle order, rotated so the smallest node comes first
    edges: tuple[tuple, ...]
    birth: float  # max edge weight on the ring = filtration value where it first exists

    @property
    def length(self) -> int:
        return len(self.nodes)


@dataclass
class CandidateResult:
    candidates: list[Candidate]
    truncated_sccs: list[int] = field(default_factory=list)  # sizes of SCCs that hit the cap
    n_sccs: int = 0


def sparsify(G: nx.DiGraph, tau: float, weight: str = "weight") -> nx.DiGraph:
    """Keep only edges with weight <= tau (the suspicious end of the filtration)."""
    H = nx.DiGraph()
    H.add_edges_from((u, v, d) for u, v, d in G.edges(data=True) if d[weight] <= tau)
    return H


def _canonical(cycle: list) -> tuple:
    i = min(range(len(cycle)), key=lambda k: cycle[k])
    return tuple(cycle[i:] + cycle[:i])


def candidate_rings(
    G: nx.DiGraph,
    tau: float,
    max_len: int = 6,
    dedupe_overlap: float = 0.8,
    max_cycles_per_scc: int = 20000,
    weight: str = "weight",
) -> CandidateResult:
    """Enumerate candidate rings on the sparsified graph.

    dedupe_overlap: a cycle is dropped if >= this fraction of its edges are
    already covered by a single kept cycle that is shorter, or of equal length
    with lower (or equal) birth. Shorter cycles go first so that a planted ring
    is never replaced by a longer detour around it. (§3.3 says "keep the
    lowest-birth one"; this deviation is noted in the README.)
    Note: a k-cycle shares at most k-2 edges with any other cycle of length
    <= k (k-1 shared edges would force the closing edge), so at the default 0.8
    nothing is dropped unless max_len >= 10 (verified: none up to 9). Lower dedupe_overlap to merge more.
    max_cycles_per_scc: enumeration cap per SCC; capped SCCs are reported,
    never silently dropped.
    """
    H = sparsify(G, tau, weight)
    cycles: list[Candidate] = []
    truncated: list[int] = []
    sccs = [c for c in nx.strongly_connected_components(H) if len(c) >= 2]
    for comp in sccs:
        sub = H.subgraph(comp)
        found = list(islice(nx.simple_cycles(sub, length_bound=max_len), max_cycles_per_scc + 1))
        if len(found) > max_cycles_per_scc:
            truncated.append(len(comp))
            found = found[:max_cycles_per_scc]
        for cyc in found:
            if len(cyc) < 2:
                continue  # self-loops are not rings
            nodes = _canonical(list(cyc))
            edges = tuple((nodes[i], nodes[(i + 1) % len(nodes)]) for i in range(len(nodes)))
            birth = max(H.edges[e][weight] for e in edges)
            cycles.append(Candidate(nodes, edges, birth))

    # de-duplicate: shorter first, then lowest birth (§3.3, 80% shared edges). With
    # dedupe_overlap=0.8 this removes nothing for max_len <= 9 (verified on dense windows; first
    # removals at max_len 10): a distinct cycle shares at most L-2 edges with a length-L ring, a
    # fraction <= 1 - 2/L, which reaches 0.8 only at L = 10
    # (tests/test_candidates.py::test_dedup_removes_nothing_at_default_overlap). Near-duplicate rings
    # therefore all stay candidates; Phase 3 CV must group by ring identity.
    cycles.sort(key=lambda c: (c.length, c.birth, c.nodes))
    kept: list[Candidate] = []
    kept_sets: list[set] = []
    for c in cycles:
        es = set(c.edges)
        if any(len(es & k) / len(es) >= dedupe_overlap for k in kept_sets):
            continue
        kept.append(c)
        kept_sets.append(es)
    return CandidateResult(kept, truncated, len(sccs))


def neighborhood(G: nx.DiGraph, nodes, hops: int = 1) -> nx.DiGraph:
    """Induced subgraph on `nodes` plus everything within `hops` (either direction)."""
    U = G.to_undirected(as_view=True)
    keep = set(nodes)
    frontier = set(nodes)
    for _ in range(hops):
        frontier = {n for f in frontier for n in U.neighbors(f)} - keep
        keep |= frontier
    return G.subgraph(keep).copy()
