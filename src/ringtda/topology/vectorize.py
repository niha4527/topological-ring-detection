"""Per-candidate topological (T) features (proposal v3 §3.4).

Local persistence of a candidate ring: weighted directed-flag persistence (H0, H1, complex
up to dimension 2) of the induced subgraph on the ring plus its 1-hop neighbourhood, then
a fixed-length vectorisation of the H1 diagram.

Only topological features are produced here. Ring birth, degrees, centrality and
behavioural statistics belong to D / B (§3.4), not T.

Infinite bars (§3.4). Landscapes need finite deaths, so a class that never dies gets
death = cap, where the cap is a documented rule (config `cap.rule`): the window's max edge
weight (default), a fixed constant, or window max * factor. Infinite H1 bars are counted
separately *before* capping, so the count does not depend on the cap rule. A bar born at or
above the cap has no extent below it: it is dropped from the landscape input and the
scalar persistence summaries but still counted. A finite death above the cap (possible
with the fixed rule) is clipped to the cap. Because the cap fixes how much persistence an
infinite bar carries, sensitivity to the rule is to be checked in the Phase 3 sweeps.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import networkx as nx
import numpy as np
import yaml

from ringtda.topology.candidates import Candidate, neighborhood
from ringtda.topology.persistence import directed_flag_persistence

log = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "topology.yaml"
CAP_RULES = ("window_max", "fixed", "window_max_x")


def load_config(path: str | Path | None = None) -> dict:
    with open(path or CONFIG_PATH) as f:
        return yaml.safe_load(f)


def resolve_cap(window_max: float, cap_cfg: dict) -> float:
    """Value that replaces an infinite death."""
    rule = cap_cfg["rule"]
    if rule == "window_max":
        return float(window_max)
    if rule == "fixed":
        return float(cap_cfg["fixed"])
    if rule == "window_max_x":
        return float(window_max) * float(cap_cfg["factor"])
    raise ValueError(f"unknown cap rule {rule!r}; expected one of {CAP_RULES}")


def cap_diagram(h1: np.ndarray, cap: float) -> tuple[np.ndarray, int]:
    """Return (finite H1 diagram for vectorising, number of infinite bars).

    Bars with birth >= cap are removed from the diagram but are still in the count.
    """
    h1 = np.asarray(h1, dtype=float).reshape(-1, 2)
    n_inf = int(np.isinf(h1[:, 1]).sum())
    if len(h1) == 0:
        return h1, 0
    d = np.minimum(h1[:, 1], cap)
    keep = h1[:, 0] < cap
    return np.column_stack([h1[keep, 0], d[keep]]), n_inf


@lru_cache(maxsize=None)
def _landscape(num: int, res: int, sample_range: tuple[float, float]):
    from gudhi.representations import Landscape

    L = Landscape(num_landscapes=num, resolution=res, sample_range=list(sample_range))
    L.fit([np.zeros((0, 2))])  # explicit sample_range: the grid does not depend on the data
    return L


def landscape_vector(dgm: np.ndarray, num_landscapes: int, resolution: int, sample_range) -> np.ndarray:
    """Flat landscape vector of length num_landscapes * resolution; zeros for an empty diagram."""
    if len(dgm) == 0:
        return np.zeros(num_landscapes * resolution)
    L = _landscape(int(num_landscapes), int(resolution), tuple(float(x) for x in sample_range))
    return L.transform([dgm])[0]


def persistence_entropy(lifetimes: np.ndarray) -> float:
    s = float(np.sum(lifetimes))
    if len(lifetimes) == 0 or s <= 0:
        return 0.0
    p = lifetimes / s
    return float(-np.sum(p * np.log(p)))


def feature_names(config: dict | None = None, prefix: str = "T_local") -> list[str]:
    lc = (config or load_config())["landscape"]
    land = [f"{prefix}_land_{k}_{j}" for k in range(lc["num_landscapes"]) for j in range(lc["resolution"])]
    scalars = ["h1_count", "inf_h1", "h0_count", "total_pers", "entropy"]
    if prefix == "T_local":
        scalars.append("ring_inf")
    return land + [f"{prefix}_{s}" for s in scalars]


def local_graph(full: nx.DiGraph, sparsified: nx.DiGraph, config: dict | None = None) -> nx.DiGraph:
    """Graph the candidate neighbourhood is taken from, per config `local.local_graph`."""
    choice = (config or load_config())["local"]["local_graph"]
    if choice == "sparsified":
        return sparsified
    if choice == "full":
        return full
    raise ValueError(f"unknown local_graph {choice!r}; expected 'sparsified' or 'full'")


def local_diagrams(G: nx.DiGraph, candidate: Candidate, config: dict | None = None, backend: str | None = None):
    """(neighbourhood subgraph, {0: H0, 1: H1}) of the ring plus its hops-neighbourhood in G."""
    cfg = config or load_config()
    H = neighborhood(G, candidate.nodes, hops=cfg["local"]["hops"])
    return H, directed_flag_persistence(H, backend=backend or cfg["local"]["backend"])


def ring_has_infinite_bar(h1: np.ndarray, ring_birth: float) -> bool:
    """Birth-match proxy: an infinite H1 bar born at the ring's birth (within float32 tolerance).

    Only the fallback of `ring_survives` for oversized neighbourhoods. It misses rings whose loop
    is homologous to an earlier loop (the bar is then born before the ring), and a different
    loop born at the same weight would match.
    """
    h1 = np.asarray(h1, dtype=float).reshape(-1, 2)
    inf = h1[np.isinf(h1[:, 1]), 0]
    return bool(np.isclose(inf, ring_birth, rtol=1e-4, atol=1e-5).any())


def _in_span(vec: int, basis: dict[int, int]) -> bool:
    """Z/2 membership: reduce `vec` (a bitmask) against a basis keyed by highest set bit."""
    while vec:
        top = vec.bit_length() - 1
        if top not in basis:
            return False
        vec ^= basis[top]
    return True


def ring_survives(H: nx.DiGraph, ring_edges, max_edges: int = 5000, max_triangles: int = 200000) -> bool | None:
    """Exact test: is the ring's loop never filled in the directed flag complex of H?

    The ring's cycle is a Z/2 vector over the directed 1-simplices of H (a reciprocal pair is two
    distinct edges). The complex has dimension 2, so the loop survives (is a nonzero class of H1
    of the full complex) iff the vector is NOT in the column space of the Z/2 boundary matrix of
    the directed 2-simplices (a, b, c) with a->b, a->c, b->c, whose boundary is ab + bc + ac.
    Filtration values are ignored: the question is whether the loop is ever filled, which is what
    an infinite bar means under the window-max cap.

    Ring membership only picks the cycle; ring birth is not used. The column stays in T because
    it requires this homology computation (§3.4). Returns False if a ring edge is not in H (no
    such loop), and None if H exceeds max_edges / max_triangles (the caller falls back).
    Elimination is Z/2 Gaussian elimination on Python-int bitmasks (equivalent to a bool matrix).
    """
    edges = [(u, v) for u, v in H.edges() if u != v]
    if len(edges) > max_edges:
        return None
    ring = list(ring_edges)
    if not ring or any(e not in H.edges or e[0] == e[1] for e in ring):
        return False
    idx = {e: i for i, e in enumerate(edges)}
    succ = {n: set() for n in H.nodes()}
    for u, v in edges:
        succ[u].add(v)
    basis: dict[int, int] = {}
    n_tri = 0
    for a in H.nodes():
        for b in succ[a]:
            for c in succ[a] & succ[b]:
                if c == a or c == b:
                    continue
                n_tri += 1
                if n_tri > max_triangles:
                    return None
                col = (1 << idx[a, b]) | (1 << idx[b, c]) | (1 << idx[a, c])
                while col:
                    top = col.bit_length() - 1
                    if top in basis:
                        col ^= basis[top]
                    else:
                        basis[top] = col
                        break
    z = 0
    for e in ring:
        z ^= 1 << idx[e]
    return not _in_span(z, basis)


def vectorize_diagrams(
    dgms: dict, window_max: float, config: dict, prefix: str, ring_inf: float | None = None
) -> dict[str, float]:
    """Capped H1 landscape plus scalar summaries of {0: H0, 1: H1}, keys `<prefix>_*`.

    With ring_inf, also `<prefix>_ring_inf` (the value comes from `ring_survives`).
    """
    cap = resolve_cap(window_max, config["cap"])
    capped, n_inf = cap_diagram(dgms[1], cap)
    lc = config["landscape"]
    land = landscape_vector(capped, lc["num_landscapes"], lc["resolution"], lc["sample_range"])
    life = capped[:, 1] - capped[:, 0] if len(capped) else np.zeros(0)
    out = {
        f"{prefix}_land_{k}_{j}": float(land[k * lc["resolution"] + j])
        for k in range(lc["num_landscapes"])
        for j in range(lc["resolution"])
    }
    out.update(
        {
            f"{prefix}_h1_count": float(len(dgms[1])),
            f"{prefix}_inf_h1": float(n_inf),
            f"{prefix}_h0_count": float(len(dgms[0])),
            f"{prefix}_total_pers": float(life.sum()),
            f"{prefix}_entropy": persistence_entropy(life),
        }
    )
    if ring_inf is not None:
        out[f"{prefix}_ring_inf"] = float(ring_inf)
    return out


def local_topology_features(
    G: nx.DiGraph, candidate: Candidate, window_max: float, config: dict | None = None, backend: str | None = None
) -> dict[str, float]:
    """Flat dict of T features for one candidate ring, with a fixed key set.

    G is the window graph the neighbourhood is taken from (the caller decides whether
    that is the full or the sparsified graph, see `local_graph`); window_max is the max
    edge weight of the candidate's window and sets the infinite-bar cap under the default rule.
    """
    cfg = config or load_config()
    H, dgms = local_diagrams(G, candidate, cfg, backend)
    lim = cfg["local"]["ring_inf_limits"]
    ring_inf = ring_survives(H, candidate.edges, lim["max_edges"], lim["max_triangles"])
    if ring_inf is None:  # oversized neighbourhood: documented fallback = birth-match proxy
        log.warning("ring_inf fallback (birth-match): neighbourhood of %s has %d edges", candidate.nodes, H.number_of_edges())
        ring_inf = ring_has_infinite_bar(dgms[1], candidate.birth)
    return vectorize_diagrams(dgms, window_max, cfg, "T_local", ring_inf=ring_inf)


def window_topology_features(
    Gs: nx.DiGraph, window_max: float, config: dict | None = None, backend: str | None = None
) -> dict[str, float]:
    """Window-level T features (`T_win_*`): persistence of the whole *sparsified* window graph (§4.4).

    Computed once per window and shared by all of its candidates. window_max is the max edge
    weight of the full window graph, so the cap matches the local features.
    """
    cfg = config or load_config()
    dgms = directed_flag_persistence(Gs, backend=backend or cfg["local"]["backend"])
    return vectorize_diagrams(dgms, window_max, cfg, "T_win")
