"""B features: graph-baseline (density / centrality) features of a candidate ring (proposal v3 §3.4).

Computed on the FULL window graph for the ring members and the same hops-neighbourhood the T features
use (configs/topology.yaml `local.hops`). T takes its neighbourhood from the sparsified graph by default,
so B sees at least as much of the graph as T; this is stated, not hidden.

Inputs are only the weighted window graph and a Candidate: no planted structures, labels, attempt ids
or scenario names.

Betweenness is unweighted and directed. Exact betweenness is infeasible on IBM windows (~400k nodes), so
windows with more than `exact_max_nodes` nodes use k-source sampled betweenness with a fixed seed,
computed once per window (`window_baseline`) and shared by all candidates (configs/features.yaml).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import yaml

from ringtda.topology.candidates import Candidate, neighborhood, sparsify

CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "features.yaml"
B_NAMES = [
    "B_deg_mean", "B_deg_max", "B_indeg_mean", "B_indeg_max", "B_outdeg_mean", "B_outdeg_max",
    "B_betw_mean", "B_betw_max", "B_clust_mean", "B_w_var", "B_density", "B_nbhd_size",
]  # fmt: skip


def load_config(path: str | Path | None = None) -> dict:
    with open(path or CONFIG_PATH) as f:
        return yaml.safe_load(f)


class WindowBaseline:
    """Per-window cache: betweenness (computed once) and lazily computed local clustering."""

    def __init__(self, G: nx.DiGraph, config: dict | None = None, cache_path: str | Path | None = None):
        """cache_path: optional file caching the betweenness of this window across runs (e.g. another
        tau_quantile on the same window). The cache stores the settings and the graph size and is only
        used if they match; the cached values are the ones the computation would produce."""
        bc = (config or load_config())["betweenness"]
        self.G = G
        self.Gu = G.to_undirected(as_view=True)
        self._clust: dict = {}
        stamp = {"bc": dict(bc), "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges()}
        if cache_path is not None and Path(cache_path).exists():
            with open(cache_path, "rb") as f:
                hit = pickle.load(f)
            if hit["stamp"] == stamp:
                self.betweenness, self.betweenness_exact, self.from_cache = hit["betweenness"], hit["exact"], True
                return
        self.from_cache = False
        BG = G
        if bc["graph"] == "sparsified":
            w = np.array([d["weight"] for _, _, d in G.edges(data=True)])
            BG = sparsify(G, float(np.quantile(w, bc["sparsify_quantile"]))) if len(w) else G
        exact = BG.number_of_nodes() <= bc["exact_max_nodes"]
        self.betweenness = nx.betweenness_centrality(BG, k=None if exact else bc["k"], seed=bc["seed"], normalized=True)
        self.betweenness_exact = exact
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(str(cache_path) + ".tmp")
            with open(tmp, "wb") as f:
                pickle.dump({"stamp": stamp, "betweenness": self.betweenness, "exact": exact}, f)
            tmp.replace(cache_path)

    def clustering(self, v) -> float:
        if v not in self._clust:
            self._clust[v] = nx.clustering(self.Gu, v)
        return self._clust[v]


def baseline_features(G: nx.DiGraph, candidate: Candidate, ctx: WindowBaseline, hops: int = 1) -> dict[str, float]:
    """Flat dict of B features for one candidate (fixed key set `B_NAMES`)."""
    nodes = candidate.nodes
    ind = np.array([G.in_degree(v) for v in nodes], dtype=float)
    outd = np.array([G.out_degree(v) for v in nodes], dtype=float)
    deg = ind + outd
    betw = np.array([ctx.betweenness.get(v, 0.0) for v in nodes], dtype=float)
    clust = np.array([ctx.clustering(v) for v in nodes], dtype=float)
    H = neighborhood(G, nodes, hops=hops)
    n = H.number_of_nodes()
    w = np.array([d["weight"] for _, _, d in H.edges(data=True)], dtype=float)
    return {
        "B_deg_mean": float(deg.mean()), "B_deg_max": float(deg.max()),
        "B_indeg_mean": float(ind.mean()), "B_indeg_max": float(ind.max()),
        "B_outdeg_mean": float(outd.mean()), "B_outdeg_max": float(outd.max()),
        "B_betw_mean": float(betw.mean()), "B_betw_max": float(betw.max()),
        "B_clust_mean": float(clust.mean()),
        "B_w_var": float(w.var()) if len(w) else 0.0,
        "B_density": float(H.number_of_edges() / (n * (n - 1))) if n > 1 else 0.0,
        "B_nbhd_size": float(n),
    }  # fmt: skip
