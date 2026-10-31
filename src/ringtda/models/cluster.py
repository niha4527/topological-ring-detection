"""Ward clustering of positive rings against pattern types (proposal v3 §3.5). Exploratory; nothing here is a gate.

Vectors are standardized on the population (constant columns become 0). Ward linkage, Euclidean. Metrics against a
reference labelling: adjusted Rand index, purity, silhouette. Baselines: the mean over random permutations of the
cluster labels (same cluster sizes) and the majority-class purity. The grouped bootstrap resamples whole groups
(rings that share an edge or an attempt are not independent), reclusters each resample and recomputes the metrics.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score


def standardize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    sd = X.std(axis=0)
    return (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)


def ward_labels(Z: np.ndarray, k: int) -> np.ndarray:
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)


def purity(clusters, truth) -> float:
    clusters, truth = np.asarray(clusters), np.asarray(truth)
    return float(sum(np.bincount(np.unique(truth[clusters == c], return_inverse=True)[1]).max() for c in np.unique(clusters)) / len(truth))


def majority_purity(truth) -> float:
    return float(np.bincount(np.unique(truth, return_inverse=True)[1]).max() / len(truth))


def sil(Z, labels) -> float:
    labels = np.asarray(labels)
    n = len(np.unique(labels))
    return float(silhouette_score(Z, labels)) if 2 <= n < len(labels) else float("nan")


def best_k(Z: np.ndarray, k_range=(2, 8)) -> tuple[int, np.ndarray]:
    """k in k_range with the best silhouette of the Ward partition (smallest k on ties)."""
    ks = [k for k in range(k_range[0], k_range[1] + 1) if k < len(Z)]
    s = [sil(Z, ward_labels(Z, k)) for k in ks]
    return ks[int(np.nanargmax(s))], ward_labels(Z, ks[int(np.nanargmax(s))])


def metrics(Z, clusters, truth, groups=None) -> dict:
    out = {"ari_type": adjusted_rand_score(truth, clusters), "purity_type": purity(clusters, truth), "silhouette_clusters": sil(Z, clusters), "silhouette_types": sil(Z, truth)}
    if groups is not None:
        out["ari_group"] = adjusted_rand_score(groups, clusters)
    return out


def permutation_baseline(clusters, truth, groups=None, n: int = 1000, seed: int = 0) -> dict:
    """Mean ARI / purity of random labellings with the same cluster sizes (cluster labels permuted over the rings)."""
    rng = np.random.default_rng(seed)
    clusters = np.asarray(clusters)
    a, p, g = [], [], []
    for _ in range(n):
        c = rng.permutation(clusters)
        a.append(adjusted_rand_score(truth, c))
        p.append(purity(c, truth))
        if groups is not None:
            g.append(adjusted_rand_score(groups, c))
    out = {"ari_type_random": float(np.mean(a)), "purity_type_random": float(np.mean(p)), "purity_majority": majority_purity(truth)}
    if groups is not None:
        out["ari_group_random"] = float(np.mean(g))
    return out


def grouped_bootstrap(Z, truth, groups, k, n_boot=500, seed=0, choose_k=None) -> list[dict]:
    """Resample groups with replacement, recluster (fixed k, or best silhouette over choose_k), metrics per resample.
    Resamples with fewer than 2 pattern types or fewer rings than k are skipped (counted by the caller via len)."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_of = {g: np.flatnonzero(groups == g) for g in ug}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([idx_of[g] for g in pick])
        if len(np.unique(truth[idx])) < 2 or len(idx) <= (choose_k[0] if choose_k else k):
            continue
        Zb = Z[idx]
        if choose_k:
            kb, lab = best_k(Zb, choose_k)
        else:
            kb, lab = k, ward_labels(Zb, k)
        m = metrics(Zb, lab, truth[idx], groups[idx])
        m["k"] = kb
        out.append(m)
    return out
