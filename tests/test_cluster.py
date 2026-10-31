import numpy as np
import pytest

from ringtda.models.cluster import best_k, grouped_bootstrap, majority_purity, metrics, permutation_baseline, purity, standardize, ward_labels


def two_types(seed=0, n=20):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(0, 0.1, (n, 3)), rng.normal(5, 0.1, (n, 3))])
    truth = np.array([0] * n + [1] * n)
    groups = np.repeat(np.arange(2 * n // 4), 4)
    return X, truth, groups


def test_perfect_separation_gives_ari_one_and_purity_one():
    X, truth, groups = two_types()
    Z = standardize(X)
    lab = ward_labels(Z, 2)
    m = metrics(Z, lab, truth, groups)
    assert m["ari_type"] == pytest.approx(1.0) and m["purity_type"] == 1.0 and m["silhouette_clusters"] > 0.8
    assert best_k(Z, (2, 5))[0] == 2


def test_identical_data_gives_ari_near_zero():
    rng = np.random.default_rng(1)
    X = np.ones((40, 3))
    truth = rng.integers(0, 2, 40)
    lab = ward_labels(standardize(X + rng.normal(0, 1e-9, X.shape)), 2)  # no structure: arbitrary split
    assert abs(metrics(standardize(X), lab, truth)["ari_type"]) < 0.2


def test_permutation_baseline_is_near_chance_and_majority_purity_is_the_largest_share():
    X, truth, groups = two_types()
    lab = ward_labels(standardize(X), 2)
    b = permutation_baseline(lab, truth, groups, n=200, seed=0)
    assert abs(b["ari_type_random"]) < 0.1 and b["purity_majority"] == 0.5 and purity(np.zeros(40, int), truth) == majority_purity(truth)


def test_determinism_and_constant_columns():
    X, truth, groups = two_types()
    X = np.hstack([X, np.ones((len(X), 1))])
    Z = standardize(X)
    assert np.isfinite(Z).all() and (Z[:, -1] == 0).all()
    assert (ward_labels(Z, 2) == ward_labels(Z, 2)).all()
    a = grouped_bootstrap(Z, truth, groups, 2, n_boot=20, seed=3)
    b = grouped_bootstrap(Z, truth, groups, 2, n_boot=20, seed=3)
    assert a == b and len(a) > 0
