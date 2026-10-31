import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from ringtda.eval.metrics import GroupedBootstrap, average_precision, mean_ap_with_ci, paired_difference


def test_average_precision_known_example_and_matches_sklearn():
    y = [1, 0, 1, 0]
    s = [0.9, 0.8, 0.7, 0.1]
    assert average_precision(y, s) == pytest.approx(0.5 * 1.0 + 0.5 * (2 / 3))  # hand-computed
    rng = np.random.default_rng(0)
    y, s = rng.integers(0, 2, 200), rng.integers(0, 6, 200).astype(float)  # many tied scores
    assert average_precision(y, s) == pytest.approx(average_precision_score(y, s))
    w = rng.integers(0, 4, 200)
    assert average_precision(y, s, w) == pytest.approx(average_precision_score(y, s, sample_weight=w))


def test_bootstrap_ap_equals_weighted_ap_and_ci_contains_point_for_perfect_separation():
    groups = np.repeat(np.arange(30), 2)
    y = (groups < 10).astype(int)
    perfect = y.astype(float) + np.random.default_rng(1).normal(0, 0.01, len(y))
    gb = GroupedBootstrap(groups, n_boot=200, seed=3)
    boot = gb.ap(y, perfect)
    for k in range(gb.n_boot):  # every resample, including those that miss the top-ranked candidates
        w = gb.counts[k][np.searchsorted(gb.groups_, groups)]
        assert boot[k] == pytest.approx(average_precision(y, perfect, w), abs=1e-6)
    res = mean_ap_with_ci(y, [perfect, perfect], gb)
    assert res["ap"] == 1.0 and res["ci_low"] <= res["ap"] <= res["ci_high"]


def test_paired_difference_of_identical_scores_is_exactly_zero():
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(40), 3)
    y = (rng.random(len(groups)) < 0.2).astype(int)
    s = rng.normal(size=len(y))
    gb = GroupedBootstrap(groups, n_boot=100, seed=0)
    a, b = mean_ap_with_ci(y, [s], gb), mean_ap_with_ci(y, [s.copy()], gb)
    d = paired_difference(a, b)
    assert d["diff"] == 0.0 and d["ci_low"] == 0.0 and d["ci_high"] == 0.0


def test_resample_without_positives_is_nan_not_an_error():
    groups = np.arange(6)
    y = np.array([1, 0, 0, 0, 0, 0])
    gb = GroupedBootstrap(groups, n_boot=300, seed=0)
    out = gb.ap(y, np.arange(6.0))
    assert np.isnan(out).any() and np.isfinite(out[~np.isnan(out)]).all()


def test_zero_weight_head_is_not_nan():
    # the best-scored candidates have weight 0 (their groups are absent from the resample)
    y, s = [1, 1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6, 0.5]
    w = [0, 0, 1, 1, 1]
    assert average_precision(y, s, w) == pytest.approx(average_precision_score(y, s, sample_weight=w))
    gb = GroupedBootstrap(np.arange(5), n_boot=400, seed=1)
    out = gb.ap(np.array(y), np.array(s))
    ref = [average_precision(y, s, gb.counts[k]) for k in range(400)]
    nan_ref = np.isnan(ref)
    assert np.array_equal(np.isnan(out), nan_ref) and np.allclose(out[~nan_ref], np.array(ref)[~nan_ref], atol=1e-6)
    assert np.isnan(out).sum() < 400 * 0.2  # only resamples without a positive are undefined
