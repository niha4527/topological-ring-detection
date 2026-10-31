"""Groups and fold assignments on synthetic fixtures (README "Phase 3 design decisions")."""

import numpy as np
import pandas as pd

from ringtda.eval.splits import assign_groups, group_summary, stratified_group_cv, time_ordered_split, transfer_split


def ring(nodes):
    n = len(nodes)
    return tuple((nodes[i], nodes[(i + 1) % n]) for i in range(n))


def test_candidates_sharing_an_edge_share_a_group():
    e = [ring([0, 1, 2]), ring([0, 1, 3]), ring([5, 6, 7])]  # first two share the edge 0->1
    g = assign_groups(e, [-1, -1, -1])
    assert g[0] == g[1] and g[2] != g[0]


def test_negative_sharing_an_edge_with_a_positive_joins_its_group():
    e = [ring([0, 1, 2]), ring([1, 2, 9])]  # share 1->2
    g = assign_groups(e, [7, -1])  # first is a positive of attempt 7
    assert g[0] == g[1]


def test_same_attempt_merges_disjoint_candidates():
    e = [ring([0, 1, 2]), ring([10, 11, 12]), ring([20, 21, 22])]
    g = assign_groups(e, [3, 3, -1])
    assert g[0] == g[1] and g[2] != g[0]


def test_strict_variant_merges_candidates_sharing_an_account():
    e = [ring([0, 1, 2]), ring([0, 5, 6])]  # share account 0, no edge
    nodes = [(0, 1, 2), (0, 5, 6)]
    assert assign_groups(e, [-1, -1])[0] != assign_groups(e, [-1, -1])[1]
    s = assign_groups(e, [-1, -1], nodes, strict=True)
    assert s[0] == s[1]


def fixture(n_groups=40, per=3, pos_groups=12, seed=0):
    groups = np.repeat(np.arange(n_groups), per)
    y = np.zeros(len(groups))
    y[np.isin(groups, np.arange(pos_groups))] = 1
    y[np.arange(len(y)) % 7 == 3] = np.nan  # some ambiguous rows
    return groups, y


def test_no_group_spans_two_folds_and_every_fold_has_positives():
    groups, y = fixture()
    runs = stratified_group_cv(groups, y, k=5, repeats=5)
    assert len(runs) == 5
    for r in runs:
        fold = r["fold"]
        assert (fold >= 0).all()
        assert (pd.Series(fold).groupby(groups).nunique() == 1).all()
        assert r["k"] == 5 and r["fold_without_positive"] == 0
    assert len({tuple(r["fold"]) for r in runs}) > 1  # repeats differ


def test_fallback_to_three_folds_when_a_fold_would_have_no_positive():
    groups = np.repeat(np.arange(30), 2)
    y = np.zeros(len(groups))
    y[:4] = 1  # two positive groups only
    runs = stratified_group_cv(groups, y, k=5, fallback_k=3, repeats=2)
    assert all(r["k"] == 3 for r in runs)
    assert all((pd.Series(r["fold"]).groupby(groups).nunique() == 1).all() for r in runs)


def test_time_ordered_split_has_no_group_on_both_sides_and_drops_mixed_windows():
    # group 0 first seen day 1 (train), group 1 day 9 (test), group 2 day 6 (gap); windows 0-2
    df = pd.DataFrame(
        {
            "window_id": [0, 0, 1, 1, 2, 2, 3],
            "t_seen": [1.0, 1.5, 1.0, 6.0, 9.0, 9.5, 2.0],
        }
    )
    groups = np.array([0, 0, 0, 2, 1, 1, 0])
    out = time_ordered_split(df, groups, 5.0, 7.0)
    assert set(out.loc[out.group == 0, "side"]) <= {"train", "dropped"}
    assert set(out.loc[out.group == 1, "side"]) == {"test"}
    assert set(out.loc[out.group == 2, "side"]) == {"gap"}
    # no group appears on both sides
    sides = out[out.side.isin(["train", "test"])].groupby("group")["side"].nunique()
    assert (sides == 1).all()
    # a window holding both a train-side and a test-side candidate is dropped
    mixed = pd.DataFrame({"window_id": [0, 0], "t_seen": [1.0, 9.0]})
    m = time_ordered_split(mixed, np.array([0, 1]), 5.0, 7.0)
    assert set(m.side) == {"dropped"}


def test_transfer_counts_shared_accounts():
    a = pd.DataFrame({"nodes": [(0, 1, 2), (3, 4)]})
    b = pd.DataFrame({"nodes": [(0, 1), (2, 3)]})
    acc_a = np.array(["x_1", "x_2", "x_3", "x_4", "x_5"])
    acc_b = np.array(["x_3", "y_1", "y_2", "y_3"])  # x_3 shared: id 2 in a, id 0 in b
    r = transfer_split(a, b, acc_a, acc_b)
    assert r == {"n_shared_accounts": 1, "train_candidates_using_shared": 1, "test_candidates_using_shared": 1}


def test_group_summary():
    s = group_summary(np.array([0, 0, 1, 2]), np.array([1.0, 0, np.nan, 0]))
    assert s == {"n_groups": 3, "largest_group_share": 0.5, "positive_groups": 1}
