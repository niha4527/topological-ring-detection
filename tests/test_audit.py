"""Pattern-behaviour audit helpers on tiny hand-made tables (no IBM data needed)."""

import numpy as np
import pandas as pd
import pytest

from ringtda.eval.audit import add_balance_only_weight, auc, edge_membership, imputation_counts, ring_scores, summarize
from ringtda.graph.weights import edge_weights


def test_auc_perfect_identical_and_direction():
    assert auc([0.9, 0.8, 0.1, 0.2, 0.3], [1, 1, 0, 0, 0], [0, 0, 1, 1, 1]) == 1.0
    assert auc([1, 2, 1, 2], [1, 1, 0, 0], [0, 0, 1, 1]) == 0.5  # identical distributions
    assert auc([0.9, 0.8, 0.1, 0.2, 0.3], [1, 1, 0, 0, 0], [0, 0, 1, 1, 1], lower_is_suspicious=True) == 0.0
    assert auc([0.1, 0.2, 0.9, 0.8, 0.7], [1, 1, 0, 0, 0], [0, 0, 1, 1, 1], lower_is_suspicious=True) == 1.0
    assert np.isnan(auc([1, 2], [0, 0], [1, 1]))


def toy_pairs():
    # 10 pairs; weights tie at 1.0 for eight of them
    w = [1.0] * 8 + [0.5, 0.2]
    return pd.DataFrame(
        {"src": range(10), "dst": range(10, 20), "s_rec": 0.0, "s_bal": 0.5, "s_reg": 0.5, "n_s_rec": 0.0, "n_s_bal": [0.0] * 8 + [1.0, 1.0],
         "n_s_reg": 0.5, "weight": w, "gap_cv": [np.nan, np.nan] + [0.3] * 8}
    )  # fmt: skip


def test_tie_rate_and_decile_share_on_ten_pairs():
    p = add_balance_only_weight(toy_pairs())
    groups = {"A": np.array([True, False, False, False, False, False, False, False, True, False]), "background": np.array([True] * 8 + [False] * 2)}
    s = summarize(p, groups, "w").set_index(["group", "score"])
    r = s.loc[("A", "weight")]
    assert r.n_pairs == 2 and r.tie_window_median == 0.5 and r.tie_window_mode == 0.5  # pair 0 ties at 1.0, pair 8 (0.5) does not
    # q10 of the weights = 0.2 + 0.9 * 0.3 = 0.47: only the 0.2 pair is strictly below it
    assert r.decile_share == 0.0 and r.decile_share_all == 0.1
    b = summarize(p, {"A": np.array([False] * 8 + [True, True]), "background": np.array([True] * 8 + [False] * 2)}, "w").set_index(["group", "score"])
    assert b.loc[("A", "weight")].decile_share == 0.5
    assert b.loc[("A", "weight")].auc_vs_background == 1.0  # lower weight = more suspicious, perfectly separated
    assert b.loc[("A", "n_s_bal")].auc_vs_background == 1.0 and b.loc[("A", "n_s_rec")].auc_vs_background == 0.5  # all ties
    assert b.loc[("A", "weight_bal_only")].auc_vs_background == 1.0


def test_edge_membership_groups():
    tx = pd.DataFrame(
        {
            "src": [0, 1, 2, 3, 4, 5], "dst": [1, 2, 3, 4, 5, 6],
            "pattern": ["CYCLE", "CYCLE", "STACK", "unattributed", "background", "background"],
            "laundering": [True, True, True, True, False, False],
        }
    )  # fmt: skip
    pairs = pd.DataFrame({"src": [0, 1, 2, 3, 4, 5], "dst": [1, 2, 3, 4, 5, 6]})
    g = edge_membership(pairs, tx)
    assert set(g) == {"CYCLE", "STACK", "unattributed", "background"}
    assert g["CYCLE"].tolist() == [True, True, False, False, False, False]
    assert g["STACK"].tolist() == [False, False, True, False, False, False]
    assert g["unattributed"].tolist() == [False, False, False, True, False, False]
    assert g["background"].tolist() == [False, False, False, False, True, True]


def test_balance_only_weight_matches_edge_weights_variant():
    from ringtda.graph.windows import aggregate_pairs

    rng = np.random.default_rng(0)
    tx = pd.DataFrame({"timestamp": rng.uniform(0, 5, 80), "src": rng.integers(0, 8, 80), "dst": rng.integers(0, 8, 80), "amount": rng.uniform(1, 9, 80)})
    tx = tx[tx.src != tx.dst]
    p = edge_weights(aggregate_pairs(tx))
    assert np.allclose(add_balance_only_weight(p)["weight_bal_only"], edge_weights(aggregate_pairs(tx), scores=("s_bal",))["weight"])


def test_imputation_counts():
    p = toy_pairs()
    groups = {"A": np.array([True, True, False, False, False, False, False, False, False, False]), "background": ~np.array([True, True] + [False] * 8)}
    c = imputation_counts(p, groups, "w").set_index("group")
    assert c.loc["A", "n_imputed"] == 2 and c.loc["A", "share_imputed"] == 1.0 and c.loc["background", "n_imputed"] == 0


def test_ring_scores():
    p = toy_pairs()
    cands = pd.DataFrame({"edges": [((0, 10), (8, 18)), ((1, 11), (2, 12))], "kind": ["CYCLE", "background"], "y": [1.0, 0.0]})
    r = ring_scores(p, cands)
    assert r.birth.tolist() == [1.0, 1.0] and r.n_s_bal_mean.tolist() == [0.5, 0.0] and r.n_s_bal_min.tolist() == [0.0, 0.0]
