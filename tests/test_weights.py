import numpy as np
import pandas as pd

from ringtda.graph.weights import compute_scores, edge_weights, rank_normalize
from ringtda.graph.windows import aggregate_pairs, iter_windows


def _tx(rows):
    return pd.DataFrame(rows, columns=["timestamp", "src", "dst", "amount"])


def test_reciprocal_pair_scores_one():
    tx = _tx([(0, 1, 2, 100.0), (10, 2, 1, 100.0)])
    s = compute_scores(aggregate_pairs(tx)).set_index(["src", "dst"])
    assert np.isclose(s.loc[(1, 2), "s_rec"], 1.0)


def test_one_way_edge_has_zero_reciprocity():
    tx = _tx([(0, 1, 2, 100.0)])
    s = compute_scores(aggregate_pairs(tx))
    assert np.isclose(s["s_rec"].iloc[0], 0.0)


def test_ring_nodes_are_balanced_even_without_reciprocity():
    # A->B->C->A, equal amounts: no pair is reciprocal, every node nets to zero
    tx = _tx([(0, "A", "B", 50.0), (1, "B", "C", 50.0), (2, "C", "A", 50.0)])
    s = compute_scores(aggregate_pairs(tx))
    assert np.allclose(s["s_rec"], 0.0)
    assert np.allclose(s["s_bal"], 1.0)


def test_few_transactions_get_median_regularity():
    rows = [(t, 1, 2, 10.0) for t in (0, 10, 20, 30)]  # perfectly regular, CV 0
    rows += [(5, 3, 4, 10.0), (500, 3, 4, 10.0)]  # only 1 gap -> undefined
    rows += [(t, 5, 6, 10.0) for t in (0, 1, 50, 51)]  # irregular
    s = compute_scores(aggregate_pairs(_tx(rows))).set_index(["src", "dst"])
    assert np.isclose(s.loc[(1, 2), "s_reg"], 1.0)
    known = s.loc[[(1, 2), (5, 6)], "s_reg"]
    assert np.isclose(s.loc[(3, 4), "s_reg"], known.median())


def test_rank_normalize_ties_at_bottom_map_to_zero():
    r = rank_normalize(pd.Series([0, 0, 0, 0, 0.5, 1.0]))
    assert np.allclose(r.iloc[:4], 0.0)
    assert np.isclose(r.iloc[-1], 1.0)


def test_weight_in_unit_interval_and_lower_is_more_suspicious():
    tx = _tx([(0, "A", "B", 50.0), (1, "B", "A", 50.0), (2, "X", "Y", 999.0)])
    w = edge_weights(aggregate_pairs(tx)).set_index(["src", "dst"])["weight"]
    assert ((w >= 0) & (w <= 1)).all()
    assert w[("A", "B")] < w[("X", "Y")]


def test_windows_cover_range():
    tx = _tx([(t, 1, 2, 1.0) for t in range(100)])
    wins = list(iter_windows(tx, length=30, stride=10))
    assert wins[0][0] == 0 and len(wins[0][2]) == 30
    assert [w[0] for w in wins] == list(range(0, 100, 10))
    assert all(len(w[2]) > 0 for w in wins)
