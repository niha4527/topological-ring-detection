"""Hand-checked D features (direction_features.py)."""

import numpy as np
import networkx as nx
import pytest

from ringtda.graph.direction_features import D_NAMES, direction_features, gap_cv_fill
from ringtda.topology.candidates import Candidate


def graph(edges):
    """edges: (u, v, weight, volume, count, gap_cv, n_s_rec, n_s_bal, n_s_reg)"""
    G = nx.DiGraph()
    for u, v, w, vol, cnt, gap, rec, bal, reg in edges:
        G.add_edge(u, v, weight=w, volume=vol, count=cnt, gap_cv=gap, n_s_rec=rec, n_s_bal=bal, n_s_reg=reg)
    return G


def ring(nodes, G):
    n = len(nodes)
    es = tuple((nodes[i], nodes[(i + 1) % n]) for i in range(n))
    return Candidate(tuple(nodes), es, max(G.edges[e]["weight"] for e in es))


def test_cyclic_triangle_with_an_extra_inbound_edge():
    G = graph(
        [
            (0, 1, 0.1, 10, 4, 0.2, 0.0, 0.9, 0.8),
            (1, 2, 0.2, 10, 4, 0.2, 0.0, 0.7, 0.6),
            (2, 0, 0.3, 10, 4, 0.2, 0.0, 0.5, 0.4),
            (3, 0, 0.9, 5, 1, np.nan, 0.0, 0.0, 0.5),  # node 0 now has in=2, out=1
        ]
    )
    f = direction_features(G, ring([0, 1, 2], G))
    assert list(f) == D_NAMES
    assert f["D_length"] == 3
    assert f["D_birth"] == pytest.approx(0.3)  # max edge weight on the ring
    assert f["D_mean_w"] == pytest.approx(0.2) and f["D_min_w"] == pytest.approx(0.1)
    assert f["D_bal_mean"] == pytest.approx(0.7) and f["D_bal_min"] == pytest.approx(0.5)
    assert f["D_reg_mean"] == pytest.approx(0.6) and f["D_reg_min"] == pytest.approx(0.4)
    assert f["D_rec_mean"] == 0.0
    # node 0: |2-1|/3 = 1/3; nodes 1, 2 balanced
    assert f["D_deg_imb_mean"] == pytest.approx(1 / 9) and f["D_deg_imb_max"] == pytest.approx(1 / 3)
    assert f["D_vol_ratio"] == pytest.approx(1.0) and f["D_vol_cv"] == pytest.approx(0.0)
    assert f["D_log_volume"] == pytest.approx(np.log1p(30))
    assert f["D_gap_cv_mean"] == pytest.approx(0.2) and f["D_log_count_mean"] == pytest.approx(np.log(4))


def test_reciprocal_pair_volume_ratio_and_gap_fallback():
    G = graph(
        [
            (0, 1, 0.2, 10, 2, np.nan, 1.0, 1.0, 0.5),  # gap_cv undefined (< 3 transactions)
            (1, 0, 0.3, 40, 8, 0.5, 1.0, 1.0, 0.5),
        ]
    )
    fill = gap_cv_fill(G)
    assert fill == pytest.approx(0.5)  # median of the defined values
    f = direction_features(G, ring([0, 1], G), fill=0.3)
    assert f["D_length"] == 2 and f["D_birth"] == pytest.approx(0.3)
    assert f["D_vol_ratio"] == pytest.approx(0.25)  # 10 / 40
    assert f["D_vol_cv"] == pytest.approx(15 / 25)  # std 15, mean 25
    assert f["D_gap_cv_mean"] == pytest.approx((0.3 + 0.5) / 2)  # NaN replaced by the fill
    assert f["D_log_count_mean"] == pytest.approx(np.log(4))  # (ln 2 + ln 8) / 2
    assert f["D_deg_imb_mean"] == 0.0 and f["D_deg_imb_max"] == 0.0
    assert np.isfinite(list(f.values())).all()


def test_gap_fill_without_any_defined_gap_is_zero_and_zero_volume_is_safe():
    G = graph([(0, 1, 0.1, 0.0, 1, np.nan, 0, 0, 0.5), (1, 0, 0.1, 0.0, 1, np.nan, 0, 0, 0.5)])
    assert gap_cv_fill(G) == 0.0
    f = direction_features(G, ring([0, 1], G))
    assert f["D_vol_ratio"] == 1.0 and f["D_vol_cv"] == 0.0 and f["D_gap_cv_mean"] == 0.0
    assert np.isfinite(list(f.values())).all()
