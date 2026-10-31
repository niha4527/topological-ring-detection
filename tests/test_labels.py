import numpy as np
import pandas as pd

from ringtda.data.synthetic import Planted
from ringtda.eval.labels import edge_backing, label_candidates, label_ring
from ringtda.topology.candidates import Candidate


def _tx(rows):
    return pd.DataFrame(rows, columns=["timestamp", "src", "dst", "amount", "pattern_id", "pattern"])


def _cand(nodes):
    return Candidate(tuple(nodes), tuple((nodes[i], nodes[(i + 1) % len(nodes)]) for i in range(len(nodes))), 0.1)


# structure 0: wash 3-ring 0->1->2->0; structure 1: market maker hub 10 with clients 11, 12
TX = _tx(
    [
        (0, 0, 1, 1.0, 0, "wash_ring"), (1, 1, 2, 1.0, 0, "wash_ring"), (2, 2, 0, 1.0, 0, "wash_ring"),
        (3, 10, 11, 1.0, 1, "market_maker"), (4, 11, 10, 1.0, 1, "market_maker"),
        (5, 2, 1, 1.0, -1, "background"),  # background on the reverse of a ring edge
        (6, 5, 6, 1.0, -1, "background"), (7, 6, 5, 1.0, -1, "background"),
        (8, 1, 0, 1.0, -1, "decoy"),  # decoy and context rows back nothing
        (9, 0, 2, 1.0, 0, "context"),
    ]
)
PLANTED = [Planted(0, "wash_ring", (0, 1, 2)), Planted(1, "market_maker", (10, 11, 12))]


def test_backing_ignores_background_decoy_and_context():
    b = edge_backing(TX)
    assert b[(0, 1)] == {0} and b[(10, 11)] == {1}
    assert (2, 1) not in b and (1, 0) not in b and (0, 2) not in b


def test_label_kinds():
    df = label_candidates([_cand([0, 1, 2]), _cand([10, 11]), _cand([5, 6]), _cand([1, 2])], TX, PLANTED)
    assert df["kind"].tolist() == ["wash_ring", "market_maker", "background", "ambiguous"]
    assert df["y"].iloc[:3].tolist() == [1.0, 0.0, 0.0] and np.isnan(df["y"].iloc[3])
    assert df["pattern_id"].tolist() == [0, 1, -1, -1]


def test_threshold_x():
    backing = {(0, 1): {7}, (1, 2): {7}, (2, 3): {7}, (3, 4): {7}}  # 4 of 5 edges backed by one structure
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    assert label_ring(edges, backing, {7: "wash_ring"}, x=0.8) == ("wash_ring", 7)
    assert label_ring(edges, backing, {7: "wash_ring"}, x=1.0) == ("ambiguous", -1)


def test_split_between_two_structures_is_ambiguous():
    backing = {(0, 1): {1}, (1, 0): {2}}
    assert label_ring([(0, 1), (1, 0)], backing, {1: "wash_ring", 2: "market_maker"})[0] == "ambiguous"
