import numpy as np

from ringtda.data.synthetic import SyntheticConfig
from ringtda.eval.recall import cycle_attempts, ibm_recall, synthetic_recall
from ringtda.synthetic_table import synthetic_candidate_table
import pandas as pd

TINY = dict(n_accounts=80, n_background=200, ring_sizes=(3,), rings_per_size=1, n_transitive_decoys=0, dense_rings_per_size=0,
            n_chorded_rings=0, n_shortcut_filled=1, hard_negatives=False)


def test_table_has_labels_scenarios_and_seed_prefixed_groups():
    a = synthetic_candidate_table(SyntheticConfig(**TINY, seed=0))
    b = synthetic_candidate_table(SyntheticConfig(**TINY, seed=1), group_offset=10_000)
    assert (a.kind == "wash_ring").any() and set(a.loc[a.kind == "wash_ring", "scenario"]) == {"isolated"}
    assert a.loc[a.y == 1, "group"].nunique() >= 1 and len(a) == a.group.size
    assert a.group.max() < 10_000 <= b.group.min()  # seeds never share a group id
    assert (a.loc[a.kind == "wash_ring", "y"] == 1).all()


def test_synthetic_recall_finds_the_planted_ring_at_a_loose_tau_only_if_candidates_contain_it():
    from ringtda.data.synthetic import generate
    from ringtda.features import window_graphs
    from ringtda.topology.candidates import candidate_rings

    tx, pl = generate(SyntheticConfig(**TINY, seed=0))
    _, G, _, tau = window_graphs(tx, 0.25)
    r = synthetic_recall(candidate_rings(G, tau=tau, max_len=6).candidates, tx, pl)
    assert len(r["rings"]) == 1 and r["rings"][0][2] is True
    assert synthetic_recall([], tx, pl)["rings"][0][2] is False


def test_ibm_recall_counts_complete_and_recovered_cycle_attempts():
    tx = pd.DataFrame({"timestamp": [0.5, 1.0, 1.5, 4.0, 9.0], "src": [0, 1, 2, 5, 6], "dst": [1, 2, 0, 6, 5],
                       "pattern_id": [0, 0, 0, 1, 1], "pattern": ["CYCLE"] * 5})
    att = cycle_attempts(tx)
    assert att.set_index("attempt_id").n_accounts.to_dict() == {0: 3, 1: 2}
    cands = pd.DataFrame({"window_id": [0], "nodes": [(2, 0, 1)], "kind": ["background"], "y": [0.0], "attempt": [-1]})
    r = ibm_recall(att, cands, {0: 0.0}, 5.0, max_len=6)
    assert (r["n_complete"], r["n_recovered"], r["n_recovered_labelled"], r["recall_of_complete"]) == (1, 1, 0, 1.0)  # attempt 1 ends on day 9: not complete
    assert np.isnan(ibm_recall(att, cands, {0: 20.0}, 5.0, 6)["recall_of_complete"])
