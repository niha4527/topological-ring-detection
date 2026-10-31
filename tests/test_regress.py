import numpy as np
import pandas as pd
import pytest

from ringtda.data.synthetic import SyntheticConfig
from ringtda.models.regress import assert_no_excluded, grouped_folds, match_next, reg_metrics, ring_volume, severity_pairs

TINY = dict(n_accounts=80, n_background=300, ring_sizes=(3,), rings_per_size=1, n_transitive_decoys=0, dense_rings_per_size=0,
            n_chorded_rings=0, n_shortcut_filled=0, hard_negatives=False)


def test_ring_volume_counts_only_the_rings_wash_rows():
    tx = pd.DataFrame({"pattern_id": [3, 3, 3, 4, -1], "pattern": ["wash_ring", "wash_ring", "decoy", "wash_ring", "background"], "amount": [1.0, 2.0, 5.0, 7.0, 9.0]})
    assert ring_volume(tx, 3) == 3.0 and ring_volume(tx, 4) == 7.0 and ring_volume(tx, 9) == 0.0


def test_match_next_needs_same_accounts_and_same_ring():
    nxt = [(frozenset({1, 2, 3}), 5)]
    assert match_next((3, 1, 2), 5, nxt) and not match_next((1, 2), 5, nxt) and not match_next((1, 2, 3), 6, nxt)


def test_target_alignment_on_a_planted_ring():
    from ringtda.data.synthetic import generate

    cfg = SyntheticConfig(**TINY, seed=0)
    tx, planted = generate(cfg)
    pid = [p.pattern_id for p in planted if p.is_positive][0]
    df = severity_pairs(cfg, n_windows=7)
    assert len(df) > 0 and set(df.pattern_id) == {pid}
    for r in df.itertuples():  # the target is the ring's volume in the NEXT window, not the current one
        nxt = tx[(tx.timestamp >= (r.window_id + 1) * 86400) & (tx.timestamp < (r.window_id + 2) * 86400)]
        assert r.vol_next == pytest.approx(ring_volume(nxt, pid)) and r.y_next == pytest.approx(np.log1p(r.vol_next))


def test_leakage_guard_and_folds_and_metrics():
    with pytest.raises(AssertionError):
        assert_no_excluded(["D_birth", "D_log_volume"], ["D_log_volume"])
    assert_no_excluded(["D_birth"], ["D_log_volume"])
    groups = np.repeat(np.arange(20), 3)
    for fold in grouped_folds(groups, 5, 3, seed=1):
        for g in np.unique(groups):
            assert len(set(fold[groups == g])) == 1
    m = reg_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 5.0])
    assert m["rmse"] == pytest.approx(np.sqrt(4 / 3)) and m["mae"] == pytest.approx(2 / 3) and m["spearman"] == pytest.approx(1.0)
