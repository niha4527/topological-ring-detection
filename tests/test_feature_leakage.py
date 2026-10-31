"""Leakage guard for the B / D / T feature sets (proposal v3 §3.4, §3.8)."""

import inspect

import numpy as np
import pytest

import ringtda.features as F
from ringtda.data.synthetic import SyntheticConfig, generate
from ringtda.graph import baseline_features as B
from ringtda.graph import direction_features as D
from ringtda.topology.vectorize import load_config

CFG = load_config()
SETS = F.feature_sets(CFG)
FORBIDDEN = ("label", "kind", "scenario", "attempt", "planted", "group", "window_id", "candidate_id", "pattern")
SMALL = dict(
    n_accounts=80, n_background=150, ring_sizes=(3,), rings_per_size=1, n_transitive_decoys=0, dense_rings_per_size=0,
    n_chorded_rings=0, n_shortcut_filled=1, hard_negatives=False,
)  # fmt: skip


def windows(*seeds, **kw):
    return [generate(SyntheticConfig(seed=s, **{**SMALL, **kw}))[0] for s in seeds]


def test_prefixes_are_disjoint_and_sets_are_explicit_unions():
    b, d, t = SETS["B"], [c for c in SETS["B+D"] if c.startswith("D_")], SETS["T"]
    assert all(c.startswith("B_") for c in b) and all(c.startswith("T_") for c in t)
    assert not (set(b) & set(d)) and not (set(b) & set(t)) and not (set(d) & set(t))
    assert SETS["B+D"] == b + d and SETS["B+D+T"] == b + d + t
    for cols in SETS.values():
        assert len(cols) == len(set(cols))


def test_no_label_like_or_key_columns_in_any_set():
    for cols in SETS.values():
        for c in cols:
            assert c.lower() not in ("y",) and not any(f in c.lower() for f in FORBIDDEN), c


def test_ring_birth_is_in_d_only():
    assert "D_birth" in SETS["B+D"] and "D_birth" not in SETS["B"] and "D_birth" not in SETS["T"]
    assert not any("birth" in c.lower() for c in SETS["T"] + SETS["B"])
    # T_local_ring_inf (is the ring's loop never filled) is allowed in T
    assert "T_local_ring_inf" in SETS["T"]


def test_no_legacy_report_columns():
    for cols in SETS.values():
        assert not [c for c in cols if c.startswith("topo_") or c in ("birth_eq", "length")]


def test_tables_have_fixed_width_no_nan_and_aligned_keys():
    small, other = windows(0, 1), windows(2)
    full = F.full_feature_table(small)
    assert list(full.columns) == F.KEYS + SETS["B+D+T"]
    assert list(F.full_feature_table(other).columns) == list(full.columns)  # width independent of the window
    assert np.isfinite(full[SETS["B+D+T"]].to_numpy()).all()
    keys = lambda df: set(map(tuple, df[F.KEYS].to_numpy()))  # noqa: E731
    t, bd = F.t_feature_table(small), F.bd_feature_table(small)
    assert keys(t) == keys(bd) == keys(full)
    assert list(full[F.KEYS].itertuples(index=False)) == list(t[F.KEYS].itertuples(index=False))
    # the joined T block equals the stand-alone T table
    assert np.allclose(full[SETS["T"]].to_numpy(), t[SETS["T"]].to_numpy())
    assert np.allclose(full[SETS["B+D"]].to_numpy(), bd[SETS["B+D"]].to_numpy())


def test_no_t_column_duplicates_the_ring_birth():
    df = F.full_feature_table(windows(0))
    birth = df["D_birth"].to_numpy()
    for c in SETS["T"] + SETS["B"]:
        col = df[c].to_numpy()
        assert not (np.ptp(col) > 0 and np.allclose(col, birth)), c


def test_feature_builders_take_no_labels_or_planted_structures():
    bad = ("planted", "label", "labels", "attempt", "attempts", "scenario", "y", "pattern")
    for fn in (D.direction_features, B.baseline_features, B.WindowBaseline.__init__, F.t_feature_table, F.bd_feature_table, F.full_feature_table):
        assert not set(inspect.signature(fn).parameters) & set(bad), fn.__qualname__


def test_features_do_not_change_when_label_columns_are_removed_from_the_transactions():
    tx, _ = generate(SyntheticConfig(seed=0, **SMALL))
    assert {"pattern_id", "pattern"} <= set(tx.columns)
    plain = tx[["timestamp", "src", "dst", "amount"]]
    a, b = F.full_feature_table([tx]), F.full_feature_table([plain])
    assert a.equals(b)


@pytest.mark.parametrize("kw", [{}, {"hard_negatives": True, "n_accounts": 400, "n_background": 300}])
def test_synthetic_tables_build_with_and_without_hard_negatives(kw):
    df = F.full_feature_table(windows(0, **kw))
    assert len(df) > 0 and df.shape[1] == len(F.KEYS) + len(SETS["B+D+T"])
