"""Engineering gate (proposal v3 §3.4, §4.4): the T feature table is well-formed, window
features are computed once per window, and known topology shows up in the infinite-bar counts.
Correctness only; nothing here asserts that T helps."""

import numpy as np
import pytest

import ringtda.features as F
from ringtda.data.synthetic import SyntheticConfig, generate
from ringtda.eval.labels import label_candidates
from ringtda.topology.persistence import available_backends
from ringtda.topology.vectorize import feature_names, load_config, local_diagrams

CFG = load_config()
# one planted structure on a sparse background (a lone ring has degenerate rank-normalised weights)
ONLY = dict(
    n_accounts=80, n_background=150, ring_sizes=(), rings_per_size=0, n_transitive_decoys=0, dense_rings_per_size=0,
    n_chorded_rings=0, n_shortcut_filled=0, hard_negatives=False,
)


def window(**kw):
    return generate(SyntheticConfig(**{**ONLY, **kw}))


def inf_count(tx, planted, kind):
    df, cands = F.t_feature_table([tx], return_candidates=True)
    lab = label_candidates(cands[0], tx, planted)
    rows = df[(lab["kind"] == kind).to_numpy()]
    assert len(rows), f"no {kind} candidate"
    return rows["T_local_inf_h1"].to_numpy()


def test_default_local_graph_is_sparsified():
    assert CFG["local"]["local_graph"] == "sparsified"


def test_isolated_ring_has_infinite_h1():
    for seed in range(3):
        tx, planted = window(ring_sizes=(3,), rings_per_size=1, seed=seed)
        assert (inf_count(tx, planted, "wash_ring") >= 1).all()


def test_shortcut_filled_ring_has_no_infinite_h1_under_default():
    for seed in range(3):
        tx, planted = window(n_shortcut_filled=1, seed=seed)
        assert (inf_count(tx, planted, "shortcut_filled") == 0).all()


def test_transitive_decoy_is_never_a_candidate_ring():
    # a transitive triangle contains no directed cycle
    tx, planted = window(n_transitive_decoys=1)
    _, cands = F.t_feature_table([tx], return_candidates=True)
    assert "transitive_decoy" not in set(label_candidates(cands[0], tx, planted)["kind"])


def test_table_shape_keys_and_no_nan():
    wins = [generate(SyntheticConfig(seed=s, **{**ONLY, "ring_sizes": (3,), "rings_per_size": 1, "n_shortcut_filled": 1, "n_background": 150}))[0] for s in range(2)]
    df = F.t_feature_table(wins)
    assert list(df.columns) == ["window_id", "candidate_id", *feature_names(CFG, "T_local"), *feature_names(CFG, "T_win")]
    assert set(df.window_id) == {0, 1}
    assert not df.duplicated(["window_id", "candidate_id"]).any()
    assert np.isfinite(df.drop(columns=["window_id", "candidate_id"]).to_numpy()).all()
    # window features are constant within a window
    assert (df.groupby("window_id")[[c for c in df if c.startswith("T_win_")]].nunique() == 1).all().all()
    # T only: no D / B columns
    assert not [c for c in df if c.startswith(("birth", "length", "mean_", "min_", "deg", "topo_"))]


def test_window_features_computed_once_per_window(monkeypatch):
    calls = []
    real = F.window_topology_features
    monkeypatch.setattr(F, "window_topology_features", lambda *a, **k: calls.append(1) or real(*a, **k))
    wins = [generate(SyntheticConfig(seed=s, **{**ONLY, "ring_sizes": (3,), "rings_per_size": 2, "n_background": 150}))[0] for s in range(3)]
    df = F.t_feature_table(wins)
    assert len(df) > 3 and len(calls) == 3


@pytest.mark.skipif(len(available_backends()) < 2, reason="needs both backends")
def test_backends_agree_on_table():
    tx, _ = window(ring_sizes=(3,), rings_per_size=1, n_shortcut_filled=1, n_background=100)
    a = F.t_feature_table([tx], backend="flagser")
    b = F.t_feature_table([tx], backend="reference")
    assert a.shape == b.shape and len(a) >= 2
    assert np.allclose(a.to_numpy(float), b.to_numpy(float), atol=1e-5)  # flagser is float32


def test_window_candidates_interface_unchanged():
    tx, planted = window(ring_sizes=(3,), rings_per_size=1)
    df, Z = F.window_candidates(tx, planted)
    assert {"kind", "y", "birth_eq", "topo_h1_count", "topo_total_pers", "topo_max_pers", "topo_n_infinite", "topo_nbhd_edges"} <= set(df.columns)
    assert len(Z) == len(df)


def test_ring_inf_in_table():
    # seed 0: the ring's loop is homologous to an earlier reciprocal-pair loop (its infinite bar is
    # born before the ring birth); the exact test still finds it
    for seed in range(3):
        tx, planted = window(ring_sizes=(3,), rings_per_size=1, seed=seed)
        df, cands = F.t_feature_table([tx], return_candidates=True)
        kinds = label_candidates(cands[0], tx, planted)["kind"].to_numpy()
        assert (df.T_local_ring_inf[kinds == "wash_ring"] == 1.0).all()
        tx, planted = window(n_shortcut_filled=1, seed=seed)
        df, cands = F.t_feature_table([tx], return_candidates=True)
        kinds = label_candidates(cands[0], tx, planted)["kind"].to_numpy()
        assert (df.T_local_ring_inf[kinds == "shortcut_filled"] == 0.0).all()


def test_tau_quantile_changes_candidate_count():
    tx, _ = window(n_accounts=60, n_background=400, ring_sizes=(3,), rings_per_size=1)
    n = {q: len(F.t_feature_table([tx], tau_quantile=q, max_len=4)) for q in (0.25, 0.5)}
    assert n[0.25] < n[0.5]
    assert len(F.t_feature_table([tx], max_len=4)) == n[0.25]  # default comes from the config


def test_max_len_default_is_six():
    import inspect

    assert inspect.signature(F.t_feature_table).parameters["max_len"].default == 6
    assert CFG["candidates"]["max_len"] == 6


def test_ring_inf_exact_seed0_case_the_birth_match_missed():
    from ringtda.topology.vectorize import ring_has_infinite_bar

    tx, planted = window(seed=0, ring_sizes=(3,), rings_per_size=1)
    df, cands = F.t_feature_table([tx], return_candidates=True)
    m = (label_candidates(cands[0], tx, planted)["kind"] == "wash_ring").to_numpy()
    assert (df.T_local_ring_inf[m] == 1.0).all()
    # the old birth-match proxy gave 0 here
    _, G, Gs, _ = F.window_graphs(tx)
    c = [c for c, k in zip(cands[0], m) if k][0]
    _, dgms = local_diagrams(Gs, c)
    assert not ring_has_infinite_bar(dgms[1], c.birth)


def test_ring_inf_zero_for_shortcut_ring_in_dense_background_even_if_other_loops_survive():
    tx, planted = generate(SyntheticConfig(seed=2, hard_negatives=False))
    df, cands = F.t_feature_table([tx], max_len=5, return_candidates=True)
    m = (label_candidates(cands[0], tx, planted)["kind"] == "shortcut_filled").to_numpy()
    assert m.any() and (df.T_local_inf_h1[m] > 0).any()  # other infinite loops are in the neighbourhood
    assert (df.T_local_ring_inf[m] == 0.0).all()


@pytest.mark.skipif(len(available_backends()) < 2, reason="needs both backends")
def test_ring_inf_backend_independent():
    tx, _ = window(ring_sizes=(3,), rings_per_size=1, n_shortcut_filled=1, n_background=100)
    a = F.t_feature_table([tx], backend="flagser").T_local_ring_inf.to_numpy()
    b = F.t_feature_table([tx], backend="reference").T_local_ring_inf.to_numpy()
    assert len(a) >= 2 and (a == b).all()
