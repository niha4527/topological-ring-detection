import numpy as np
import pytest

from ringtda.eval.splits import stratified_group_cv
from ringtda.models.classify import MODELS, assert_group_disjoint, load_config, make_model, named_feature_sets, oof_scores

FORBIDDEN = ("label", "kind", "scenario", "attempt", "planted", "group", "window_id", "candidate_id", "pattern")


def toy(n_groups=40, per=4, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), per)
    y = (groups % 5 == 0).astype(int)
    X = rng.normal(size=(len(y), 5)) + y[:, None] * 1.5
    return X, y, groups


def test_no_group_in_both_train_and_test_of_any_fold():
    X, y, groups = toy()
    for run in stratified_group_cv(groups, y, k=5, repeats=3, seed=1):
        assert_group_disjoint(groups, run["fold"])
    bad = np.arange(len(y)) % 5
    with pytest.raises(AssertionError):
        assert_group_disjoint(groups, bad)  # groups are spread over folds


def test_scaler_is_fitted_on_the_training_part_only():
    X, y, _ = toy()
    tr, te = slice(0, 100), slice(100, None)
    m = make_model("logreg")
    m.fit(X[tr], y[tr])
    assert np.allclose(m.named_steps["scale"].mean_, X[tr].mean(axis=0))
    assert not np.allclose(m.named_steps["scale"].mean_, X[te].mean(axis=0))


@pytest.mark.parametrize("name", MODELS)
def test_oof_scores_cover_every_row_and_separate_the_toy_classes(name):
    from ringtda.eval.metrics import average_precision

    X, y, groups = toy()
    fold = stratified_group_cv(groups, y, k=4, repeats=1, seed=0)[0]["fold"]
    cfg = load_config()
    cfg["models"]["rf"]["n_estimators"] = 30
    s = oof_scores(X, y, groups, fold, name, cfg)
    assert np.isfinite(s).all() and average_precision(y, s) > 0.8


def test_feature_sets_have_no_label_or_key_columns():
    sets = named_feature_sets()
    assert set(sets) == {"B", "B+D", "B+D+T", "T", "D", "T_local", "T_win"}
    for cols in sets.values():
        assert cols and not any(f in c.lower() for c in cols for f in FORBIDDEN) and "y" not in cols
    assert set(sets["T_local"]) | set(sets["T_win"]) == set(sets["T"])
    assert "D_birth" in sets["D"] and "D_birth" not in sets["T"]
