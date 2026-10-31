"""Classifiers for the B / D / T feature sets (proposal v3 §3.5): scikit-learn only, hyperparameters fixed
in configs/classify.yaml (no search). Scaling is part of the pipeline, so it is fitted inside each training
fold. Scores are decision functions (logistic regression, SVM) or class-1 probabilities (random forest)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ringtda.features import feature_sets

CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "classify.yaml"
MODELS = ("logreg", "rf", "svm")


def load_config(path: str | Path | None = None) -> dict:
    with open(path or CONFIG_PATH) as f:
        return yaml.safe_load(f)


def named_feature_sets(config: dict | None = None) -> dict[str, list[str]]:
    """B, B+D, B+D+T, T plus the controls D, T_local, T_win (T_win is constant within a window)."""
    s = feature_sets(config)
    T = s["T"]
    return {**s, "D": [c for c in s["B+D"] if c.startswith("D_")], "T_local": [c for c in T if c.startswith("T_local_")], "T_win": [c for c in T if c.startswith("T_win_")]}


def make_model(name: str, cfg: dict | None = None, seed: int = 0) -> Pipeline:
    cfg = cfg or load_config()
    p = cfg["models"][name]
    if name == "logreg":
        clf = LogisticRegression(C=p["C"], max_iter=p["max_iter"], class_weight="balanced")
    elif name == "rf":
        clf = RandomForestClassifier(n_estimators=p["n_estimators"], class_weight="balanced_subsample", random_state=seed, n_jobs=p["n_jobs"])
    elif name == "svm":
        clf = SVC(kernel="rbf", C=p["C"], gamma=p["gamma"], class_weight="balanced")
    else:
        raise ValueError(f"unknown model {name!r}")
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def scores(model: Pipeline, X) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


def assert_group_disjoint(groups: np.ndarray, fold: np.ndarray) -> None:
    """No group may appear in more than one fold (hence never in both train and test)."""
    for f in np.unique(fold):
        if set(groups[fold == f]) & set(groups[fold != f]):
            raise AssertionError(f"a group appears in fold {f} and in another fold")


def oof_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray, fold: np.ndarray, name: str, cfg: dict | None = None, seed: int = 0) -> np.ndarray:
    """Out-of-fold scores of one repeat: each fold is scored by a model fitted on the other folds only."""
    assert_group_disjoint(groups, fold)
    out = np.full(len(y), np.nan)
    for f in np.unique(fold):
        te = fold == f
        m = make_model(name, cfg, seed)
        m.fit(X[~te], y[~te])
        out[te] = scores(m, X[te])
    return out
