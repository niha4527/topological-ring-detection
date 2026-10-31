"""Labelled B/D/T candidate table for one synthetic window (proposal v3 §3.7, §4).

One row per candidate: the keys, all feature columns of `features.full_feature_table`, the ring label
(`kind`, `pattern_id`, `y`; y is NaN for ambiguous candidates), the planted scenario of positives
(isolated / dense / chorded) and a group id. Groups are a union-find over candidates that share an edge,
merged with the planted structure id (eval/splits.assign_groups); `seed_offset` keeps seeds apart.
Labels and planted ids are attached AFTER the features, which never see them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ringtda.data.synthetic import SyntheticConfig, generate
from ringtda.eval.labels import label_candidates
from ringtda.eval.splits import assign_groups
from ringtda.features import full_feature_table


def synthetic_candidate_table(cfg: SyntheticConfig, tau_quantile: float | None = None, max_len: int = 6, group_offset: int = 0) -> pd.DataFrame:
    tx, planted = generate(cfg)
    feats, cands = full_feature_table([tx], tau_quantile=tau_quantile, max_len=max_len, return_candidates=True)
    c = cands[0]
    lab = label_candidates(c, tx, planted)
    scen = {p.pattern_id: p.scenario for p in planted}
    df = feats.copy()
    df["kind"] = lab["kind"].to_numpy()
    df["pattern_id"] = lab["pattern_id"].to_numpy()
    df["y"] = lab["y"].to_numpy()
    df["scenario"] = [scen[p] if p >= 0 and k == "wash_ring" else k for p, k in zip(df.pattern_id, df.kind)]
    # candidate order in `feats` is the order of `c` (candidate_id = position), so rows align
    assert (df.candidate_id.to_numpy() == np.arange(len(c))).all() and len(df) == len(c)
    g = assign_groups([x.edges for x in c], df.pattern_id.to_numpy())
    df["group"] = g + group_offset
    df["seed"] = cfg.seed
    return df
