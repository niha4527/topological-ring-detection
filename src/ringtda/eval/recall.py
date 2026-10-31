"""Candidate-generation recall (proposal v3 §3.3, §4): how many ground-truth rings the candidate step returns.

IBM AML: a CYCLE attempt is *complete* in a window if all its transactions fall inside it; it is *recovered*
if a candidate of that window has exactly the attempt's account set (the README measurement); the §3.7 label of
that candidate is reported separately. Synthetic: a planted wash ring is recovered if some candidate is labelled as that planted ring.
Recall needs no features.
"""

from __future__ import annotations

import pandas as pd


def cycle_attempts(tx: pd.DataFrame) -> pd.DataFrame:
    """One row per CYCLE attempt: attempt_id, n_accounts, accounts (frozenset), start_day, end_day."""
    t = tx[(tx["pattern_id"] >= 0) & (tx["pattern"] == "CYCLE")]
    g = t.groupby("pattern_id")
    acc = t.melt(id_vars="pattern_id", value_vars=["src", "dst"]).groupby("pattern_id")["value"].agg(lambda v: frozenset(v.tolist()))
    out = pd.DataFrame({"accounts": acc, "start_day": g["timestamp"].min(), "end_day": g["timestamp"].max()})
    out["n_accounts"] = out["accounts"].map(len)
    return out.reset_index(names="attempt_id")


def ibm_recall(att: pd.DataFrame, cands: pd.DataFrame, window_starts: dict[int, float], length_days: float, max_len: int, window_ids=None) -> dict:
    """Recall of complete CYCLE attempts over the given windows (None = all) for one candidate table (one tau).

    complete = attempts with all rows inside a window; recovered = a candidate of that window whose account set
    equals the attempt's (the README measurement: 8 of the 11 reachable attempts in days 0-5). n_recovered_labelled
    counts only those whose candidate also carries a positive CYCLE label for the attempt under the §3.7 rule.
    Recall is reported against all complete attempts and against those with <= max_len accounts (the reachable ones)."""
    wids = list(window_starts) if window_ids is None else list(window_ids)
    complete, rec, rec_lab = set(), set(), set()
    for w in wids:
        s = window_starts[w]
        here = att[(att.start_day >= s) & (att.end_day < s + length_days)]
        complete |= set(here.attempt_id)
        c = cands[cands.window_id == w]
        if not len(here) or not len(c):
            continue
        by_nodes = {frozenset(n) for n in c["nodes"]}
        rec |= set(here.loc[here.accounts.isin(by_nodes), "attempt_id"])
        rec_lab |= set(c.loc[(c.kind == "CYCLE") & (c.y == 1), "attempt"].astype(int)) & set(here.attempt_id)
    small = set(att.loc[att.n_accounts <= max_len, "attempt_id"]) & complete
    n_cand = int(cands.window_id.isin(wids).sum())
    return {
        "n_windows": len(wids), "n_complete": len(complete), "n_complete_le_maxlen": len(small), "n_recovered": len(rec), "n_recovered_labelled": len(rec_lab),
        "recall_of_complete": len(rec) / len(complete) if complete else float("nan"),
        "recall_of_reachable": len(rec & small) / len(small) if small else float("nan"),
        "candidates": n_cand, "candidates_per_window": n_cand / len(wids) if wids else float("nan"),
    }


def synthetic_recall(candidates, tx: pd.DataFrame, planted, x: float = 0.8) -> dict:
    """Per planted wash ring (scenario isolated / dense / chorded): recovered or not, plus the candidate count."""
    from ringtda.eval.labels import label_candidates

    lab = label_candidates(candidates, tx, planted, x)
    hit = set(lab.loc[lab.kind == "wash_ring", "pattern_id"].astype(int))
    rings = [(p.pattern_id, p.scenario) for p in planted if p.is_positive]
    return {"rings": [(pid, sc, pid in hit) for pid, sc in rings], "n_candidates": len(candidates)}
