"""Labels, groups and positive-group counts for the HI-Medium candidates (no features, no models).

Adapted from experiments/ibm_split_report.py. Reads medium/data/, writes ONLY inside medium/out/.
Rebuilds the candidates of every window with the same code and the same tau as medium/build_features.py (same order,
so candidate_id = cand_id), labels them with the section 3.7 rule, assigns groups, and CROSS-CHECKS against the
medium/out/<ds>_tau<tau>/ids_window_NN.csv.gz files written by the feature builder (same window, same candidate id,
same account sequence). It stops with an error if anything does not line up.

    python -I medium/make_labels.py --dataset HI --tau 0.05

Output: medium/out/ibm_candidates_<ds>-Medium_<tau>.pkl and medium/out/label_counts_<ds>-Medium_<tau>.json
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from ringtda.data.ibm_aml import load_config, load_ibm_files
from ringtda.eval.splits import assign_groups, group_summary, stratified_group_cv
from ringtda.ibm_table import ibm_candidate_tables

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["HI", "LI"], required=True)
ap.add_argument("--tau", type=float, default=0.05)
args = ap.parse_args()

HERE = Path(__file__).resolve().parent
name = f"{args.dataset}-Medium"
FEAT = HERE / "out" / f"{args.dataset}_tau{args.tau}"
cfg = load_config()
sp = cfg["splits"]
t0 = time.time()


def log(*a):
    print(time.strftime("%H:%M:%S"), f"[{time.time() - t0:5.0f}s]", *a, flush=True)


data = load_ibm_files(HERE / "data" / f"{name}_Trans.csv", HERE / "data" / f"{name}_Patterns.txt", name)
log(f"{name}: loaded {len(data.tx)} transactions, {len(data.attempts)} attempts")
df = ibm_candidate_tables(data, cfg, tau_quantiles=[args.tau])[args.tau]
log(f"candidate table: {len(df)} rows over {df.window_id.nunique()} windows")

# ---- cross-check against the feature builder's id files
n_checked = 0
for wid, g in df.groupby("window_id"):
    ids = pd.read_csv(FEAT / f"ids_window_{wid:02d}.csv.gz")
    assert len(ids) == len(g), f"window {wid}: {len(ids)} feature rows vs {len(g)} candidates"
    assert (ids.candidate_id.to_numpy() == g.cand_id.to_numpy()).all(), f"window {wid}: candidate ids differ"
    mine = [" ".join(map(str, n)) for n in g.nodes]
    assert mine == ids.nodes.tolist(), f"window {wid}: account sequences differ"
    n_checked += len(g)
log(f"cross-check OK: {n_checked} candidates match the feature rows exactly (window, id and accounts)")

# ---- groups and counts
df = df.assign(
    group=assign_groups(df.edges.tolist(), df.attempt.to_numpy()),
    group_strict=assign_groups(df.edges.tolist(), df.attempt.to_numpy(), df.nodes.tolist(), strict=True),
)
df.to_pickle(HERE / "out" / f"ibm_candidates_{name}_{args.tau}.pkl")


def counts(d):
    pos = d[d.y == 1]
    r = {
        "candidates": len(d), "negatives": int((d.y == 0).sum()), "positives": len(pos), "ambiguous": int(d.y.isna().sum()),
        "groups": int(d.group.nunique()), "positive_groups": int(pos.group.nunique()),
        "positive_attempts": int(pos.attempt.nunique()),
        "cycle_positives": int((pos.kind == "CYCLE").sum()), "cycle_positive_groups": int(pos[pos.kind == "CYCLE"].group.nunique()),
        "positives_by_type": {k: int(v) for k, v in pos.kind.value_counts().items()},
    }
    primary = d[d.y.notna()]
    r["primary_candidates"] = len(primary)
    r["prevalence_primary"] = float(primary.y.mean())
    return r


out = {"dataset": name, "tau": args.tau, "edge_groups": counts(df)}
d2 = df.assign(group=df.group_strict)
out["account_groups_strict"] = counts(d2)
gs = group_summary(df.group.to_numpy(), df.y.to_numpy())
out["largest_group_share"] = gs["largest_group_share"]

# positive-group support per training/test side over the project's CV (5 folds x 5 repeats)
sup = []
for run in stratified_group_cv(df.group.to_numpy(), df.y.to_numpy(), sp["k"], sp["fallback_k"], sp["repeats"], sp["seed"]):
    for f in range(run["k"]):
        te = df[run["fold"] == f]
        sup.append(int(te[te.y == 1].group.nunique()))
out["cv_test_positive_groups_per_fold"] = {"min": min(sup), "median": float(pd.Series(sup).median()), "max": max(sup), "n_folds_below_5": int(sum(s < 5 for s in sup)), "n_folds": len(sup)}
json.dump(out, open(HERE / "out" / f"label_counts_{name}_{args.tau}.json", "w"), indent=1)
log(json.dumps(out["edge_groups"], indent=1))
log("strict (account) grouping:", out["account_groups_strict"]["positive_groups"], "positive groups")
log("CV test-fold positive groups:", out["cv_test_positive_groups_per_fold"])
log("DONE")
