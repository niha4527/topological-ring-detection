"""Counts for the Phase 3 evaluation schemes on IBM AML (README "Phase 3 design decisions", decision 1).

    python experiments/ibm_split_report.py [--datasets HI LI]

Builds the candidate tables (5-day windows, 1-day stride, max_len 6, tau_quantile in {0.02, 0.05, 0.10})
and, per scheme, counts candidates by label, positives by pattern type, positive groups, cycle-only
positives and windows kept/dropped. Counts only: no features, no models. Writes
results/phase3_features/ibm_split_counts.csv (+ the candidate tables as pickles).
"""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from ringtda.data.ibm_aml import load_config, load_ibm
from ringtda.eval.splits import assign_groups, group_summary, stratified_group_cv, time_ordered_split, transfer_split
from ringtda.ibm_table import ibm_candidate_tables

logging.basicConfig(level=logging.WARNING)
ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", default=["HI", "LI"])
args = ap.parse_args()
cfg = load_config()
sp = cfg["splits"]
OUT = Path("results") / "phase3_features"
OUT.mkdir(parents=True, exist_ok=True)


def counts(d: pd.DataFrame) -> dict:
    pos = d[d.y == 1]
    r = {
        "candidates": len(d), "negatives": int((d.y == 0).sum()), "positives": len(pos), "ambiguous": int(d.y.isna().sum()),
        "groups": int(d.group.nunique()), "positive_groups": int(pos.group.nunique()),
        "cycle_positives": int((pos.kind == "CYCLE").sum()), "cycle_positive_groups": int(pos[pos.kind == "CYCLE"].group.nunique()),
    }
    for t, n in pos.kind.value_counts().items():
        r[f"pos_{t}"] = int(n)
    return r


rows, tables, accounts = [], {}, {}
for ds in args.datasets:
    t0 = time.time()
    data = load_ibm(ds)
    accounts[ds] = data.accounts
    print(f"{ds}: {data.stats}")
    tabs = ibm_candidate_tables(data, cfg)
    print(f"{ds}: tables built in {time.time() - t0:.0f}s")
    for q, df in tabs.items():
        df = df.assign(
            group=assign_groups(df.edges.tolist(), df.attempt.to_numpy()),
            group_strict=assign_groups(df.edges.tolist(), df.attempt.to_numpy(), df.nodes.tolist(), strict=True),
        )
        tables[ds, q] = df
        df.to_pickle(OUT / f"ibm_candidates_{ds}_{q}.pkl")
        for gcol, scheme in (("group", "grouped_cv_edge"), ("group_strict", "grouped_cv_account")):
            d = df.assign(group=df[gcol])
            gs = group_summary(d.group.to_numpy(), d.y.to_numpy())
            rows.append({"dataset": ds, "tau_quantile": q, "scheme": scheme, "part": "all", **counts(d), "largest_group_share": gs["largest_group_share"]})
            for run in stratified_group_cv(d.group.to_numpy(), d.y.to_numpy(), sp["k"], sp["fallback_k"], sp["repeats"], sp["seed"]):
                for f in range(run["k"]):
                    rows.append({"dataset": ds, "tau_quantile": q, "scheme": scheme, "part": f"fold{f}", "repeat": run["repeat"], "k": run["k"], **counts(d[run["fold"] == f])})
        t = time_ordered_split(df, df.group.to_numpy(), sp["time_train_max_day"], sp["time_test_min_day"])
        n_win, n_drop = t.window_id.nunique(), t[t.side == "dropped"].window_id.nunique()
        for side in ("train", "test"):
            rows.append({"dataset": ds, "tau_quantile": q, "scheme": "time_ordered", "part": side, **counts(t[t.side == side]), "windows_kept": n_win - n_drop, "windows_dropped": n_drop, "windows_total": n_win})
        rows.append({"dataset": ds, "tau_quantile": q, "scheme": "time_ordered", "part": "gap", **counts(t[t.side == "gap"])})

for q in cfg["candidates"]["tau_quantiles"]:
    if all((ds, q) in tables for ds in ("HI", "LI")):
        for a, b in (("HI", "LI"), ("LI", "HI")):
            tr = transfer_split(tables[a, q], tables[b, q], accounts[a], accounts[b])
            for part, src in (("train", a), ("test", b)):
                rows.append({"dataset": f"{a}->{b}", "tau_quantile": q, "scheme": "transfer", "part": part, **counts(tables[src, q]), **tr})

res = pd.DataFrame(rows)
res.to_csv(OUT / "ibm_split_counts.csv", index=False)

pd.set_option("display.width", 250)
cols = ["candidates", "negatives", "positives", "ambiguous", "positive_groups", "cycle_positives"]
print("\n=== all candidates / time-ordered / transfer ===")
print(res[res.part.isin(["all", "train", "test", "gap"])][["dataset", "tau_quantile", "scheme", "part", *cols, "windows_kept", "windows_dropped"]].to_string(index=False))
print("\n=== positives per fold, repeat 0 (min / max over folds) ===")
f = res[(res.part.str.startswith("fold")) & (res.repeat == 0)]
g = f.groupby(["dataset", "tau_quantile", "scheme"]).agg(k=("k", "first"), pos_min=("positives", "min"), pos_max=("positives", "max"), posgroups_min=("positive_groups", "min"), posgroups_max=("positive_groups", "max"), cand_min=("candidates", "min"), cand_max=("candidates", "max"))
print(g.to_string())
print("\n=== folds without a positive over all repeats, fallbacks to k=3 ===")
print(f"fold-runs with 0 positives: {int(((res.part.str.startswith('fold')) & (res.positives == 0)).sum())}; runs at k=3: {int(res[(res.part == 'fold0') & (res.k == 3)].shape[0])} of {int(res[res.part == 'fold0'].shape[0])}")
