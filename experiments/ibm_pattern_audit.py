"""IBM AML pattern-behaviour audit (proposal v3 §3.7, §4.10): do s_rec, s_bal, s_reg and the combined weight
separate laundering pairs from background pairs, per pattern type? Measurement only; no weights change.

    nice -n 10 python experiments/ibm_pattern_audit.py --dataset HI [--windows 0 2]

One window at a time (window 0 = days 0-5, window 2 = days 2-7 by default), then pooled over the audited windows
(a pair present in two windows counts twice). Writes results/phase3_features/audit/<dataset>_*.csv and
<dataset>_audit_summary.md; the figure comes from experiments/ibm_pattern_audit_plot.py.
"""

import argparse
import resource
import time
from pathlib import Path

import pandas as pd

from ringtda.data.ibm_aml import iter_ibm_windows, load_config, load_ibm
from ringtda.eval.audit import COLUMNS, add_balance_only_weight, edge_membership, imputation_counts, ring_scores, summarize
from ringtda.graph.weights import edge_weights
from ringtda.graph.windows import aggregate_pairs

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["HI", "LI"], default="HI")
ap.add_argument("--windows", type=int, nargs="+", default=[0, 2])
args = ap.parse_args()
OUT = Path("results") / "phase3_features" / "audit"
OUT.mkdir(parents=True, exist_ok=True)
cfg = load_config()
t0 = time.time()
data = load_ibm(args.dataset)
print(f"{args.dataset}: loaded in {time.time() - t0:.0f}s; pattern types in the patterns file: {sorted(data.attempts.pattern.unique())}", flush=True)

cand_pk = Path("results/phase3_features") / f"ibm_candidates_{args.dataset}_0.05.pkl"
cands = pd.read_pickle(cand_pk) if cand_pk.exists() else None

summ, imps, rings, compact = [], [], [], []
for wid, start, end, w in iter_ibm_windows(data, cfg["window"]["length_days"], cfg["window"]["stride_days"]):
    if wid not in args.windows:
        continue
    t1 = time.time()
    pairs = add_balance_only_weight(edge_weights(aggregate_pairs(w)))
    groups = edge_membership(pairs, w)
    label = f"w{wid} (days {start:.0f}-{end:.0f})"
    summ.append(summarize(pairs, groups, label))
    imps.append(imputation_counts(pairs, groups, label))
    if cands is not None:
        c = cands[cands.window_id == wid]
        r = ring_scores(pairs, c)
        rings.append(r.assign(window=label))
    keep = pairs[COLUMNS + ["gap_cv"]].copy()
    for g, m in groups.items():
        keep["g_" + g] = m
    compact.append(keep)
    print(f"{label}: {len(pairs)} pairs, laundering groups { {g: int(m.sum()) for g, m in groups.items()} } ({time.time() - t1:.0f}s)", flush=True)
    del pairs

pool = pd.concat(compact, ignore_index=True)
pgroups = {c[2:]: pool[c].to_numpy() for c in pool.columns if c.startswith("g_")}
summ.append(summarize(pool, pgroups, "pooled"))
imps.append(imputation_counts(pool, pgroups, "pooled"))
S, I = pd.concat(summ, ignore_index=True), pd.concat(imps, ignore_index=True)
S.insert(0, "dataset", args.dataset)
I.insert(0, "dataset", args.dataset)
S.to_csv(OUT / f"{args.dataset}_audit_summary.csv", index=False)
I.to_csv(OUT / f"{args.dataset}_audit_imputation.csv", index=False)
if rings:
    R = pd.concat(rings, ignore_index=True)
    R["group"] = R.kind.where(R.y == 1, R.kind.where(R.kind == "background", "ambiguous"))
    RS = R.groupby(["window", "group"]).agg(n=("y", "size"), birth=("birth", "mean"), bal_mean=("n_s_bal_mean", "mean"), rec_mean=("n_s_rec_mean", "mean"), reg_mean=("n_s_reg_mean", "mean"), bal_min=("n_s_bal_min", "mean")).reset_index()
    RS.insert(0, "dataset", args.dataset)
    RS.to_csv(OUT / f"{args.dataset}_audit_rings.csv", index=False)

# markdown summary: numbers only
pd.set_option("display.width", 220)
fmt = lambda df: "```\n" + df.round(3).to_string() + "\n```\n"  # noqa: E731
md = [f"# IBM {args.dataset}-Small pattern-behaviour audit\n", f"Windows audited: {args.windows} (5 days each) and pooled. Direction: higher score / lower weight = more suspicious.\n"]
for w in S.window.unique():
    s = S[S.window == w]
    md.append(f"\n## {w}\n\nPairs per group:\n" + fmt(s.drop_duplicates("group").set_index("group")[["n_pairs"]].T))
    md.append("Edge-level AUC of group vs background:\n" + fmt(s[s.group != "background"].pivot(index="group", columns="score", values="auc_vs_background")[COLUMNS]))
    md.append("Tail share = share of the group's pairs strictly inside the most suspicious 10% tail of the window (for weight: the lowest 10% of weights; baseline = share of all pairs, 0.100 when no ties; all pairs: " + ", ".join(f"{c} {s[s.score == c].decile_share_all.iloc[0]:.3f}" for c in COLUMNS) + "):\n" + fmt(s.pivot(index="group", columns="score", values="decile_share")[COLUMNS]))
    md.append("Share exactly equal to the window median (tie rate):\n" + fmt(s.pivot(index="group", columns="score", values="tie_window_median")[COLUMNS]))
    md.append("Median:\n" + fmt(s.pivot(index="group", columns="score", values="median")[COLUMNS]))
md.append("\n## gap_cv imputation (pairs with < 3 transactions)\n" + fmt(I.pivot(index="group", columns="window", values="share_imputed")))
if rings:
    md.append("\n## Rings (candidates at tau_quantile 0.05, max_len 6): mean over candidates\n" + fmt(RS.set_index(["window", "group"])))
(OUT / f"{args.dataset}_audit_summary.md").write_text("\n".join(md))
print(f"done in {time.time() - t0:.0f}s, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.1f} GB", flush=True)
