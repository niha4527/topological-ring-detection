"""Classification on IBM HI-Medium: same protocol as experiments/ibm_classify.py, primary variant only.

Reads medium/out/<ds>_tau<tau>/features_window_*.csv.gz and medium/out/ibm_candidates_<ds>-Medium_<tau>.pkl (from
medium/build_features.py and medium/make_labels.py), joined one-to-one on (window_id, candidate_id). Fixed hyperparameters
(configs/classify.yaml, no search), stratified GROUP k-fold with repeats, pooled out-of-fold PR-AUC, grouped bootstrap 95% CIs,
paired differences inside the same resamples. Ambiguous candidates are excluded (primary variant). Writes ONLY to
medium/out/classify/<ds>-Medium_tau<tau>/<tag>/. Report only; no pass/fail gate.

    python -I medium/classify.py --models logreg --tag lr
"""

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ringtda.eval.metrics import GroupedBootstrap, mean_ap_with_ci, paired_difference
from ringtda.eval.splits import stratified_group_cv
from ringtda.models.classify import load_config, named_feature_sets, oof_scores, positive_group_support

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="HI")
ap.add_argument("--tau", default="0.05")
ap.add_argument("--models", nargs="+", default=["logreg"])
ap.add_argument("--sets", nargs="+", default=["B", "B+D", "B+D+T", "B+D+T_local"])
ap.add_argument("--repeats", type=int, default=None)
ap.add_argument("--n-boot", type=int, default=None)
ap.add_argument("--rf-trees", type=int, default=None)
ap.add_argument("--rf-jobs", type=int, default=None, help="threads only; with a fixed random_state this does not change the forest")
ap.add_argument("--tag", default="run")
args = ap.parse_args()

HERE = Path(__file__).resolve().parent
name = f"{args.dataset}-Medium"
cfg = load_config()
if args.repeats:
    cfg["cv"]["repeats"] = args.repeats
if args.n_boot:
    cfg["bootstrap"]["n_boot"] = args.n_boot
if args.rf_trees:
    cfg["models"]["rf"]["n_estimators"] = args.rf_trees
if args.rf_jobs:
    cfg["models"]["rf"]["n_jobs"] = args.rf_jobs
OUT = HERE / "out" / "classify" / f"{name}_tau{args.tau}" / args.tag
OUT.mkdir(parents=True, exist_ok=True)
t_start = time.time()

F = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(HERE / "out" / f"{args.dataset}_tau{args.tau}" / "features_window_*.csv.gz")))], ignore_index=True)
L = pd.read_pickle(HERE / "out" / f"ibm_candidates_{name}_{float(args.tau)}.pkl")
L = L[["window_id", "cand_id", "kind", "y", "group"]].rename(columns={"cand_id": "candidate_id"})
df = F.merge(L, on=["window_id", "candidate_id"], how="outer", indicator=True, validate="one_to_one")
assert (df["_merge"] == "both").all() and len(df) == len(L) == len(F), "feature/label join is not one-to-one"
df = df.drop(columns="_merge")
print(f"{name}: {len(df)} candidates joined one-to-one; positives {int((df.y == 1).sum())}, ambiguous {int(df.y.isna().sum())}", flush=True)
SETS = named_feature_sets()
for s in args.sets:
    assert not df[SETS[s]].isna().any().any(), s

sub = df[df.y.notna()].reset_index(drop=True)
y = sub.y.to_numpy(float).astype(int)
groups = sub.group.to_numpy()
n_pos, n_posg = int(y.sum()), int(pd.Series(groups[y == 1]).nunique())
runs = stratified_group_cv(groups, y.astype(float), cfg["cv"]["k"], cfg["cv"]["fallback_k"], cfg["cv"]["repeats"], cfg["cv"]["seed"])
ks = [r["k"] for r in runs]
nofold = [r["fold_without_positive"] for r in runs]
support = min(positive_group_support(groups, y, r["fold"]) for r in runs)
gb = GroupedBootstrap(groups, cfg["bootstrap"]["n_boot"], cfg["bootstrap"]["seed"])
print(f"primary: rows {len(sub)}, positives {n_pos} in {n_posg} groups, prevalence {n_pos / len(y):.4f}, k per repeat {ks}, "
      f"folds without positive {nofold}, smallest positive-group support on a side {support}", flush=True)

rows, boots, oofs = [], {}, {}
for mname in args.models:
    for sname in args.sets:
        t0 = time.time()
        X = sub[SETS[sname]].to_numpy(float)
        sc = [oof_scores(X, y, groups, runs[r]["fold"], mname, cfg, cfg["model_seed"] + r) for r in range(len(runs))]
        oofs[f"primary|{mname}|{sname}"] = np.stack(sc)
        res = mean_ap_with_ci(y, sc, gb)
        boots[(mname, sname)] = res
        rows.append({"dataset": name, "tau": args.tau, "variant": "primary", "feature_set": sname, "model": mname, "n_features": len(SETS[sname]),
                     "pr_auc": res["ap"], "ci_low": res["ci_low"], "ci_high": res["ci_high"], "prevalence": n_pos / len(y), "n_rows": len(y),
                     "n_pos": n_pos, "n_pos_groups": n_posg, "min_pos_groups_on_a_side": int(support), "k_per_repeat": ",".join(map(str, ks)),
                     "folds_without_positive": int(sum(nofold)), "repeats": len(sc), "pr_auc_by_repeat": ",".join(f"{a:.4f}" for a in res["ap_by_repeat"]),
                     "n_boot": gb.n_boot, "n_boot_nan": res["n_boot_nan"], "cv_seed": cfg["cv"]["seed"], "model_seed": cfg["model_seed"], "boot_seed": cfg["bootstrap"]["seed"]})  # fmt: skip
        print(f"  {sname:12s} {mname:6s} PR-AUC {res['ap']:.3f} [{res['ci_low']:.3f}, {res['ci_high']:.3f}] ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)

s = -sub["D_birth"].to_numpy(float)
res = mean_ap_with_ci(y, [s], gb)
rows.append({"dataset": name, "tau": args.tau, "variant": "primary", "feature_set": "D_birth_alone", "model": "none", "n_features": 1, "pr_auc": res["ap"], "ci_low": res["ci_low"],
             "ci_high": res["ci_high"], "prevalence": n_pos / len(y), "n_rows": len(y), "n_pos": n_pos, "n_pos_groups": n_posg})  # fmt: skip
print(f"  D_birth alone (no model) PR-AUC {res['ap']:.3f} [{res['ci_low']:.3f}, {res['ci_high']:.3f}]", flush=True)
pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)
np.savez_compressed(OUT / "oof_scores.npz", **oofs)

pairs = []
for mname in args.models:
    for a, b in cfg["paired"]:
        if (mname, a) in boots and (mname, b) in boots:
            d = paired_difference(boots[mname, a], boots[mname, b])
            pairs.append({"dataset": name, "variant": "primary", "model": mname, "comparison": f"{a} minus {b}", **d, "n_boot": gb.n_boot})
P = pd.DataFrame(pairs)
P.to_csv(OUT / "paired_differences.csv", index=False)
yaml.safe_dump({"config": cfg, "args": vars(args), "runtime_seconds": time.time() - t_start, "n_candidates": len(df)}, open(OUT / "config_used.yaml", "w"))
print("\nPaired differences (same resamples):")
for q in P.itertuples():
    print(f"  {q.model:6s} {q.comparison:24s} {q.diff:+.3f} [{q.ci_low:+.3f}, {q.ci_high:+.3f}]  share>0 {q.share_boot_positive:.2f}", flush=True)
print(f"done in {time.time() - t_start:.0f}s -> {OUT}", flush=True)
