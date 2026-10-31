"""Classification harness on IBM AML (README "Phase 3 design decisions"). Report only; no pass/fail gate.

    nice -n 10 python experiments/ibm_classify.py --dataset HI --tau 0.05

Reads the B/D/T feature files of experiments/ibm_feature_build.py and the labels/groups of the split report
(results/phase3_features/ibm_candidates_<ds>_<tau>.pkl), joined one-to-one on (window_id, candidate_id).
Fixed hyperparameters (configs/classify.yaml, no search). Stratified GROUP k-fold with repeats, pooled
out-of-fold PR-AUC, grouped bootstrap 95% CIs, paired differences inside the same resamples.

Variants: primary (ambiguous excluded), ambiguous_positive, cycle_only (positives = CYCLE attempts, other
positives and ambiguous candidates excluded; background kept), strict_groups (primary labels, groups merged
by shared account). Controls (primary only): D, T_local, T_win, and the single-feature detector -D_birth.
"""

import argparse
import copy
import glob
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ringtda.eval.metrics import GroupedBootstrap, average_precision, ci, mean_ap_with_ci, paired_difference
from ringtda.eval.splits import stratified_group_cv
from ringtda.models.classify import MODELS, load_config, named_feature_sets, oof_scores, positive_group_support

ap_ = argparse.ArgumentParser()
ap_.add_argument("--dataset", default="HI")
ap_.add_argument("--tau", default="0.05")
ap_.add_argument("--variants", nargs="+", default=["primary", "ambiguous_positive", "cycle_only", "strict_groups"])
ap_.add_argument("--only-sets", nargs="+", default=None, help="restrict feature sets (quick test)")
ap_.add_argument("--repeats", type=int, default=None)
ap_.add_argument("--n-boot", type=int, default=None)
ap_.add_argument("--rf-trees", type=int, default=None)
ap_.add_argument("--out", default=None)
ap_.add_argument("--append", action="store_true", help="run only the ablation sets and ADD their rows to the existing results in --out (existing rows untouched)")
ap_.add_argument("--require-pos-groups", type=int, default=0, help="skip the cycle_only variant unless every train and test side has at least this many positive groups")
args = ap_.parse_args()

cfg = load_config()
if args.repeats:
    cfg["cv"]["repeats"] = args.repeats
if args.n_boot:
    cfg["bootstrap"]["n_boot"] = args.n_boot
if args.rf_trees:
    cfg["models"]["rf"]["n_estimators"] = args.rf_trees
OUT = Path(args.out) if args.out else Path("results/phase3_features/classify") / f"{args.dataset}_tau{args.tau}"
OUT.mkdir(parents=True, exist_ok=True)
t_start = time.time()

# ---- load and join (one-to-one on window_id, candidate_id)
fdir = Path("results/phase3_features/ibm") / f"{args.dataset}_tau{args.tau}"
F = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(fdir / "features_window_*.csv.gz")))], ignore_index=True)
L = pd.read_pickle(f"results/phase3_features/ibm_candidates_{args.dataset}_{float(args.tau)}.pkl")
L = L[["window_id", "cand_id", "kind", "y", "group", "group_strict"]].rename(columns={"cand_id": "candidate_id"})
df = F.merge(L, on=["window_id", "candidate_id"], how="outer", indicator=True, validate="one_to_one")
assert (df["_merge"] == "both").all() and len(df) == len(L) == len(F), "feature/label join is not one-to-one"
df = df.drop(columns="_merge").copy()
print(f"{args.dataset}: {len(df)} candidates joined one-to-one; positives {int((df.y == 1).sum())}, ambiguous {int(df.y.isna().sum())}", flush=True)
SETS = named_feature_sets()
for name, cols in SETS.items():
    assert not df[cols].isna().any().any(), name

models = list(MODELS)
ablation = cfg.get("ablation_sets", [])
base_sets = ablation if args.append else cfg["feature_sets"] + ablation
feat_sets = [s for s in base_sets if not args.only_sets or s in args.only_sets]
ctrl_sets = [] if args.append else [s for s in cfg["control_sets"] if not args.only_sets or s in args.only_sets]
all_sets = [s for s in cfg["feature_sets"] + ablation + cfg["control_sets"]]
old_R, old_P, unusable = None, None, []
if args.append:
    old_R, old_P = pd.read_csv(OUT / "results.csv"), pd.read_csv(OUT / "paired_differences.csv")
    with np.load(OUT / "oof_scores.npz") as z:
        old_oof = {k: z[k] for k in z.files}


def variant(name):
    y = df.y.copy()
    if name in ("primary", "strict_groups"):
        keep = y.notna()
    elif name == "ambiguous_positive":
        keep, y = pd.Series(True, index=df.index), y.fillna(1.0)
    elif name == "cycle_only":
        keep = (y == 0) | ((y == 1) & (df.kind == "CYCLE"))
    else:
        raise ValueError(name)
    gcol = "group_strict" if name == "strict_groups" else "group"
    return keep.to_numpy(), y, gcol


rows, boots, oofs = [], {}, (dict(old_oof) if args.append else {})
for vname in args.variants:
    keep, yfull, gcol = variant(vname)
    sub = df[keep].reset_index(drop=True)
    y = yfull[keep].to_numpy(float).astype(int)
    groups = sub[gcol].to_numpy()
    n_pos, n_posg = int(y.sum()), int(pd.Series(groups[y == 1]).nunique())
    runs = stratified_group_cv(groups, y.astype(float), cfg["cv"]["k"], cfg["cv"]["fallback_k"], cfg["cv"]["repeats"], cfg["cv"]["seed"])
    ks = [r["k"] for r in runs]
    nofold = [r["fold_without_positive"] for r in runs]
    support = min(positive_group_support(groups, y, r["fold"]) for r in runs)
    print(f"[{vname}] smallest number of positive groups on a train or test side over all folds: {support}", flush=True)
    if vname == "cycle_only" and support < args.require_pos_groups:
        print(f"[{vname}] UNUSABLE: a train or test side has {support} positive groups (< {args.require_pos_groups}); skipped", flush=True)
        unusable.append({"variant": vname, "min_positive_groups_on_a_side": int(support), "positives": n_pos, "positive_groups": n_posg})
        continue
    gb = GroupedBootstrap(groups, cfg["bootstrap"]["n_boot"], cfg["bootstrap"]["seed"])
    if args.append:  # rebuild the bootstrap results of the stored out-of-fold scores (same resamples) for the paired rows
        for sname in all_sets:
            for mname in models:
                key = f"{vname}|{mname}|{sname}"
                if key in old_oof:
                    boots[(vname, mname, sname)] = mean_ap_with_ci(y, list(old_oof[key]), gb)
                    ref = old_R[(old_R.variant == vname) & (old_R.model == mname) & (old_R.feature_set == sname)].pr_auc.iloc[0]
                    assert abs(boots[(vname, mname, sname)]["ap"] - ref) < 1e-9, f"stored scores do not reproduce results.csv for {key}"
    win_pos = [w for w in sub.window_id.unique() if y[(sub.window_id == w).to_numpy()].sum() > 0]
    print(f"[{vname}] rows {len(sub)}, positives {n_pos} in {n_posg} groups, k per repeat {ks}, folds without positive {nofold}", flush=True)

    def record(sname, mname, scores_by_repeat, n_feat):
        res = mean_ap_with_ci(y, scores_by_repeat, gb)
        per_win = np.mean([np.mean([average_precision(y[m], s[m]) for w in win_pos for m in [(sub.window_id == w).to_numpy()]]) for s in scores_by_repeat])
        boots[(vname, mname, sname)] = res
        rows.append(
            {"dataset": args.dataset, "tau": args.tau, "variant": vname, "feature_set": sname, "model": mname, "n_features": n_feat, "pr_auc": res["ap"], "ci_low": res["ci_low"], "ci_high": res["ci_high"],
             "prevalence": n_pos / len(y), "n_rows": len(y), "n_pos": n_pos, "n_pos_groups": n_posg, "min_pos_groups_on_a_side": int(support), "k_per_repeat": ",".join(map(str, ks)), "folds_without_positive": int(sum(nofold)),
             "repeats": len(scores_by_repeat), "pr_auc_by_repeat": ",".join(f"{a:.4f}" for a in res["ap_by_repeat"]), "per_window_pr_auc": per_win, "n_windows_with_pos": len(win_pos),
             "n_boot": gb.n_boot, "n_boot_nan": res["n_boot_nan"], "cv_seed": cfg["cv"]["seed"], "model_seed": cfg["model_seed"], "boot_seed": cfg["bootstrap"]["seed"]}
        )  # fmt: skip

    sets_here = feat_sets + (ctrl_sets if vname == "primary" else [])
    for sname in sets_here:
        X = sub[SETS[sname]].to_numpy(float)
        for mname in models:
            t0 = time.time()
            sc = [oof_scores(X, y, groups, runs[r]["fold"], mname, cfg, cfg["model_seed"] + r) for r in range(len(runs))]
            oofs[f"{vname}|{mname}|{sname}"] = np.stack(sc)
            record(sname, mname, sc, len(SETS[sname]))
            r_ = rows[-1]
            print(f"  {vname:18s} {sname:8s} {mname:6s} PR-AUC {r_['pr_auc']:.3f} [{r_['ci_low']:.3f}, {r_['ci_high']:.3f}] ({time.time() - t0:.0f}s)", flush=True)
    if vname == "primary" and not args.append and (not args.only_sets or "D" in args.only_sets):
        s = -sub["D_birth"].to_numpy(float)
        record("D_birth_alone", "none", [s], 1)
        r_ = rows[-1]
        print(f"  primary            D_birth  none   PR-AUC {r_['pr_auc']:.3f} [{r_['ci_low']:.3f}, {r_['ci_high']:.3f}]", flush=True)

R = pd.DataFrame(rows)
if args.append:
    R = pd.concat([old_R, R], ignore_index=True)
R.to_csv(OUT / "results.csv", index=False)
np.savez_compressed(OUT / "oof_scores.npz", **oofs)

pairs = []
for vname in args.variants:
    for mname in models:
        for a, b in cfg["paired"]:
            if (vname, mname, a) in boots and (vname, mname, b) in boots:
                d = paired_difference(boots[vname, mname, a], boots[vname, mname, b])
                pairs.append({"dataset": args.dataset, "variant": vname, "model": mname, "comparison": f"{a} minus {b}", **d, "n_boot": cfg["bootstrap"]["n_boot"]})
P = pd.DataFrame(pairs)
if args.append:
    done = set(zip(old_P.variant, old_P.model, old_P.comparison))
    P = pd.concat([old_P, P[[(v, m, c) not in done for v, m, c in zip(P.variant, P.model, P.comparison)]]], ignore_index=True)
P.to_csv(OUT / "paired_differences.csv", index=False)

# ---- config and seeds stored with the results (NFR5)
cfg_file = OUT / ("config_used_ablation.yaml" if args.append else "config_used.yaml")
yaml.safe_dump({"config": cfg, "args": vars(args), "runtime_seconds": time.time() - t_start, "n_candidates": len(df), "unusable_variants": unusable}, open(cfg_file, "w"))

# ---- markdown summary: numbers only
f3 = lambda r: f"{r.pr_auc:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]"  # noqa: E731
md = [f"# IBM {args.dataset}-Small classification, tau_quantile {args.tau}, max_len 6\n",
      f"Fixed hyperparameters (configs/classify.yaml, no search). Stratified group k-fold (k={cfg['cv']['k']}, fallback {cfg['cv']['fallback_k']}), {cfg['cv']['repeats']} repeats, "
      f"pooled out-of-fold PR-AUC (mean over repeats), grouped bootstrap 95% CI ({cfg['bootstrap']['n_boot']} resamples). Nothing is a pass/fail gate.\n"]  # fmt: skip
for vname in [v for v in args.variants if (R.variant == v).any()]:
    r0 = R[R.variant == vname].iloc[0]
    md.append(f"\n## {vname}: {r0.n_rows} candidates, {r0.n_pos} positives ({r0.prevalence:.4f}) in {r0.n_pos_groups} groups; k per repeat {r0.k_per_repeat}; folds without positive {r0.folds_without_positive}\n")
    t = R[(R.variant == vname) & (R.model != "none")].pivot(index="feature_set", columns="model", values="pr_auc").reindex([s for s in all_sets if s in set(R.feature_set)])
    cell = {(r.feature_set, r.model): f3(r) for r in R[(R.variant == vname)].itertuples()}
    md.append("PR-AUC [95% CI] (mean over repeats), prevalence " + f"{r0.prevalence:.4f}:\n\n| feature set | " + " | ".join(models) + " |\n|---|" + "---|" * len(models))
    for s in t.index:
        md.append(f"| {s} | " + " | ".join(cell.get((s, m), "") for m in models) + " |")
    if vname == "primary" and ("D_birth_alone", "none") in cell:
        md.append(f"\n-D_birth alone (no model): {cell[('D_birth_alone', 'none')]}")
    pw = R[(R.variant == vname) & (R.model != "none")].pivot(index="feature_set", columns="model", values="per_window_pr_auc")
    md.append(f"\nMean of per-window PR-AUC over the {r0.n_windows_with_pos} windows with at least one positive:\n\n" + pw.round(3).to_string() + "\n")
    p = P[P.variant == vname]
    md.append("Paired differences (same resamples):\n\n| model | comparison | diff | 95% CI | share of resamples > 0 |\n|---|---|---|---|---|")
    md += [f"| {q.model} | {q.comparison} | {q.diff:+.3f} | [{q.ci_low:+.3f}, {q.ci_high:+.3f}] | {q.share_boot_positive:.2f} |" for q in p.itertuples()]
(OUT / "summary.md").write_text("\n".join(md) + "\n")
print(f"done in {time.time() - t_start:.0f}s -> {OUT}", flush=True)
