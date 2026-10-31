"""HI <-> LI transfer (README "Phase 3 design decisions"). Report only; no pass/fail gate.

    nice -n 10 python experiments/ibm_transfer.py --tau 0.05

Train on every primary-variant candidate (ambiguous excluded) of one dataset, test on every one of the other,
in both directions. Scaling is fitted on the training dataset only. PR-AUC with a grouped bootstrap CI over the
TEST groups; paired differences inside the same resamples. This is transfer, not a time-leak-free test: both
datasets come from the same generator and cover the same dates (only 3 accounts are shared, used by no candidate).
Fixed hyperparameters from configs/classify.yaml.
"""

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ringtda.eval.metrics import GroupedBootstrap, mean_ap_with_ci, paired_difference
from ringtda.models.classify import MODELS, load_config, named_feature_sets, transfer_scores

ap_ = argparse.ArgumentParser()
ap_.add_argument("--tau", default="0.05")
ap_.add_argument("--n-boot", type=int, default=None)
ap_.add_argument("--out", default="results/phase3_features/classify/transfer")
args = ap_.parse_args()
cfg = load_config()
if args.n_boot:
    cfg["bootstrap"]["n_boot"] = args.n_boot
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
t_start = time.time()
SETS = named_feature_sets()
FSETS = ["B", "B+D", "B+D+T", "B+D+T_local", "T", "D"]
PAIRS = [("B+D+T_local", "B+D"), ("B+D+T", "B+D")]


def load(ds):
    fdir = Path("results/phase3_features/ibm") / f"{ds}_tau{args.tau}"
    F = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(fdir / "features_window_*.csv.gz")))], ignore_index=True)
    L = pd.read_pickle(f"results/phase3_features/ibm_candidates_{ds}_{float(args.tau)}.pkl")
    L = L[["window_id", "cand_id", "y", "group"]].rename(columns={"cand_id": "candidate_id"})
    d = F.merge(L, on=["window_id", "candidate_id"], how="outer", indicator=True, validate="one_to_one")
    assert (d["_merge"] == "both").all() and len(d) == len(L) == len(F)
    d = d[d.y.notna()].reset_index(drop=True)  # primary variant: ambiguous excluded on both sides
    return d.copy()


data = {ds: load(ds) for ds in ("HI", "LI")}
rows, pairs = [], []
for train, test in (("HI", "LI"), ("LI", "HI")):
    tr, te = data[train], data[test]
    ytr, yte = tr.y.to_numpy(int), te.y.to_numpy(int)
    gte = te.group.to_numpy()
    n_posg = int(pd.Series(gte[yte == 1]).nunique())
    gb = GroupedBootstrap(gte, cfg["bootstrap"]["n_boot"], cfg["bootstrap"]["seed"])
    print(f"{train}->{test}: train {len(tr)} rows ({ytr.sum()} pos), test {len(te)} rows ({yte.sum()} pos in {n_posg} groups)", flush=True)
    boots = {}

    def record(sname, mname, sc):
        res = mean_ap_with_ci(yte, [sc], gb)
        boots[(mname, sname)] = res
        rows.append({"train": train, "test": test, "feature_set": sname, "model": mname, "pr_auc": res["ap"], "ci_low": res["ci_low"], "ci_high": res["ci_high"], "prevalence": yte.mean(),
                     "n_test_rows": len(yte), "n_test_pos": int(yte.sum()), "n_test_pos_groups": n_posg, "n_train_rows": len(ytr), "n_train_pos": int(ytr.sum()),
                     "n_boot": gb.n_boot, "n_boot_nan": res["n_boot_nan"], "model_seed": cfg["model_seed"], "boot_seed": cfg["bootstrap"]["seed"]})  # fmt: skip
        print(f"  {sname:12s} {mname:6s} PR-AUC {res['ap']:.3f} [{res['ci_low']:.3f}, {res['ci_high']:.3f}]", flush=True)

    for sname in FSETS:
        cols = SETS[sname]
        for mname in MODELS:
            record(sname, mname, transfer_scores(tr[cols].to_numpy(float), ytr, te[cols].to_numpy(float), mname, cfg, cfg["model_seed"]))
    record("D_birth_alone", "none", -te["D_birth"].to_numpy(float))
    for mname in MODELS:
        for a, b in PAIRS:
            pairs.append({"train": train, "test": test, "model": mname, "comparison": f"{a} minus {b}", **paired_difference(boots[mname, a], boots[mname, b]), "n_boot": gb.n_boot})

R, P = pd.DataFrame(rows), pd.DataFrame(pairs)
R.to_csv(OUT / "results.csv", index=False)
P.to_csv(OUT / "paired_differences.csv", index=False)
yaml.safe_dump({"config": cfg, "args": vars(args), "runtime_seconds": time.time() - t_start}, open(OUT / "config_used.yaml", "w"))
f3 = lambda r: f"{r.pr_auc:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]"  # noqa: E731
md = ["# IBM HI <-> LI transfer, tau_quantile " + args.tau + ", max_len 6\n",
      "Transfer, not a time-leak-free test: both datasets come from the same generator and cover the same dates; only 3 accounts are shared, used by no candidate. "
      "Primary variant (ambiguous excluded in train and test); scaling fitted on the training dataset; CI = grouped bootstrap over the test groups.\n"]  # fmt: skip
for (train, test), g in R.groupby(["train", "test"], sort=False):
    r0 = g.iloc[0]
    md.append(f"\n## train {train} -> test {test}: {r0.n_test_rows} test rows, {r0.n_test_pos} positives ({r0.prevalence:.4f}) in {r0.n_test_pos_groups} groups; train positives {r0.n_train_pos}\n")
    md.append("| feature set | " + " | ".join(MODELS) + " |\n|---|" + "---|" * len(MODELS))
    cell = {(r.feature_set, r.model): f3(r) for r in g.itertuples()}
    md += [f"| {s} | " + " | ".join(cell[s, m] for m in MODELS) + " |" for s in FSETS]
    md.append(f"\n-D_birth alone (no model): {cell['D_birth_alone', 'none']}\n")
    q = P[(P.train == train) & (P.test == test)]
    md.append("| model | comparison | diff | 95% CI | share of resamples > 0 |\n|---|---|---|---|---|")
    md += [f"| {x.model} | {x.comparison} | {x.diff:+.3f} | [{x.ci_low:+.3f}, {x.ci_high:+.3f}] | {x.share_boot_positive:.2f} |" for x in q.itertuples()]
(OUT / "summary.md").write_text("\n".join(md) + "\n")
print(f"done in {time.time() - t_start:.0f}s -> {OUT}", flush=True)
