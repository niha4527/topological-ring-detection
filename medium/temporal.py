"""Time-aware classical features on IBM HI-Medium: does a ring's history across overlapping windows help?

A ring (same account cycle) appears in several consecutive 5-day windows (stride 1 day). For each candidate this adds CAUSAL features
computed only from EARLIER windows (never later ones, never labels, never the window index or any absolute time):
  tp_seen        number of earlier windows in which the same ring existed
  tp_streak      consecutive windows immediately before this one in which it existed
  tp_birth_prev  ring birth in the previous window it existed in (the current birth if none: neutral fill)
  tp_birth_delta current birth minus tp_birth_prev
  tp_birth_min   lowest birth over its history (current birth if none)
  tp_birth_mean  mean birth over its history (current birth if none)
  tp_vol_delta   D_log_volume minus its previous value (0 if none)
  tp_bal_delta   D_bal_mean minus its previous value (0 if none)
  tp_reg_delta   D_reg_mean minus its previous value (0 if none)
Same protocol as medium/classify.py: primary variant, fixed settings, stratified group 5-fold x 5 repeats, pooled out-of-fold PR-AUC, grouped
bootstrap CIs, paired differences. Gradient boosting uses scikit-learn defaults with a fixed seed (no search). Exploratory.
Writes ONLY to medium/out/temporal/<tag>/.

    python -I medium/temporal.py --models logreg hgb --tag lr_hgb
"""

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ringtda.eval.metrics import GroupedBootstrap, mean_ap_with_ci, paired_difference
from ringtda.eval.splits import stratified_group_cv
from ringtda.models.classify import load_config, make_model, named_feature_sets, scores

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="HI")
ap.add_argument("--tau", default="0.05")
ap.add_argument("--models", nargs="+", default=["logreg", "hgb"])
ap.add_argument("--tag", default="run")
ap.add_argument("--rf-jobs", type=int, default=6)
ap.add_argument("--sets", nargs="+", default=None)
args = ap.parse_args()

HERE = Path(__file__).resolve().parent
name = f"{args.dataset}-Medium"
cfg = load_config()
cfg["models"]["rf"]["n_jobs"] = args.rf_jobs
OUT = HERE / "out" / "temporal" / args.tag
OUT.mkdir(parents=True, exist_ok=True)
t_start = time.time()
fdir = HERE / "out" / f"{args.dataset}_tau{args.tau}"

F = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(fdir / "features_window_*.csv.gz")))], ignore_index=True)
I = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(fdir / "ids_window_*.csv.gz")))], ignore_index=True)[["window_id", "candidate_id", "nodes"]]
L = pd.read_pickle(HERE / "out" / f"ibm_candidates_{name}_{float(args.tau)}.pkl")[["window_id", "cand_id", "y", "group"]].rename(columns={"cand_id": "candidate_id"})
df = F.merge(I, on=["window_id", "candidate_id"], validate="one_to_one").merge(L, on=["window_id", "candidate_id"], validate="one_to_one")
assert len(df) == len(F) == len(L)
df = df.sort_values(["window_id", "candidate_id"]).reset_index(drop=True)

# ---- causal temporal features: only rows of strictly earlier windows
TP = ["tp_seen", "tp_streak", "tp_birth_prev", "tp_birth_delta", "tp_birth_min", "tp_birth_mean", "tp_vol_delta", "tp_bal_delta", "tp_reg_delta"]
hist = {}  # ring key -> list of (window_id, birth, vol, bal, reg) for windows already processed
rows = []
for wid, g in df.groupby("window_id", sort=True):
    new = []
    for r in g.itertuples():
        h = hist.get(r.nodes, [])
        b = r.D_birth
        if h:
            pw, pb, pv, pbal, preg = h[-1]
            streak = 0
            for w_ in range(wid - 1, -1, -1):
                if any(x[0] == w_ for x in h):
                    streak += 1
                else:
                    break
            bs = [x[1] for x in h]
            rows.append((r.Index, len(h), streak, pb, b - pb, min(bs), float(np.mean(bs)), r.D_log_volume - pv, r.D_bal_mean - pbal, r.D_reg_mean - preg))
        else:
            rows.append((r.Index, 0, 0, b, 0.0, b, b, 0.0, 0.0, 0.0))
        new.append((r.nodes, (wid, b, r.D_log_volume, r.D_bal_mean, r.D_reg_mean)))
    for k, v in new:  # add this window to the history only after all of its rows are done (no same-window leakage)
        hist.setdefault(k, []).append(v)
T = pd.DataFrame(rows, columns=["Index"] + TP).set_index("Index")
df = df.join(T)
print(f"{name}: {len(df)} candidates; with history (seen before): {int((df.tp_seen > 0).sum())} ({(df.tp_seen > 0).mean():.1%}); positives with history: "
      f"{int(((df.tp_seen > 0) & (df.y == 1)).sum())} of {int((df.y == 1).sum())}", flush=True)

SETS = named_feature_sets()
SETS = {**SETS, "WID": ["window_id"], "TP": TP, "B+D+TP": SETS["B+D"] + TP, "B+D+T+TP": SETS["B+D+T"] + TP, "B+D+T_local+TP": SETS["B+D+T_local"] + TP}
USE = args.sets or ["B+D", "B+D+TP", "B+D+T", "B+D+T+TP", "B+D+T_local", "B+D+T_local+TP"]
PAIRS = [("B+D+TP", "B+D"), ("B+D+T+TP", "B+D+T"), ("B+D+T_local+TP", "B+D+T_local"), ("B+D+T", "B+D")]

sub = df[df.y.notna()].reset_index(drop=True)
y = sub.y.to_numpy(float).astype(int)
groups = sub.group.to_numpy()
n_pos, n_posg = int(y.sum()), int(pd.Series(groups[y == 1]).nunique())
runs = stratified_group_cv(groups, y.astype(float), cfg["cv"]["k"], cfg["cv"]["fallback_k"], cfg["cv"]["repeats"], cfg["cv"]["seed"])
gb = GroupedBootstrap(groups, cfg["bootstrap"]["n_boot"], cfg["bootstrap"]["seed"])
print(f"primary: rows {len(sub)}, positives {n_pos} in {n_posg} groups, prevalence {n_pos / len(y):.4f}", flush=True)


def factory(mname, seed):
    if mname == "hgb":
        return Pipeline([("scale", StandardScaler()), ("clf", HistGradientBoostingClassifier(random_state=seed))])
    return make_model(mname, cfg, seed)


def oof(X, mname, run, seed):
    out = np.full(len(y), np.nan)
    for f in np.unique(run["fold"]):
        te = run["fold"] == f
        m = factory(mname, seed).fit(X[~te], y[~te])
        out[te] = scores(m, X[te])
    return out


rows_out, boots = [], {}
for mname in args.models:
    for sname in USE:
        t0 = time.time()
        X = sub[SETS[sname]].to_numpy(float)
        sc = [oof(X, mname, runs[r], cfg["model_seed"] + r) for r in range(len(runs))]
        res = mean_ap_with_ci(y, sc, gb)
        boots[(mname, sname)] = res
        rows_out.append({"model": mname, "feature_set": sname, "n_features": len(SETS[sname]), "pr_auc": res["ap"], "ci_low": res["ci_low"], "ci_high": res["ci_high"], "n_pos_groups": n_posg, "prevalence": n_pos / len(y)})
        print(f"  {sname:15s} {mname:6s} PR-AUC {res['ap']:.3f} [{res['ci_low']:.3f}, {res['ci_high']:.3f}] ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(rows_out).to_csv(OUT / "results.csv", index=False)

P = []
for mname in args.models:
    for a, b in PAIRS:
        if (mname, a) not in boots or (mname, b) not in boots:
            continue
        d = paired_difference(boots[mname, a], boots[mname, b])
        P.append({"model": mname, "comparison": f"{a} minus {b}", **d})
P = pd.DataFrame(P)
P.to_csv(OUT / "paired_differences.csv", index=False)
yaml.safe_dump({"args": vars(args), "runtime_seconds": time.time() - t_start, "temporal_features": TP}, open(OUT / "config_used.yaml", "w"))
print("\nPaired differences (same resamples):")
for q in P.itertuples():
    print(f"  {q.model:6s} {q.comparison:30s} {q.diff:+.3f} [{q.ci_low:+.3f}, {q.ci_high:+.3f}]", flush=True)
print(f"done in {time.time() - t_start:.0f}s -> {OUT}", flush=True)
