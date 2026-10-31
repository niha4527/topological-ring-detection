"""Phase 4: next-window severity regression on SYNTHETIC rings (v3 §3.5). Exploratory; nothing is a gate.

    nice -n 10 python experiments/synth_severity.py

IBM AML has no next-window wash volume, so this is synthetic-only: the generator decides that a planted ring persists
(its rounds run all week). Windows are consecutive days of one generated week; a ring's window-t candidate is matched to
the same ring in window t+1 by account set. Target log1p(wash volume in t+1). Columns that encode the ring's own
current volume or activity are excluded from every feature set (configs/regress.yaml). Grouped CV by ring identity,
repeats, pooled out-of-fold predictions, grouped-bootstrap CIs, paired differences in the same resamples.
Outputs: results/phase4_severity/.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ringtda.data.synthetic import SyntheticConfig
from ringtda.models.classify import named_feature_sets
from ringtda.models.regress import MODELS, assert_no_excluded, ci, grouped_bootstrap_metrics, grouped_folds, oof_predict, reg_metrics, severity_pairs

cfg = yaml.safe_load(open("configs/regress.yaml"))
OUT = Path("results/phase4_severity")
(OUT / "cache").mkdir(parents=True, exist_ok=True)
t0 = time.time()
D = cfg["data"]
parts = []
for seed in range(D["seeds"]):
    f = OUT / "cache" / f"seed{seed}.pkl"
    if f.exists():
        parts.append(pd.read_pickle(f))
        continue
    df = severity_pairs(SyntheticConfig(seed=seed, hard_negatives=False), D["window_length_days"] * 86400.0, D["n_windows"], D["tau_quantile"], D["max_len"], group_offset=seed * 1_000_000)
    df.to_pickle(f)
    parts.append(df)
    print(f"[{time.time() - t0:4.0f}s] seed {seed}: {len(df)} pairs, {df.group.nunique()} rings", flush=True)
data = pd.concat(parts, ignore_index=True)
y = data.y_next.to_numpy(float)
groups = data.group.to_numpy()
print(f"population: {len(data)} (candidate, window) pairs, {len(np.unique(groups))} rings, by scenario {data.groupby('scenario').group.agg(['size', 'nunique']).to_dict('index')}", flush=True)

SETS = {k: [c for c in v if c not in cfg["exclude_columns"]] for k, v in named_feature_sets().items() if k in cfg["feature_sets"]}
for k, v in SETS.items():
    assert_no_excluded(v, cfg["exclude_columns"])
folds = grouped_folds(groups, cfg["cv"]["k"], cfg["cv"]["repeats"], cfg["cv"]["seed"])
preds = {"mean": [oof_predict(None, y, f, "mean", cfg) for f in folds]}
for sname, cols in SETS.items():
    X = data[cols].to_numpy(float)
    for m in MODELS:
        t = time.time()
        preds[f"{sname}|{m}"] = [oof_predict(X, y, f, m, cfg, cfg["model_seed"] + r) for r, f in enumerate(folds)]
        mm = reg_metrics(y, preds[f"{sname}|{m}"][0])
        print(f"  {sname:8s} {m:6s} repeat-0 RMSE {mm['rmse']:.3f} Spearman {mm['spearman']:.3f} ({time.time() - t:.0f}s)", flush=True)
preds["persistence_ref"] = [data["D_log_volume"].to_numpy(float)]  # reference only: the excluded column used as the predictor
boot = grouped_bootstrap_metrics(y, preds, groups, cfg["bootstrap"]["n_boot"], cfg["bootstrap"]["seed"])
rows = []
for key, reps in preds.items():
    sname, _, m = key.partition("|")
    pt = [reg_metrics(y, p) for p in reps]
    r = {"feature_set": sname if m else key, "model": m or "none", "n_pairs": len(y), "n_rings": len(np.unique(groups)), "n_features": len(SETS[sname]) if m else 0}
    for met in ("rmse", "mae", "spearman"):
        r[met] = float(np.nanmean([x[met] for x in pt]))
        r[f"{met}_ci_low"], r[f"{met}_ci_high"] = ci(boot[key][met])
    rows.append(r)
R = pd.DataFrame(rows)
R.to_csv(OUT / "results.csv", index=False)
pairs = []
for m in MODELS:
    for a, b in (("B+D+T", "B+D"), ("B+D", "B"), ("B+D+T", "B")):
        for met in ("rmse", "mae", "spearman"):
            d = boot[f"{a}|{m}"][met] - boot[f"{b}|{m}"][met]
            pt = R[(R.feature_set == a) & (R.model == m)][met].iloc[0] - R[(R.feature_set == b) & (R.model == m)][met].iloc[0]
            lo, hi = ci(d)
            pairs.append({"model": m, "comparison": f"{a} minus {b}", "metric": met, "diff": pt, "ci_low": lo, "ci_high": hi})
P = pd.DataFrame(pairs)
P.to_csv(OUT / "paired_differences.csv", index=False)
sc = data.groupby("scenario").agg(pairs=("group", "size"), rings=("group", "nunique")).reset_index()
sc.to_csv(OUT / "population_by_scenario.csv", index=False)
yaml.safe_dump({"config": cfg, "runtime_seconds": time.time() - t0, "target_mean": float(y.mean()), "target_std": float(y.std()), "excluded": cfg["exclude_columns"]}, open(OUT / "config_used.yaml", "w"))
f3 = lambda r, s: f"{r[s]:.3f} [{r[s + '_ci_low']:.3f}, {r[s + '_ci_high']:.3f}]"  # noqa: E731
md = ["# Phase 4: next-window severity regression (SYNTHETIC only)\n",
      f"{len(data)} (candidate, window) pairs from {len(np.unique(groups))} planted rings ({D['seeds']} seeds, hard negatives off, {D['n_windows']} one-day windows of one week; ring present in windows t and t+1, matched by account set). "
      f"Target log1p(wash volume in t+1): mean {y.mean():.3f}, std {y.std():.3f}. The generator decides persistence (rings run all week). Excluded columns: {', '.join(cfg['exclude_columns'])}. "
      f"Grouped CV by ring ({cfg['cv']['k']} folds, {cfg['cv']['repeats']} repeats), pooled out-of-fold, grouped-bootstrap 95% CI ({cfg['bootstrap']['n_boot']} resamples over rings). RMSE and MAE in log1p units (lower better); Spearman (higher better).\n",
      "Population by scenario:\n\n```\n" + sc.to_string(index=False) + "\n```\n"]  # fmt: skip
for met, lab in (("rmse", "RMSE"), ("mae", "MAE"), ("spearman", "Spearman")):
    md.append(f"\n## {lab} [95% CI]\n\n| feature set | " + " | ".join(MODELS) + " |\n|---|" + "---|" * len(MODELS))
    for s in SETS:
        md.append(f"| {s} | " + " | ".join(f3(R[(R.feature_set == s) & (R.model == m)].iloc[0], met) for m in MODELS) + " |")
    md.append(f"\nmean predictor: {f3(R[R.feature_set == 'mean'].iloc[0], met)}; persistence reference (excluded D_log_volume as the predictor): {f3(R[R.feature_set == 'persistence_ref'].iloc[0], met)}")
md.append("\n## Paired differences (same resamples; RMSE and MAE: negative = first is better)\n\n| model | comparison | metric | diff | 95% CI |\n|---|---|---|---|---|")
md += [f"| {r.model} | {r.comparison} | {r.metric} | {r.diff:+.3f} | [{r.ci_low:+.3f}, {r.ci_high:+.3f}] |" for r in P.itertuples()]
(OUT / "summary.md").write_text("\n".join(md) + "\n")
print(f"done in {time.time() - t0:.0f}s -> {OUT}", flush=True)
