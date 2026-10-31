"""Diagnostic of the severity result (experiments/synth_severity.py): is B+D (no volume) forecasting or recovering generator structure?

    nice -n 10 python experiments/severity_diagnostic.py

Same pairs (cached tables of the severity run), same groups, folds, repeats and bootstrap resamples. Adds: a fair persistence
baseline (y = a + b * D_log_volume fitted inside each training fold); structure-only baselines used as DIAGNOSTIC predictors,
not features (D_length; scenario one-hot taken from the generator labels; both) with ridge, random forest and gradient boosting
(fixed hyperparameters, configs/regress.yaml); random-forest impurity importance of B+D (top 10). Outputs: results/phase4_severity/diagnostic/.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ringtda.models.classify import named_feature_sets
from ringtda.models.regress import MODELS, ci, grouped_bootstrap_metrics, grouped_folds, make_model, oof_predict, reg_metrics

cfg = yaml.safe_load(open("configs/regress.yaml"))
SEV = Path("results/phase4_severity")
OUT = SEV / "diagnostic"
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
data = pd.concat([pd.read_pickle(SEV / "cache" / f"seed{s}.pkl") for s in range(cfg["data"]["seeds"])], ignore_index=True)
y = data.y_next.to_numpy(float)
groups = data.group.to_numpy()
folds = grouped_folds(groups, cfg["cv"]["k"], cfg["cv"]["repeats"], cfg["cv"]["seed"])
BD = [c for c in named_feature_sets()["B+D"] if c not in cfg["exclude_columns"]]
scen = pd.get_dummies(data.scenario).astype(float)
X_sets = {"B+D": data[BD].to_numpy(float), "D_length": data[["D_length"]].to_numpy(float), "scenario": scen.to_numpy(float),
          "D_length+scenario": np.hstack([data[["D_length"]].to_numpy(float), scen.to_numpy(float)])}  # fmt: skip


def persistence_fit(fold):
    """y = a + b * D_log_volume, fitted on the training folds only."""
    v = data["D_log_volume"].to_numpy(float)
    out = np.full(len(y), np.nan)
    for f in np.unique(fold):
        te = fold == f
        b, a = np.polyfit(v[~te], y[~te], 1)
        out[te] = a + b * v[te]
    return out


preds = {"mean": [oof_predict(None, y, f, "mean", cfg) for f in folds], "persistence_fit": [persistence_fit(f) for f in folds]}
for sname, X in X_sets.items():
    for m in MODELS:
        preds[f"{sname}|{m}"] = [oof_predict(X, y, f, m, cfg, cfg["model_seed"] + r) for r, f in enumerate(folds)]
        print(f"[{time.time() - t0:3.0f}s] {sname:18s} {m:6s} repeat-0 RMSE {reg_metrics(y, preds[f'{sname}|{m}'][0])['rmse']:.3f}", flush=True)
boot = grouped_bootstrap_metrics(y, preds, groups, cfg["bootstrap"]["n_boot"], cfg["bootstrap"]["seed"])
rows = []
for key, reps in preds.items():
    sname, _, m = key.partition("|")
    pt = [reg_metrics(y, p) for p in reps]
    r = {"predictor": sname, "model": m or "none", "n_pairs": len(y), "n_rings": len(np.unique(groups))}
    for met in ("rmse", "mae", "spearman"):
        r[met] = float(np.nanmean([x[met] for x in pt]))
        r[f"{met}_ci_low"], r[f"{met}_ci_high"] = ci(boot[key][met])
    rows.append(r)
R = pd.DataFrame(rows)
R.to_csv(OUT / "results.csv", index=False)
pairs = []
for m in MODELS:
    for ref in ("D_length", "scenario", "D_length+scenario"):
        for met in ("rmse", "mae", "spearman"):
            d = boot[f"B+D|{m}"][met] - boot[f"{ref}|{m}"][met]
            pt = R[(R.predictor == "B+D") & (R.model == m)][met].iloc[0] - R[(R.predictor == ref) & (R.model == m)][met].iloc[0]
            lo, hi = ci(d)
            pairs.append({"model": m, "comparison": f"B+D minus {ref}", "metric": met, "diff": pt, "ci_low": lo, "ci_high": hi, "ci_includes_0": bool(lo <= 0 <= hi)})
    for met in ("rmse", "mae", "spearman"):
        d = boot[f"B+D|{m}"][met] - boot["persistence_fit"][met]
        pt = R[(R.predictor == "B+D") & (R.model == m)][met].iloc[0] - R[R.predictor == "persistence_fit"][met].iloc[0]
        lo, hi = ci(d)
        pairs.append({"model": m, "comparison": "B+D minus persistence_fit", "metric": met, "diff": pt, "ci_low": lo, "ci_high": hi, "ci_includes_0": bool(lo <= 0 <= hi)})
P = pd.DataFrame(pairs)
P.to_csv(OUT / "paired_differences.csv", index=False)
rf = make_model("rf", cfg, cfg["model_seed"]).fit(X_sets["B+D"], y)
imp = pd.DataFrame({"column": BD, "rf_impurity_importance": rf.named_steps["m"].feature_importances_}).sort_values("rf_impurity_importance", ascending=False).reset_index(drop=True)
imp.to_csv(OUT / "rf_importance_BD.csv", index=False)
yaml.safe_dump({"config": cfg, "runtime_seconds": time.time() - t0, "n_pairs": len(y)}, open(OUT / "config_used.yaml", "w"))
f3 = lambda r, s: f"{r[s]:.3f} [{r[s + '_ci_low']:.3f}, {r[s + '_ci_high']:.3f}]"  # noqa: E731
md = ["# Severity diagnostic (SYNTHETIC)\n", f"{len(y)} pairs, {len(np.unique(groups))} rings; same grouped folds, repeats, model seeds and bootstrap resamples as the severity run. Structure-only predictors are diagnostics, not features (the scenario label comes from the generator).\n"]
for met in ("rmse", "mae", "spearman"):
    md.append(f"\n## {met} [95% CI]\n\n| predictor | " + " | ".join(MODELS) + " |\n|---|" + "---|" * len(MODELS))
    for s in ("B+D", "D_length", "scenario", "D_length+scenario"):
        md.append(f"| {s} | " + " | ".join(f3(R[(R.predictor == s) & (R.model == m)].iloc[0], met) for m in MODELS) + " |")
    for s in ("mean", "persistence_fit"):
        md.append(f"| {s} (no model) | {f3(R[R.predictor == s].iloc[0], met)} | | |")
md.append("\n## Paired differences, B+D minus baseline (RMSE, MAE: negative = B+D better)\n\n| model | comparison | metric | diff | 95% CI | CI includes 0 |\n|---|---|---|---|---|---|")
md += [f"| {r.model} | {r.comparison} | {r.metric} | {r.diff:+.3f} | [{r.ci_low:+.3f}, {r.ci_high:+.3f}] | {r.ci_includes_0} |" for r in P.itertuples()]
flag = P[(P.metric == "rmse") & P.ci_includes_0 & P.comparison.str.contains("D_length|scenario")]
md.append("\n## Flag\n\n" + ("Structure-only predictors whose RMSE difference to B+D has a CI including 0: " + ", ".join(f"{r.model}/{r.comparison}" for r in flag.itertuples()) if len(flag) else "No structure-only baseline matches B+D within the RMSE CI."))
md.append("\n\n## Random-forest impurity importance, B+D (no volume), top 10\n\n| rank | column | importance |\n|---|---|---|")
md += [f"| {i + 1} | {r.column} | {r.rf_impurity_importance:.3f} |" for i, r in imp.head(10).iterrows()]
(OUT / "summary.md").write_text("\n".join(md) + "\n")
print(f"done in {time.time() - t0:.0f}s -> {OUT}", flush=True)
