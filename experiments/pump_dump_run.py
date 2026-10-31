"""Phase 7 optional track: synthetic pump-and-dump classification (v3 §3.6). SYNTHETIC ONLY, NO TOPOLOGY. Exploratory; nothing is a gate.

    nice -n 10 python experiments/pump_dump_run.py

The generator author decides how separable the classes are (src/ringtda/data/pump_dump.py). Reuses the classification harness unchanged:
stratified_group_cv, oof_scores, GroupedBootstrap, mean_ap_with_ci, paired_difference, with the fixed hyperparameters of configs/classify.yaml.
Feature sets: M (temporal-motif features) and B_pd (group size and per-account volume only). Conditions (configs/pump_dump.yaml): default and
harder_exit (wider spread of planted exit times). Outputs: results/phase7_pump_dump/.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ringtda.data.pump_dump import PumpDumpConfig
from ringtda.eval.metrics import GroupedBootstrap, average_precision, mean_ap_with_ci, paired_difference
from ringtda.eval.splits import stratified_group_cv
from ringtda.models.classify import MODELS, load_config, oof_scores
from ringtda.motif.pump_dump import BPD_NAMES, M_NAMES, pump_dump_table

pc = yaml.safe_load(open("configs/pump_dump.yaml"))
cfg = load_config()
OUT = Path("results/phase7_pump_dump")
(OUT / "cache").mkdir(parents=True, exist_ok=True)
t0 = time.time()
SETS = {"M": M_NAMES, "B_pd": BPD_NAMES}
rows, pairs, counts = [], [], []
for cname, cond in pc["conditions"].items():
    parts = []
    for seed in range(pc["seeds"]):
        f = OUT / "cache" / f"{cname}_seed{seed}.pkl"
        if f.exists():
            parts.append(pd.read_pickle(f))
            continue
        gcfg = PumpDumpConfig(seed=seed, **pc["generator"], **cond)
        d = pump_dump_table(gcfg, pc["candidates"], pc["label_x"], group_offset=seed * 1_000_000)
        d.to_pickle(f)
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    counts.append({"condition": cname, "candidates": len(df), **df.kind.value_counts().to_dict()})
    sub = df[df.y.notna()].reset_index(drop=True)
    y = sub.y.to_numpy(float).astype(int)
    groups = sub.group.to_numpy()
    n_pos, n_posg = int(y.sum()), int(pd.Series(groups[y == 1]).nunique())
    runs = stratified_group_cv(groups, y.astype(float), cfg["cv"]["k"], cfg["cv"]["fallback_k"], cfg["cv"]["repeats"], cfg["cv"]["seed"])
    gb = GroupedBootstrap(groups, cfg["bootstrap"]["n_boot"], cfg["bootstrap"]["seed"])
    print(f"[{time.time() - t0:4.0f}s] {cname}: {len(sub)} candidates, {n_pos} positives (prevalence {n_pos / len(y):.3f}) in {n_posg} groups; kinds {counts[-1]}", flush=True)
    boots = {}

    def record(sname, mname, sc):
        res = mean_ap_with_ci(y, sc, gb)
        boots[sname, mname] = res
        per_seed = [np.nanmean([average_precision(y[m], s[m]) for s in sc]) if y[m].sum() else np.nan for m in ((sub.seed == sd).to_numpy() for sd in sorted(sub.seed.unique()))]
        rows.append({"condition": cname, "feature_set": sname, "model": mname, "pr_auc": res["ap"], "ci_low": res["ci_low"], "ci_high": res["ci_high"], "prevalence": n_pos / len(y), "n_rows": len(y), "n_pos": n_pos,
                     "n_pos_groups": n_posg, "seed_mean": float(np.nanmean(per_seed)), "seed_std": float(np.nanstd(per_seed, ddof=1)), "per_seed": ",".join(f"{v:.3f}" for v in per_seed), "n_boot": gb.n_boot,
                     "exit_spread_sd_h": cond["exit_spread_sd_h"]})  # fmt: skip
        print(f"  {cname:12s} {sname:5s} {mname:6s} PR-AUC {res['ap']:.3f} [{res['ci_low']:.3f}, {res['ci_high']:.3f}]", flush=True)

    for sname, cols in SETS.items():
        X = sub[cols].to_numpy(float)
        for mname in MODELS:
            record(sname, mname, [oof_scores(X, y, groups, runs[r]["fold"], mname, cfg, cfg["model_seed"] + r) for r in range(len(runs))])
    record("exit_spread_alone", "none", [-sub["M_exit_spread_h"].to_numpy(float)])  # single-feature reference, no model
    for mname in MODELS:
        d = paired_difference(boots["M", mname], boots["B_pd", mname])
        pairs.append({"condition": cname, "model": mname, "comparison": "M minus B_pd", **{k: d[k] for k in ("diff", "ci_low", "ci_high", "share_boot_positive")}, "n_boot": gb.n_boot})

R, P = pd.DataFrame(rows), pd.DataFrame(pairs)
R.to_csv(OUT / "results.csv", index=False)
P.to_csv(OUT / "paired_differences.csv", index=False)
pd.DataFrame(counts).to_csv(OUT / "candidate_kinds.csv", index=False)
yaml.safe_dump({"pump_dump_config": pc, "classify_config": cfg, "runtime_seconds": time.time() - t0}, open(OUT / "config_used.yaml", "w"))
f3 = lambda r: f"{r.pr_auc:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}] ({r.seed_mean:.3f} +- {r.seed_std:.3f})"  # noqa: E731
md = ["# Phase 7: synthetic pump-and-dump classification (SYNTHETIC ONLY, NO TOPOLOGY)\n",
      "The generator author decides separability. Cells: pooled out-of-fold PR-AUC [95% grouped-bootstrap CI] (mean +- std of the per-seed PR-AUC over "
      f"{pc['seeds']} seeds); fixed hyperparameters of configs/classify.yaml, grouped stratified CV ({cfg['cv']['repeats']} repeats), {cfg['bootstrap']['n_boot']} resamples. Nothing is a gate.\n",
      "Candidate kinds per condition (all seeds, ambiguous included):\n\n```\n" + pd.DataFrame(counts).fillna(0).to_string(index=False) + "\n```\n"]  # fmt: skip
for cname in pc["conditions"]:
    g = R[R.condition == cname]
    r0 = g.iloc[0]
    md.append(f"\n## {cname} (exit-time spread sd {r0.exit_spread_sd_h} h): {r0.n_rows} candidates, {r0.n_pos} positives (prevalence {r0.prevalence:.3f}) in {r0.n_pos_groups} groups\n\n| feature set | " + " | ".join(MODELS) + " |\n|---|" + "---|" * len(MODELS))
    for s in SETS:
        md.append(f"| {s} | " + " | ".join(f3(g[(g.feature_set == s) & (g.model == m)].iloc[0]) for m in MODELS) + " |")
    md.append(f"\n-exit spread alone (no model): {f3(g[g.feature_set == 'exit_spread_alone'].iloc[0])}\n")
    md.append("Paired M minus B_pd (same resamples):\n\n| model | diff | 95% CI | share > 0 |\n|---|---|---|---|")
    md += [f"| {q.model} | {q.diff:+.3f} | [{q.ci_low:+.3f}, {q.ci_high:+.3f}] | {q.share_boot_positive:.2f} |" for q in P[P.condition == cname].itertuples()]
md.append("\n## Change from default to harder_exit (same candidates, labels and groups in both conditions; difference of point estimates, no CI)\n\n| feature set | model | default | harder_exit | change |\n|---|---|---|---|---|")
for s in SETS:
    for m in MODELS:
        a, b = (R[(R.condition == c) & (R.feature_set == s) & (R.model == m)].pr_auc.iloc[0] for c in ("default", "harder_exit"))
        md.append(f"| {s} | {m} | {a:.3f} | {b:.3f} | {b - a:+.3f} |")
(OUT / "summary.md").write_text("\n".join(md) + "\n")
print(f"done in {time.time() - t0:.0f}s -> {OUT}", flush=True)
