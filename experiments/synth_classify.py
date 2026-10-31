"""Synthetic classification runs (proposal v3 §4, NFR7). Report only; no pass/fail gate.

    nohup nice -n 10 python -u experiments/synth_classify.py > synth.log 2>&1 &

Conditions: hard_negatives {off, on} x dense_internal_chords {default True, False}; seeds 0-5 pooled into one
table per condition (public SyntheticConfig only; `report` config untouched). Stratified GROUP k-fold with
repeats, fixed hyperparameters (configs/classify.yaml), pooled out-of-fold scores, grouped bootstrap CIs, paired
differences inside the same resamples, per-seed PR-AUC (mean +- std across seeds) from the same scores, per-scenario
PR-AUC (positives of one scenario against ALL negatives; prevalence changes per scenario) and the mean
out-of-fold score rank percentile of the topology-sensitive and hard negatives.

Variants: primary (ambiguous excluded) and ambiguous_positive. The second one is limited for runtime: feature sets
B, B+D, B+D+T, B+D+T_local, T; with hard negatives on (about 59k candidates, 94% of them ambiguous) logistic
regression only.
Outputs: results/phase3_features/classify/synthetic/<condition>/ (+ cache/ of the per-seed tables).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ringtda.data.synthetic import SyntheticConfig
from ringtda.eval.metrics import GroupedBootstrap, average_precision, mean_ap_with_ci, paired_difference
from ringtda.eval.splits import stratified_group_cv
from ringtda.models.classify import MODELS, load_config, named_feature_sets, oof_scores
from ringtda.synthetic_table import synthetic_candidate_table

ap_ = argparse.ArgumentParser()
ap_.add_argument("--seeds", type=int, default=6)
ap_.add_argument("--n-boot", type=int, default=None)
ap_.add_argument("--repeats", type=int, default=None)
ap_.add_argument("--rf-trees", type=int, default=None)
ap_.add_argument("--conditions", nargs="+", default=None, help="subset of condition names")
ap_.add_argument("--variants", nargs="+", default=["primary", "ambiguous_positive"])
ap_.add_argument("--out", default="results/phase3_features/classify/synthetic")
args = ap_.parse_args()

cfg = load_config()
if args.n_boot:
    cfg["bootstrap"]["n_boot"] = args.n_boot
if args.repeats:
    cfg["cv"]["repeats"] = args.repeats
if args.rf_trees:
    cfg["models"]["rf"]["n_estimators"] = args.rf_trees
SETS = named_feature_sets()
ORDER = ["B", "B+D", "B+D+T", "B+D+T_local", "T", "T_local", "D"]
AMB_SETS = ["B", "B+D", "B+D+T", "B+D+T_local", "T"]
PAIRS = [("B+D+T", "B+D"), ("B+D+T_local", "B+D"), ("B+D", "B")]
NEG_KINDS = ["shortcut_filled", "transitive_decoy", "arbitrage_loop", "market_maker", "hot_wallet", "background"]
CONDITIONS = {
    f"hn-{'on' if hn else 'off'}_chords-{'default' if ch else 'off'}": {"hard_negatives": hn, "dense_internal_chords": ch}
    for hn in (False, True)
    for ch in (True, False)
}
OUT = Path(args.out)
(OUT / "cache").mkdir(parents=True, exist_ok=True)
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def condition_table(name, kw):
    parts = []
    for seed in range(args.seeds):
        f = OUT / "cache" / f"{name}_seed{seed}.pkl"
        if f.exists():
            parts.append(pd.read_pickle(f))
            continue
        t = time.time()
        d = synthetic_candidate_table(SyntheticConfig(seed=seed, **kw), group_offset=seed * 1_000_000)
        d.to_pickle(f)
        parts.append(d)
        log(f"{name} seed {seed}: {len(d)} candidates built in {time.time() - t:.0f}s")
    return pd.concat(parts, ignore_index=True)


def rank_percentile(s):
    return (pd.Series(s).rank(method="average").to_numpy() - 0.5) / len(s)


for cname, kw in CONDITIONS.items():
    if args.conditions and cname not in args.conditions:
        continue
    cdir = OUT / cname
    cdir.mkdir(parents=True, exist_ok=True)
    df = condition_table(cname, kw)
    for s, cols in SETS.items():
        assert not df[cols].isna().any().any(), s
    log(f"{cname}: {len(df)} candidates, kinds {df.kind.value_counts().to_dict()}")
    res_rows, pair_rows, scen_rows, scen_pair_rows, neg_rows = [], [], [], [], []
    oofs = {}
    for vname in args.variants:
        y_full = df.y.copy()
        keep = y_full.notna() if vname == "primary" else pd.Series(True, index=df.index)
        if vname == "ambiguous_positive":
            y_full = y_full.fillna(1.0)
        sub = df[keep].reset_index(drop=True)
        y = y_full[keep].to_numpy(float).astype(int)
        groups = sub.group.to_numpy()
        n_pos, n_posg = int(y.sum()), int(pd.Series(groups[y == 1]).nunique())
        runs = stratified_group_cv(groups, y.astype(float), cfg["cv"]["k"], cfg["cv"]["fallback_k"], cfg["cv"]["repeats"], cfg["cv"]["seed"])
        gb = GroupedBootstrap(groups, cfg["bootstrap"]["n_boot"], cfg["bootstrap"]["seed"])
        sets_here = ORDER if vname == "primary" else AMB_SETS
        models_here = list(MODELS) if (vname == "primary" or not kw["hard_negatives"]) else ["logreg"]
        log(f"[{cname}/{vname}] rows {len(sub)}, positives {n_pos} ({n_pos / len(sub):.4f}) in {n_posg} groups, k {[r['k'] for r in runs]}, models {models_here}")
        boots, scores = {}, {}

        def record(sname, mname, sc, n_feat):
            res = mean_ap_with_ci(y, sc, gb)
            boots[sname, mname], scores[sname, mname] = res, sc
            per_seed = []
            for sd in sorted(sub.seed.unique()):
                m = (sub.seed == sd).to_numpy()
                per_seed.append(np.nanmean([average_precision(y[m], s[m]) for s in sc]) if y[m].sum() else np.nan)
            res_rows.append({"condition": cname, "variant": vname, "feature_set": sname, "model": mname, "n_features": n_feat, "pr_auc": res["ap"], "ci_low": res["ci_low"], "ci_high": res["ci_high"],
                             "prevalence": n_pos / len(y), "n_rows": len(y), "n_pos": n_pos, "n_pos_groups": n_posg, "seed_mean": np.nanmean(per_seed), "seed_std": np.nanstd(per_seed, ddof=1),
                             "n_seeds": int(np.isfinite(per_seed).sum()), "per_seed": ",".join(f"{v:.3f}" for v in per_seed), "pr_auc_by_repeat": ",".join(f"{a:.4f}" for a in res["ap_by_repeat"]),
                             "repeats": len(sc), "n_boot": gb.n_boot, "n_boot_nan": res["n_boot_nan"], "cv_seed": cfg["cv"]["seed"], "boot_seed": cfg["bootstrap"]["seed"]})  # fmt: skip

        for sname in sets_here:
            X = sub[SETS[sname]].to_numpy(float)
            for mname in models_here:
                t = time.time()
                sc = [oof_scores(X, y, groups, runs[r]["fold"], mname, cfg, cfg["model_seed"] + r) for r in range(len(runs))]
                oofs[f"{vname}|{mname}|{sname}"] = np.stack(sc)
                record(sname, mname, sc, len(SETS[sname]))
                r_ = res_rows[-1]
                log(f"  {vname:18s} {sname:12s} {mname:6s} PR-AUC {r_['pr_auc']:.3f} [{r_['ci_low']:.3f}, {r_['ci_high']:.3f}] seeds {r_['seed_mean']:.3f}+-{r_['seed_std']:.3f} ({time.time() - t:.0f}s)")
        if vname == "primary":
            record("D_birth_alone", "none", [-sub["D_birth"].to_numpy(float)], 1)
        for mname in models_here:
            for a, b in PAIRS:
                if (a, mname) in boots and (b, mname) in boots:
                    pair_rows.append({"condition": cname, "variant": vname, "model": mname, "comparison": f"{a} minus {b}", "scope": "overall", **{k: v for k, v in paired_difference(boots[a, mname], boots[b, mname]).items()}, "n_boot": gb.n_boot})  # fmt: skip

        if vname != "primary":
            continue
        # ---- per scenario: positives of one scenario against ALL negatives (prevalence changes)
        for scn in sorted(sub.loc[y == 1, "scenario"].unique()):
            m = ((y == 0) | ((y == 1) & (sub.scenario == scn).to_numpy()))
            ys, gs = y[m], groups[m]
            gbs = GroupedBootstrap(gs, cfg["bootstrap"]["n_boot"], cfg["bootstrap"]["seed"])
            nps, npg = int(ys.sum()), int(pd.Series(gs[ys == 1]).nunique())
            sb = {}
            for (sname, mname), sc in scores.items():
                if mname == "none":
                    continue
                sb[sname, mname] = mean_ap_with_ci(ys, [s[m] for s in sc], gbs)
                scen_rows.append({"condition": cname, "scenario": scn, "feature_set": sname, "model": mname, "pr_auc": sb[sname, mname]["ap"], "ci_low": sb[sname, mname]["ci_low"], "ci_high": sb[sname, mname]["ci_high"],
                                  "prevalence": nps / len(ys), "n_rows": len(ys), "n_pos": nps, "n_pos_groups": npg, "n_boot": gbs.n_boot})  # fmt: skip
            for mname in models_here:
                for a, b in PAIRS:
                    scen_pair_rows.append({"condition": cname, "scenario": scn, "model": mname, "comparison": f"{a} minus {b}", **paired_difference(sb[a, mname], sb[b, mname]), "n_pos": nps, "n_pos_groups": npg})
        # ---- rank percentile of the negatives (higher = looks more like a wash ring), mean over repeats
        kinds = sub.kind.to_numpy()
        for (sname, mname), sc in scores.items():
            if mname == "none":
                continue
            pct = np.mean([rank_percentile(s) for s in sc], axis=0)
            for k in ["wash_ring"] + NEG_KINDS:
                mk = kinds == k
                if mk.any():
                    neg_rows.append({"condition": cname, "feature_set": sname, "model": mname, "kind": k, "n": int(mk.sum()), "mean_rank_percentile": float(pct[mk].mean())})

    # ---- dense positives that keep a label although the ring is filled (T_local_ring_inf = 0)
    pos = df[df.y == 1]
    dense_rows = []
    for scn in sorted(pos.scenario.unique()):
        p = pos[pos.scenario == scn]
        planted = p.groupby(["seed", "pattern_id"]).T_local_ring_inf.max()
        dense_rows.append({"condition": cname, "scenario": scn, "positive_candidates": len(p), "with_ring_inf_0": int((p.T_local_ring_inf == 0).sum()), "frac_ring_inf_0": float((p.T_local_ring_inf == 0).mean()),
                           "planted_rings_with_candidate": len(planted), "planted_rings_with_ring_inf_1": int((planted == 1).sum())})  # fmt: skip

    R, P = pd.DataFrame(res_rows), pd.DataFrame(pair_rows)
    R.to_csv(cdir / "results.csv", index=False)
    P.to_csv(cdir / "paired_differences.csv", index=False)
    pd.DataFrame(scen_rows).to_csv(cdir / "per_scenario.csv", index=False)
    pd.DataFrame(scen_pair_rows).to_csv(cdir / "per_scenario_paired.csv", index=False)
    pd.DataFrame(neg_rows).to_csv(cdir / "negatives_rank.csv", index=False)
    pd.DataFrame(dense_rows).to_csv(cdir / "positives_ring_inf.csv", index=False)
    np.savez_compressed(cdir / "oof_scores.npz", **oofs)
    yaml.safe_dump({"config": cfg, "args": vars(args), "condition": kw, "seeds": list(range(args.seeds)), "n_candidates": len(df), "runtime_seconds": time.time() - T0}, open(cdir / "config_used.yaml", "w"))
    # ---- markdown summary: numbers only
    f3 = lambda r: f"{r.pr_auc:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]"  # noqa: E731
    md = [f"# Synthetic classification: {cname}\n", f"seeds 0-{args.seeds - 1} pooled; fixed hyperparameters; {cfg['cv']['repeats']} CV repeats; {cfg['bootstrap']['n_boot']} grouped-bootstrap resamples. Nothing is a pass/fail gate.\n"]  # fmt: skip
    for vname in args.variants:
        Rv = R[(R.variant == vname)]
        r0 = Rv.iloc[0]
        md.append(f"\n## {vname}: {r0.n_rows} candidates, {r0.n_pos} positives ({r0.prevalence:.4f}) in {r0.n_pos_groups} groups\n")
        md.append("PR-AUC [95% CI] (pooled out-of-fold) and, in brackets, mean +- std of the per-seed PR-AUC:\n\n| feature set | " + " | ".join(MODELS) + " |\n|---|" + "---|" * len(MODELS))
        cell = {(r.feature_set, r.model): f"{f3(r)} ({r.seed_mean:.3f} +- {r.seed_std:.3f})" for r in Rv.itertuples() if r.model != "none"}
        md += [f"| {s} | " + " | ".join(cell.get((s, m), "") for m in MODELS) + " |" for s in ORDER if (s, "logreg") in cell]
        if vname == "primary":
            rr = Rv[Rv.model == "none"].iloc[0]
            md.append(f"\n-D_birth alone (no model): {f3(rr)} ({rr.seed_mean:.3f} +- {rr.seed_std:.3f})")
        md.append("\nPaired differences (same resamples):\n\n| model | comparison | diff | 95% CI | share > 0 |\n|---|---|---|---|---|")
        md += [f"| {q.model} | {q.comparison} | {q.diff:+.3f} | [{q.ci_low:+.3f}, {q.ci_high:+.3f}] | {q.share_boot_positive:.2f} |" for q in P[P.variant == vname].itertuples()]
    S_ = pd.DataFrame(scen_rows)
    if len(S_):
        md.append("\n## per scenario (primary): positives of one scenario against all negatives\n")
        for scn, g in S_.groupby("scenario"):
            g0 = g.iloc[0]
            md.append(f"\n### {scn}: {g0.n_pos} positives ({g0.prevalence:.4f}) in {g0.n_pos_groups} groups\n\n| feature set | " + " | ".join(MODELS) + " |\n|---|" + "---|" * len(MODELS))
            c2 = {(r.feature_set, r.model): f3(r) for r in g.itertuples()}
            md += [f"| {s} | " + " | ".join(c2.get((s, m), "") for m in MODELS) + " |" for s in ORDER]
            sp = pd.DataFrame(scen_pair_rows)
            sp = sp[sp.scenario == scn]
            md.append("\n| model | comparison | diff | 95% CI |\n|---|---|---|---|")
            md += [f"| {q.model} | {q.comparison} | {q.diff:+.3f} | [{q.ci_low:+.3f}, {q.ci_high:+.3f}] |" for q in sp.itertuples()]
    N_ = pd.DataFrame(neg_rows)
    if len(N_):
        md.append("\n## mean out-of-fold score rank percentile by candidate kind (primary; 1 = top of the ranking)\n")
        for mname in MODELS:
            if mname in set(N_.model):
                t = N_[N_.model == mname].pivot(index="feature_set", columns="kind", values="mean_rank_percentile").reindex([s for s in ORDER if s in set(N_.feature_set)])
                md.append(f"\n{mname}:\n\n" + t.round(3).to_string() + "\n")
    md.append("\n## positives with T_local_ring_inf = 0 (they keep the positive label)\n\n" + pd.DataFrame(dense_rows).to_string(index=False) + "\n")
    (cdir / "summary.md").write_text("\n".join(md) + "\n")
    log(f"{cname} done -> {cdir}")
log("all done")
