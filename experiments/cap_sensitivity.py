"""Infinite-bar cap sensitivity of the T features (v3 §4.7). Report only; nothing is a gate.

    nice -n 10 python experiments/cap_sensitivity.py --source synthetic
    nice -n 10 python experiments/cap_sensitivity.py --source ibm

For each cap variant (configs/cap_sensitivity.yaml) the T columns are recomputed on the same windows and candidates; the
persistence diagrams do not depend on the cap, so they are computed once per window / candidate and vectorised under
every variant (same code path as features._feature_table's T part, checked: the window_max variant must reproduce the
stored T columns). B and D columns and labels come from the stored tables, unchanged. B+D does not depend on the cap.
The stored classification protocol is rerun (fixed hyperparameters, same folds and bootstrap resamples).
Outputs: results/phase3_features/cap_sensitivity/<source>/.
"""

import argparse
import copy
import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ringtda.eval.metrics import GroupedBootstrap, mean_ap_with_ci, paired_difference
from ringtda.eval.splits import stratified_group_cv
from ringtda.features import window_graphs
from ringtda.models.classify import load_config as load_clf, named_feature_sets, oof_scores
from ringtda.topology.candidates import candidate_rings
from ringtda.topology.persistence import directed_flag_persistence
from ringtda.topology.vectorize import load_config as load_topo, local_diagrams, local_graph, ring_has_infinite_bar, ring_survives, vectorize_diagrams

ap_ = argparse.ArgumentParser()
ap_.add_argument("--source", choices=("synthetic", "ibm"), required=True)
ap_.add_argument("--repeats", type=int, default=None)
args = ap_.parse_args()
cc = yaml.safe_load(open("configs/cap_sensitivity.yaml"))
clf = load_clf()
if args.repeats:
    clf["cv"]["repeats"] = args.repeats
SETS = named_feature_sets()
OUT = Path("results/phase3_features/cap_sensitivity") / args.source
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
TOPO = load_topo()
TL, TW = [c for c in SETS["T"] if c.startswith("T_local_")], [c for c in SETS["T"] if c.startswith("T_win_")]
TCOLS = TL + TW
VARS = cc["variants"]


def vcfg(v):
    c = copy.deepcopy(TOPO)
    c["cap"] = {**c["cap"], **{k: x for k, x in VARS[v].items() if k != "local"}}
    return c


CFG = {v: vcfg(v) for v in VARS}


def t_variants(tx, tau_quantile, max_len, n_expected=None):
    """{variant: T dataframe (candidate order of candidate_rings)} for one window."""
    _, G, Gs, tau = window_graphs(tx, tau_quantile)
    res = candidate_rings(G, tau=tau, max_len=max_len)
    if not res.candidates:
        return {v: pd.DataFrame(columns=["candidate_id"] + TCOLS) for v in VARS}
    wmax = float(max(d["weight"] for _, _, d in G.edges(data=True)))
    backend = TOPO["local"]["backend"]
    win_dgms = directed_flag_persistence(Gs, backend=backend)
    win = {v: vectorize_diagrams(win_dgms, wmax, CFG[v], "T_win") for v in VARS}
    LG = local_graph(G, Gs, TOPO)
    lim = TOPO["local"]["ring_inf_limits"]
    rows = {v: [] for v in VARS}
    for cid, c in enumerate(res.candidates):
        H, dgms = local_diagrams(LG, c, TOPO, backend)
        ring_inf = ring_survives(H, c.edges, lim["max_edges"], lim["max_triangles"])
        if ring_inf is None:
            ring_inf = ring_has_infinite_bar(dgms[1], c.birth)
        wl = max((d["weight"] for _, _, d in H.edges(data=True)), default=wmax)
        for v in VARS:
            loc = vectorize_diagrams(dgms, wl if VARS[v].get("local") else wmax, CFG[v], "T_local", ring_inf=ring_inf)
            rows[v].append({"candidate_id": cid, **loc, **win[v]})
    return {v: pd.DataFrame(r) for v, r in rows.items()}


rows, pairs, repro = [], [], []


def evaluate(name, df, tables):
    keep = df.y.notna().to_numpy()
    sub = df[keep].reset_index(drop=True)
    y = sub.y.to_numpy(float).astype(int)
    groups = sub.group.to_numpy()
    npg = len(np.unique(groups[y == 1]))
    runs = stratified_group_cv(groups, y.astype(float), clf["cv"]["k"], clf["cv"]["fallback_k"], clf["cv"]["repeats"], clf["cv"]["seed"])
    gb = GroupedBootstrap(groups, clf["bootstrap"]["n_boot"], clf["bootstrap"]["seed"])
    print(f"[{name}] rows {len(sub)}, positives {y.sum()} in {npg} groups", flush=True)
    boots = {}

    def run(sname, v, X, m):
        t = time.time()
        sc = [oof_scores(X, y, groups, runs[r]["fold"], m, clf, clf["model_seed"] + r) for r in range(len(runs))]
        res = mean_ap_with_ci(y, sc, gb)
        boots[sname, m, v] = res
        rows.append({"source": name, "variant": v, "feature_set": sname, "model": m, "pr_auc": res["ap"], "ci_low": res["ci_low"], "ci_high": res["ci_high"], "n_rows": len(y), "n_pos": int(y.sum()), "n_pos_groups": npg})  # fmt: skip
        print(f"  {name} {v:15s} {sname:12s} {m:6s} PR-AUC {res['ap']:.3f} [{res['ci_low']:.3f}, {res['ci_high']:.3f}] ({time.time() - t:.0f}s)", flush=True)

    for m in cc["models"]:
        run("B+D", "(cap-free)", sub[SETS["B+D"]].to_numpy(float), m)
    for v, T in tables.items():
        Tk = T[keep].reset_index(drop=True)
        for sname, tcols in (("B+D+T", TCOLS), ("B+D+T_local", TL)):
            X = np.hstack([sub[SETS["B+D"]].to_numpy(float), Tk[tcols].to_numpy(float)])
            for m in cc["models"]:
                run(sname, v, X, m)
    for v in tables:
        for sname in cc["sets"]:
            for m in cc["models"]:
                d = paired_difference(boots[sname, m, v], boots["B+D", m, "(cap-free)"])
                pairs.append({"source": name, "variant": v, "feature_set": sname, "model": m, "comparison": f"{sname} minus B+D", **{k: d[k] for k in ("diff", "ci_low", "ci_high", "share_boot_positive")}})
                if v != "window_max":
                    d2 = paired_difference(boots[sname, m, v], boots[sname, m, "window_max"])
                    pairs.append({"source": name, "variant": v, "feature_set": sname, "model": m, "comparison": f"{sname} under this cap minus {sname} under window_max", **{k: d2[k] for k in ("diff", "ci_low", "ci_high", "share_boot_positive")}})  # fmt: skip


if args.source == "synthetic":
    from ringtda.data.synthetic import SyntheticConfig, generate

    s = cc["sources"]["synthetic"]
    cond = s["condition"]
    parts, tabs = [], {v: [] for v in VARS}
    for seed in range(s["seeds"]):
        d = pd.read_pickle(f"results/phase3_features/classify/synthetic/cache/{cond}_seed{seed}.pkl").reset_index(drop=True)
        tx, _ = generate(SyntheticConfig(seed=seed, hard_negatives=False, dense_internal_chords=cond.endswith("default")))
        tv = t_variants(tx, TOPO["candidates"]["tau_quantile"], TOPO["candidates"]["max_len"])
        for v in VARS:
            assert len(tv[v]) == len(d) and (tv[v].candidate_id.to_numpy() == d.candidate_id.to_numpy()).all(), "candidate alignment"
            tabs[v].append(tv[v][TCOLS].reset_index(drop=True))
        parts.append(d)
        print(f"[{time.time() - t0:4.0f}s] {cond} seed {seed}: {len(d)} candidates", flush=True)
    df = pd.concat(parts, ignore_index=True)
    tables = {v: pd.concat(t, ignore_index=True) for v, t in tabs.items()}
    name = cond
else:
    from ringtda.data.ibm_aml import iter_ibm_windows, load_config as load_ibm_cfg, load_ibm

    ic, s = load_ibm_cfg(), cc["sources"]["ibm"]
    ds, tau = s["dataset"], s["tau"]
    fdir = Path("results/phase3_features/ibm") / f"{ds}_tau{tau}"
    F = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(fdir / "features_window_*.csv.gz")))], ignore_index=True)
    L = pd.read_pickle(f"results/phase3_features/ibm_candidates_{ds}_{float(tau)}.pkl")[["window_id", "cand_id", "kind", "y", "group"]].rename(columns={"cand_id": "candidate_id"})
    df = F.merge(L, on=["window_id", "candidate_id"], validate="one_to_one").reset_index(drop=True)
    data = load_ibm(ds)
    tabs = {v: [] for v in VARS}
    for wid, start, _, w in iter_ibm_windows(data, ic["window"]["length_days"], ic["window"]["stride_days"]):
        n_w = int((F.window_id == wid).sum())
        if n_w == 0:
            continue
        tv = t_variants(w, float(tau), ic["candidates"]["max_len"])
        for v in VARS:
            assert len(tv[v]) == n_w, "candidate alignment"
            tv[v]["window_id"] = wid
            tabs[v].append(tv[v])
        print(f"[{time.time() - t0:4.0f}s] {ds} window {wid}: {n_w} candidates", flush=True)
    key = df[["window_id", "candidate_id"]]
    tables = {v: key.merge(pd.concat(t, ignore_index=True), on=["window_id", "candidate_id"], how="left", validate="one_to_one")[TCOLS] for v, t in tabs.items()}
    name = ds
repro.append({"source": name, "max_abs_diff_window_max_vs_stored_T": float(np.abs(tables["window_max"].to_numpy(float) - df[TCOLS].to_numpy(float)).max()),
              **{f"max_abs_diff_{v}_vs_window_max": float(np.abs(tables[v].to_numpy(float) - tables["window_max"].to_numpy(float)).max()) for v in VARS if v != "window_max"}})  # fmt: skip
evaluate(name, df, tables)

R, P = pd.DataFrame(rows), pd.DataFrame(pairs)
R.to_csv(OUT / "results.csv", index=False)
P.to_csv(OUT / "paired_differences.csv", index=False)
pd.DataFrame(repro).to_csv(OUT / "reproduction_check.csv", index=False)
yaml.safe_dump({"cap_config": cc, "classify_config": clf, "runtime_seconds": time.time() - t0}, open(OUT / "config_used.yaml", "w"))
f3 = lambda r: f"{r.pr_auc:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]"  # noqa: E731
r0 = R.iloc[0]
md = [f"# Infinite-bar cap sensitivity ({name})\n", f"{r0.n_rows} candidates, {r0.n_pos} positives in {r0.n_pos_groups} groups (primary variant). T columns recomputed per cap variant; PR-AUC [95% grouped-bootstrap CI], fixed hyperparameters, same folds and resamples for every variant. "
      "B+D rows do not depend on the cap. The window_max variant reproduces the stored T columns to the max abs difference below.\n", "```\n" + pd.DataFrame(repro).T.to_string(header=False) + "\n```\n",
      "B+D (cap-free): " + ", ".join(f"{m} {f3(R[(R.feature_set == 'B+D') & (R.model == m)].iloc[0])}" for m in cc["models"]) + "\n"]  # fmt: skip
for sname in cc["sets"]:
    md.append(f"\n{sname}:\n\n| cap variant | " + " | ".join(cc["models"]) + " |\n|---|" + "---|" * len(cc["models"]))
    for v in VARS:
        md.append(f"| {v} | " + " | ".join(f3(R[(R.variant == v) & (R.feature_set == sname) & (R.model == m)].iloc[0]) for m in cc["models"]) + " |")
md.append("\nPaired differences (same resamples):\n\n| cap variant | set | comparison | " + " | ".join(cc["models"]) + " |\n|---|---|---|" + "---|" * len(cc["models"]))
for (v, sname, cmp_), g in P.groupby(["variant", "feature_set", "comparison"], sort=False):
    md.append(f"| {v} | {sname} | {cmp_} | " + " | ".join(f"{g[g.model == m].iloc[0]['diff']:+.3f} [{g[g.model == m].iloc[0].ci_low:+.3f}, {g[g.model == m].iloc[0].ci_high:+.3f}]" for m in cc["models"]) + " |")
(OUT / "summary.md").write_text("\n".join(md) + "\n")
print(f"done in {time.time() - t0:.0f}s -> {OUT}", flush=True)
