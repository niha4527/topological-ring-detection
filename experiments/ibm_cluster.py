"""Phase 4: Ward clustering of IBM positive rings against pattern types (v3 §3.5). Exploratory; nothing is a gate.

    nice -n 10 python experiments/ibm_cluster.py

Population per dataset: primary-variant positives of the pattern types with at least `min_groups_per_type` positive
groups. Vectors, k, linkage, bootstrap and baselines are fixed in configs/cluster.yaml. Outputs: results/phase4_cluster/.
"""

import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA

from ringtda.models.cluster import best_k, grouped_bootstrap, metrics, permutation_baseline, standardize, ward_labels

cfg = yaml.safe_load(open("configs/cluster.yaml"))
OUT = Path("results/phase4_cluster")
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
tau = cfg["dataset_tau"]
rows, counts, coords, notes = [], [], [], []


def cols(F, prefixes):
    out = []
    for p in prefixes:
        out += [c for c in F.columns if c.startswith(p + "_") and c not in out]
    return out


for ds in cfg["population"]["datasets"]:
    fdir = Path("results/phase3_features/ibm") / f"{ds}_tau{tau}"
    F = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(fdir / "features_window_*.csv.gz")))], ignore_index=True)
    L = pd.read_pickle(f"results/phase3_features/ibm_candidates_{ds}_{float(tau)}.pkl")[["window_id", "cand_id", "kind", "y", "group", "attempt"]].rename(columns={"cand_id": "candidate_id"})
    d = F.merge(L, on=["window_id", "candidate_id"], validate="one_to_one")
    pos = d[d.y == 1]
    per = pos.groupby("kind").agg(rings=("y", "size"), groups=("group", "nunique"), attempts=("attempt", "nunique")).reset_index()
    per["in_population"] = per.groups >= cfg["population"]["min_groups_per_type"]
    per.insert(0, "dataset", ds)
    counts.append(per)
    types = per[per.in_population].kind.tolist()
    if len(types) < cfg["population"]["min_types_for_dataset"]:
        notes.append(f"{ds}: unusable ({len(types)} types with >= {cfg['population']['min_groups_per_type']} positive groups)")
        continue
    pop = pos[pos.kind.isin(types)].reset_index(drop=True)
    truth, groups = pop.kind.to_numpy(), pop.group.to_numpy()
    k_fixed = len(types)
    notes.append(f"{ds}: {len(pop)} rings, {len(types)} types ({', '.join(types)}), {len(np.unique(groups))} groups; fixed k = {k_fixed}")
    print(notes[-1], flush=True)
    for vname, prefixes in cfg["vectors"].items():
        c = cols(F, prefixes)
        Z = standardize(pop[c].to_numpy(float))
        for mode in ("fixed", "silhouette"):
            if mode == "fixed":
                k, lab = k_fixed, ward_labels(Z, k_fixed)
                boots = grouped_bootstrap(Z, truth, groups, k_fixed, cfg["bootstrap"]["n_boot"], cfg["bootstrap"]["seed"])
            else:
                k, lab = best_k(Z, tuple(cfg["clustering"]["k_range"]))
                boots = grouped_bootstrap(Z, truth, groups, None, cfg["bootstrap"]["n_boot"], cfg["bootstrap"]["seed"], choose_k=tuple(cfg["clustering"]["k_range"]))
            m = metrics(Z, lab, truth, groups)
            base = permutation_baseline(lab, truth, groups, cfg["baseline"]["n_permutations"], cfg["baseline"]["seed"])
            B = pd.DataFrame(boots)
            ci = lambda s: (float(np.nanpercentile(B[s], 2.5)), float(np.nanpercentile(B[s], 97.5)))  # noqa: E731
            r = {"dataset": ds, "vector": vname, "k_mode": mode, "k": k, "n_features": len(c), "n_rings": len(pop), "n_types": len(types), "n_groups": int(len(np.unique(groups))), **m, **base,
                 "n_boot_valid": len(B), "n_boot": cfg["bootstrap"]["n_boot"], "boot_k_mean": float(B["k"].mean())}  # fmt: skip
            for s in ("ari_type", "purity_type", "ari_group", "silhouette_clusters"):
                r[f"{s}_ci_low"], r[f"{s}_ci_high"] = ci(s)
            rows.append(r)
            print(f"  {ds} {vname:22s} {mode:10s} k={k}: ARI {m['ari_type']:.3f} [{r['ari_type_ci_low']:.3f}, {r['ari_type_ci_high']:.3f}] purity {m['purity_type']:.3f} (majority {base['purity_majority']:.3f}) ARI(group) {m['ari_group']:.3f}", flush=True)
            if mode == "fixed":
                xy = PCA(2, random_state=0).fit_transform(Z)
                coords.append(pd.DataFrame({"dataset": ds, "vector": vname, "pc1": xy[:, 0], "pc2": xy[:, 1], "kind": truth, "cluster": lab, "group": groups}))

R = pd.DataFrame(rows)
R.to_csv(OUT / "results.csv", index=False)
pd.concat(counts).to_csv(OUT / "counts_by_type.csv", index=False)
pd.concat(coords).to_csv(OUT / "coords.csv", index=False)
yaml.safe_dump({"config": cfg, "notes": notes, "runtime_seconds": time.time() - t0}, open(OUT / "config_used.yaml", "w"))
f = lambda r, s: f"{r[s]:.3f} [{r[s + '_ci_low']:.3f}, {r[s + '_ci_high']:.3f}]"  # noqa: E731
md = ["# Phase 4: Ward clustering of positive rings against pattern type (exploratory)\n",
      "Ward linkage, Euclidean, vectors standardized on the population. Positive rings of the primary variant (ambiguous excluded). "
      "CI = grouped bootstrap over positive groups (resample groups with replacement, recluster), "
      f"{cfg['bootstrap']['n_boot']} resamples. Baselines: mean over {cfg['baseline']['n_permutations']} random permutations of the cluster labels, and majority-class purity. "
      "With few groups per type, clusters may separate groups rather than types: ARI against the group id is reported as a check.\n", *[f"- {n}" for n in notes], "\n## Positives per type\n",
      "```\n" + pd.concat(counts).to_string(index=False) + "\n```", ""]  # fmt: skip
for ds in R.dataset.unique():
    md.append(f"\n## {ds}\n\n| vector | k | ARI type [CI] | ARI random | purity type [CI] | purity random | purity majority | silhouette clusters [CI] | silhouette of type labels | ARI group id [CI] | ARI group random |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for r in R[R.dataset == ds].to_dict("records"):
        md.append(f"| {r['vector']} ({r['k_mode']}) | {r['k']} | {f(r, 'ari_type')} | {r['ari_type_random']:.3f} | {f(r, 'purity_type')} | {r['purity_type_random']:.3f} | {r['purity_majority']:.3f} | {f(r, 'silhouette_clusters')} | {r['silhouette_types']:.3f} | {f(r, 'ari_group')} | {r['ari_group_random']:.3f} |")
(OUT / "summary.md").write_text("\n".join(md) + "\n")
print(f"done in {time.time() - t0:.0f}s -> {OUT}", flush=True)
