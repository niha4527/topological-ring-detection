"""Per-pattern-type diagnostic of D_birth and D_bal_mean, positives vs background (primary variant, no models).

    python experiments/ibm_type_diagnostic.py --tau 0.05

Reads the feature files and the candidate pickle only. Writes results/phase3_features/classify/type_diagnostic.md.
"""

import argparse
import glob
from pathlib import Path

import pandas as pd

ap_ = argparse.ArgumentParser()
ap_.add_argument("--tau", default="0.05")
args = ap_.parse_args()
COLS = ["D_birth", "D_bal_mean"]
md = [f"# D_birth and D_bal_mean by pattern type, tau_quantile {args.tau}, max_len 6\n",
      "Primary variant (ambiguous excluded). Mean over candidates; no model runs. D_birth = ring birth (lower = more suspicious); "
      "positives = candidates labelled by §3.7 with that attempt type; background = candidates with kind 'background'.\n"]  # fmt: skip
for ds in ("HI", "LI"):
    fdir = Path("results/phase3_features/ibm") / f"{ds}_tau{args.tau}"
    F = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(fdir / "features_window_*.csv.gz")))], ignore_index=True)
    L = pd.read_pickle(f"results/phase3_features/ibm_candidates_{ds}_{float(args.tau)}.pkl")[["window_id", "cand_id", "kind", "y", "group"]].rename(columns={"cand_id": "candidate_id"})
    d = F.merge(L, on=["window_id", "candidate_id"], validate="one_to_one")
    d = d[d.y.notna()]
    bg = d[d.kind == "background"]
    md.append(f"\n## {ds}-Small: {len(d)} candidates, {int((d.y == 1).sum())} positives, {len(bg)} background\n")
    md.append("| type | positives | positive groups | D_birth pos | D_birth background | D_bal_mean pos | D_bal_mean background |\n|---|---|---|---|---|---|---|")
    for k in ["CYCLE", "GATHER-SCATTER", "SCATTER-GATHER", "STACK", "all positives"]:
        p = d[d.y == 1] if k == "all positives" else d[d.kind == k]
        if not len(p):
            md.append(f"| {k} | 0 | 0 | | | | |")
            continue
        md.append(f"| {k} | {len(p)} | {p.group.nunique()} | {p.D_birth.mean():.4f} | {bg.D_birth.mean():.4f} | {p.D_bal_mean.mean():.4f} | {bg.D_bal_mean.mean():.4f} |")
out = Path("results/phase3_features/classify/type_diagnostic.md")
out.write_text("\n".join(md) + "\n")
print(out.read_text())
