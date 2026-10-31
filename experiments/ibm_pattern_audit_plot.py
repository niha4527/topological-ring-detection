"""Figure for the pattern-behaviour audit, drawn from the summary CSV (percentile boxes: p5-p95 whiskers,
p25-p75 box, median line). Usage: python experiments/ibm_pattern_audit_plot.py HI"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ds = sys.argv[1] if len(sys.argv) > 1 else "HI"
OUT = Path("results/phase3_features/audit")
S = pd.read_csv(OUT / f"{ds}_audit_summary.csv")
S = S[S.window == "pooled"]
cols = ["n_s_rec", "n_s_bal", "n_s_reg", "weight"]
groups = list(S.group.unique())
fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 4.2), sharey=False)
for ax, col in zip(axes, cols):
    d = S[S.score == col].set_index("group").reindex(groups)
    stats = [dict(label=g, whislo=d.loc[g, "p5"], q1=d.loc[g, "p25"], med=d.loc[g, "median"], q3=d.loc[g, "p75"], whishi=d.loc[g, "p95"], fliers=[]) for g in groups]
    ax.bxp(stats, showfliers=False, orientation="vertical")
    ax.set_title({"n_s_rec": "rank-normalised s_rec", "n_s_bal": "rank-normalised s_bal", "n_s_reg": "rank-normalised s_reg", "weight": "combined weight (low = suspicious)"}[col], fontsize=9)
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"IBM {ds}-Small: score distributions by pattern group (pooled over audited windows)", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / f"{ds}_audit_scores.png", dpi=130)
print("wrote", OUT / f"{ds}_audit_scores.png")
