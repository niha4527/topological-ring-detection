"""Figures of experiments/bandit_report.py, drawn from the CSVs in results/phase6_bandit/ only."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

D = Path("results/phase6_bandit")
S = pd.read_csv(D / "summary.csv")
roll = pd.read_csv(D / "rolling_seed0.csv")
POL = ["fixed", "retune_M5", "ascent", "ducb", "swts", "oracle"]
COL = {"fixed": "tab:grey", "retune_M5": "tab:green", "ascent": "tab:blue", "ducb": "tab:red", "swts": "tab:purple", "oracle": "k"}

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
t = S[S["lambda"] == 1.0]
lags = sorted(t.lag.unique())
for i, p in enumerate(POL[:-1]):
    g = t[t.policy == p].set_index("lag").reindex(lags)
    ax[0].errorbar(np.arange(len(lags)) + (i - 2) * 0.12, g.share_mean, yerr=g.share_std, fmt="o", capsize=2, color=COL[p], label=p)
ax[0].set_xticks(range(len(lags)), [f"d={d}" for d in lags])
ax[0].axhline(1.0, c="k", lw=0.6, ls=":")
ax[0].set_ylabel("share of oracle (mean +- std over seeds)")
ax[0].set_title("lambda 1, share of hindsight-oracle reward", fontsize=10)
ax[0].legend(fontsize=7, frameon=False)
for p in POL:
    g = roll[roll.policy == p].sort_values("window")
    tp, fp, pos = (g[c].rolling(5, min_periods=1).sum() for c in ("tp", "fp", "positives"))
    ax[1].plot(g.window, tp / (tp + fp).replace(0, np.nan), color=COL[p], label=p, ls=":" if p == "oracle" else "-")
    ax[2].plot(g.window, tp / pos, color=COL[p], label=p, ls=":" if p == "oracle" else "-")
ax[1].set_title("rolling precision (5 windows), stream seed 0, d=2, lambda 1", fontsize=10)
ax[2].set_title("rolling recall (5 windows), stream seed 0, d=2, lambda 1", fontsize=10)
for a in ax[1:]:
    a.set_xlabel("window (drift increases)")
    a.set_ylim(-0.02, 1.02)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(D / "bandit_drift.png", dpi=140)
print("wrote", D / "bandit_drift.png")
