"""Figure of experiments/synth_classify.py: overall PR-AUC [CI] per condition, feature set and model, from the CSVs only."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

D = Path("results/phase3_features/classify/synthetic")
conds = sorted(p.name for p in D.iterdir() if (p / "results.csv").exists())
ORDER = ["B", "B+D", "B+D+T", "B+D+T_local", "T"]
fig, axes = plt.subplots(1, len(conds), figsize=(4.2 * len(conds), 3.8), sharey=True, squeeze=False)
for ax, c in zip(axes[0], conds):
    R = pd.read_csv(D / c / "results.csv")
    R = R[(R.variant == "primary") & (R.model != "none")]
    for j, m in enumerate(["logreg", "rf", "svm"]):
        g = R[R.model == m].set_index("feature_set").reindex(ORDER)
        x = [i + (j - 1) * 0.25 for i in range(len(ORDER))]
        ax.errorbar(x, g.pr_auc, yerr=[g.pr_auc - g.ci_low, g.ci_high - g.pr_auc], fmt="o", capsize=2, label=m)
    ax.set(title=c, xticks=range(len(ORDER)), xticklabels=ORDER, ylim=(0, 1.02))
    ax.tick_params(axis="x", rotation=45)
axes[0][0].set_ylabel("PR-AUC (primary)")
axes[0][-1].legend()
fig.tight_layout()
fig.savefig(D / "pr_auc.png", dpi=150)
