"""Figure of experiments/pump_dump_run.py from results/phase7_pump_dump/results.csv: PR-AUC [CI] by feature set and model per condition."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

D = Path("results/phase7_pump_dump")
R = pd.read_csv(D / "results.csv")
conds = list(R.condition.unique())
fig, axes = plt.subplots(1, len(conds), figsize=(5 * len(conds), 4), sharey=True, squeeze=False)
for ax, c in zip(axes[0], conds):
    g = R[(R.condition == c) & (R.model != "none")]
    for j, m in enumerate(["logreg", "rf", "svm"]):
        x = g[g.model == m].set_index("feature_set").reindex(["B_pd", "M"])
        ax.errorbar([i + (j - 1) * 0.2 for i in range(2)], x.pr_auc, yerr=[x.pr_auc - x.ci_low, x.ci_high - x.pr_auc], fmt="o", capsize=2, label=m)
    ax.axhline(g.prevalence.iloc[0], ls="--", c="k", lw=0.8)
    ax.set(title=f"{c}: prevalence {g.prevalence.iloc[0]:.3f}, {int(g.n_pos_groups.iloc[0])} positive groups", xticks=range(2), xticklabels=["B_pd", "M"], ylim=(0, 1.02))
axes[0][0].set_ylabel("PR-AUC (95% grouped-bootstrap CI)")
axes[0][-1].legend()
fig.suptitle("Synthetic pump-and-dump, no topology (generator decides separability)", fontsize=10)
fig.tight_layout()
fig.savefig(D / "pump_dump_pr_auc.png", dpi=140)
print("wrote", D / "pump_dump_pr_auc.png")
