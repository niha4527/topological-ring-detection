"""Figure for the classification harness, drawn from results.csv: PR-AUC with grouped-bootstrap CIs per
feature set and model (primary variant; dashed line = prevalence). Usage: python experiments/ibm_classify_plot.py HI 0.05"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ds, tau = (sys.argv[1] if len(sys.argv) > 1 else "HI"), (sys.argv[2] if len(sys.argv) > 2 else "0.05")
D = Path("results/phase3_features/classify") / f"{ds}_tau{tau}"
R = pd.read_csv(D / "results.csv")
R = R[(R.variant == "primary") & (R.model != "none")]
sets = [s for s in ["B", "B+D", "B+D+T", "B+D+T_local", "B+D+T_win", "T", "D", "T_local", "T_win"] if s in set(R.feature_set)]
models = ["logreg", "rf", "svm"]
fig, ax = plt.subplots(figsize=(11, 4.2))
w = 0.26
for i, m in enumerate(models):
    d = R[R.model == m].set_index("feature_set").reindex(sets)
    x = np.arange(len(sets)) + (i - 1) * w
    ax.bar(x, d.pr_auc, w, yerr=[d.pr_auc - d.ci_low, d.ci_high - d.pr_auc], capsize=2, label=m)
ax.axhline(R.prevalence.iloc[0], ls="--", c="k", lw=0.8, label=f"prevalence {R.prevalence.iloc[0]:.3f}")
ax.set_xticks(range(len(sets)), sets)
ax.set_ylabel("PR-AUC (pooled out-of-fold, 95% grouped-bootstrap CI)")
ax.set_title(f"IBM {ds}-Small, tau_quantile {tau}: positives {int(R.n_pos.iloc[0])} in {int(R.n_pos_groups.iloc[0])} groups", fontsize=10)
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(D / "pr_auc.png", dpi=130)
print("wrote", D / "pr_auc.png")
