"""F7: paired difference B+D+T minus B+D per synthetic scenario, from per_scenario_paired.csv only (no model runs).

    python experiments/plot_lift_per_scenario.py

One panel per hard-negative setting; x = scenario (isolated, dense, chorded); colour = model; filled marker =
dense_internal_chords default, hollow = off. Error bars: 95% grouped-bootstrap CI of the paired difference
(same resamples for both feature sets, resampling groups). Prevalence differs per scenario (positives of one
scenario against all negatives): it is printed under each scenario label.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

D = Path("results/phase3_features/classify/synthetic")
SCN, MODELS = ["isolated", "dense", "chorded"], ["logreg", "rf", "svm"]
COL = {"logreg": "tab:blue", "rf": "tab:orange", "svm": "tab:green"}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, hn in zip(axes, ["off", "on"]):
    prev = {}
    for ch, filled in (("default", True), ("off", False)):
        c = f"hn-{hn}_chords-{ch}"
        P = pd.read_csv(D / c / "per_scenario_paired.csv")
        S = pd.read_csv(D / c / "per_scenario.csv")
        P = P[P.comparison == "B+D+T minus B+D"]
        for i, scn in enumerate(SCN):
            prev.setdefault(scn, []).append(float(S[(S.scenario == scn) & (S.feature_set == "B") & (S.model == "logreg")].prevalence.iloc[0]))
            for j, m in enumerate(MODELS):
                r = P[(P.scenario == scn) & (P.model == m)].iloc[0]
                x = i + (j - 1) * 0.22 + (0.05 if filled else -0.05)
                ax.errorbar(x, r["diff"], yerr=[[r["diff"] - r.ci_low], [r.ci_high - r["diff"]]], fmt="o", ms=5, capsize=2, color=COL[m], mfc=COL[m] if filled else "white", label=m if (i == 0 and filled) else None)
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xticks(range(len(SCN)), [f"{s}\nprevalence {min(prev[s]):.4f}-{max(prev[s]):.4f}" for s in SCN], fontsize=8)
    ax.set_title(f"hard negatives {hn}", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("PR-AUC difference, B+D+T minus B+D\n(95% grouped-bootstrap CI)")
axes[0].legend(frameon=False, title="filled: chords default, hollow: chords off", fontsize=8, title_fontsize=7)
fig.tight_layout()
fig.savefig(D / "lift_per_scenario.png", dpi=150)
print("wrote", D / "lift_per_scenario.png")
