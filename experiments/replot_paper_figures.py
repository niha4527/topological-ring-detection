"""Re-draw two existing result figures at paper size (>= 7 pt text at the printed width).

Same CSV inputs and same numbers as ibm_classify_plot.py and plot_lift_per_scenario.py; only the layout
(size, font, subset of feature sets for F3) changes. No model is run. Run from the repository root:

    python experiments/replot_paper_figures.py --out paper/ieee/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6})
MODELS = ["logreg", "rf", "svm"]
COL = {"logreg": "#1f77b4", "rf": "#ff7f0e", "svm": "#2ca02c"}


def hi_pr_auc(out: Path) -> None:
    """F3: HI-Small primary variant, six feature sets, single column (3.4 in)."""
    R = pd.read_csv("results/phase3_features/classify/HI_tau0.05/results.csv")
    R = R[(R.variant == "primary") & (R.model != "none")]
    sets = ["B", "B+D", "B+D+T", "T", "D", "T_win"]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    w = 0.26
    for i, m in enumerate(MODELS):
        d = R[R.model == m].set_index("feature_set").reindex(sets)
        x = np.arange(len(sets)) + (i - 1) * w
        ax.bar(x, d.pr_auc, w, yerr=[d.pr_auc - d.ci_low, d.ci_high - d.pr_auc], capsize=1.2, label=m, color=COL[m],
               error_kw=dict(lw=0.6, capthick=0.6))
    ax.axhline(R.prevalence.iloc[0], ls="--", c="k", lw=0.7, label=f"prevalence {R.prevalence.iloc[0]:.4f}")
    ax.set_xticks(range(len(sets)), sets)
    ax.set_ylabel("PR-AUC (95% grouped-bootstrap CI)")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")
    ax.set_ylim(0, 1.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7, length=2)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "f3_hi_pr_auc.png", dpi=300)
    plt.close(fig)


def lift_per_scenario(out: Path) -> None:
    """F4: paired B+D+T minus B+D per synthetic scenario, full width (7.0 in)."""
    D = Path("results/phase3_features/classify/synthetic")
    scn = ["isolated", "dense", "chorded"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
    for ax, hn in zip(axes, ["off", "on"]):
        for ch, filled in (("default", True), ("off", False)):
            P = pd.read_csv(D / f"hn-{hn}_chords-{ch}" / "per_scenario_paired.csv")
            P = P[P.comparison == "B+D+T minus B+D"]
            for i, s in enumerate(scn):
                for j, m in enumerate(MODELS):
                    r = P[(P.scenario == s) & (P.model == m)].iloc[0]
                    x = i + (j - 1) * 0.22 + (0.05 if filled else -0.05)
                    ax.errorbar(x, r["diff"], yerr=[[r["diff"] - r.ci_low], [r.ci_high - r["diff"]]], fmt="o", ms=3.5,
                                capsize=1.5, lw=0.7, color=COL[m], mfc=COL[m] if filled else "white",
                                label=m if (i == 0 and filled and hn == "off") else None)
        ax.axhline(0, c="k", lw=0.6)
        ax.set_xticks(range(3), scn)
        ax.set_title(f"hard negatives {hn}", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7, length=2)
    axes[0].set_ylabel("PR-AUC difference\nB+D+T minus B+D")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left", title="filled: chords default; hollow: off", title_fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "f4_lift_per_scenario.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/ieee/figures")
    a = ap.parse_args()
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    hi_pr_auc(o)
    lift_per_scenario(o)
