"""Figures of experiments/candidate_recall.py: recall vs tau_quantile per source, drawn from the CSVs only."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

D = Path("results/phase3_features/classify/candidate_recall")
if (D / "ibm.csv").exists():
    R = pd.read_csv(D / "ibm.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, scope in zip(axes, ["days0-5", "whole_data"]):
        for (ds, ml), g in R[R.scope == scope].groupby(["source", "max_len"]):
            ax.plot(g.tau_quantile, g.recall_of_reachable, marker="o", ls="-" if ml == 6 else "--", label=f"{ds} max_len {ml}")
        ax.set(title=f"IBM, {scope}", xlabel="tau_quantile", ylim=(0, 1.02))
    axes[0].set_ylabel("recall of complete CYCLE attempts\n(with <= max_len accounts)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(D / "ibm.png", dpi=150)
if (D / "synthetic.csv").exists():
    R = pd.read_csv(D / "synthetic.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, hn in zip(axes, [False, True]):
        for (sc, ch), g in R[R.hard_negatives == hn].groupby(["scenario", "dense_internal_chords"]):
            r = g.groupby("tau_quantile").recovered.mean()
            ax.plot(r.index, r.values, marker="o", ls="-" if ch else ":", label=f"{sc}, chords {'default' if ch else 'off'}")
        ax.set(title=f"synthetic, hard negatives {'on' if hn else 'off'}", xlabel="tau_quantile", ylim=(0, 1.02))
    axes[0].set_ylabel("recall of planted wash rings (6 seeds)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(D / "synthetic.png", dpi=150)
