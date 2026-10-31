"""Figure of experiments/ibm_cluster.py from results/phase4_cluster/coords.csv: 2D PCA by pattern type (left) and by cluster (right)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

D = Path("results/phase4_cluster")
C = pd.read_csv(D / "coords.csv")
for ds, g in C.groupby("dataset"):
    vs = list(g.vector.unique())
    fig, ax = plt.subplots(2, len(vs), figsize=(4 * len(vs), 7), squeeze=False)
    for j, v in enumerate(vs):
        x = g[g.vector == v]
        for i, (col, title) in enumerate((("kind", "pattern type"), ("cluster", "Ward cluster"))):
            for lab, y in x.groupby(col):
                ax[i, j].scatter(y.pc1, y.pc2, s=14, alpha=0.7, label=str(lab))
            ax[i, j].set_title(f"{v}: by {title}", fontsize=9)
            ax[i, j].legend(fontsize=6, frameon=False)
            ax[i, j].spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"IBM {ds}-Small positive rings, PCA of the standardized vectors (descriptive)", fontsize=10)
    fig.tight_layout()
    fig.savefig(D / f"pca_{ds}.png", dpi=130)
    print("wrote", D / f"pca_{ds}.png")
