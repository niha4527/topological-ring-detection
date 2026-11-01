"""Draw the two schematic figures of the paper (pipeline, triangle persistence bars).

Deterministic; no random numbers. F2 bars are computed with the project's own
directed_flag_persistence on the toy graphs of data/toy.py, so they are not hand-drawn.

    python experiments/make_paper_figures.py --out paper/ieee/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from ringtda.data.toy import cyclic_triangle, shortcut_filled_triangle, transitive_triangle
from ringtda.topology.persistence import directed_flag_persistence

plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "axes.linewidth": 0.6})
INK, RED, BLUE = "#222222", "#c2255c", "#3b5bdb"


def pipeline(path: Path) -> None:
    """F1: eight boxes in two rows, full text width (7.0 in)."""
    fig, ax = plt.subplots(figsize=(7.0, 2.1))
    ax.set_xlim(0, 103)
    ax.set_ylim(0, 32)
    ax.axis("off")
    w, h, xs = 22, 11, [1, 27, 53, 79]
    row1 = [
        "1 Transactions\n(time, from, to,\namount)",
        "2 Window graph\nedge per pair, weight\nw (low = suspicious)",
        "3 Candidate rings\nsparsify, SCCs,\ncycles up to 6",
        "4 Directed flag\npersistence of ring\nneighbourhood -> T",
    ]
    row2 = [
        "5 B and D features\n(D holds ring birth)",
        "6 Labels: >= 80% of\nedges from one\nstructure",
        "7 Grouped CV,\nclassifiers",
        "8 PR-AUC with\ngrouped bootstrap",
    ]
    for x, t in zip(xs, row1):
        ax.add_patch(FancyBboxPatch((x, 19), w, h, boxstyle="round,pad=0.3", fc="#edf2ff", ec=INK, lw=0.7))
        ax.text(x + w / 2, 19 + h / 2, t, ha="center", va="center", fontsize=7)
    for x, t in zip(xs, row2):
        ax.add_patch(FancyBboxPatch((x, 2), w, h, boxstyle="round,pad=0.3", fc="#fff4e6", ec=INK, lw=0.7))
        ax.text(x + w / 2, 2 + h / 2, t, ha="center", va="center", fontsize=7)
    kw = dict(arrowstyle="-|>", mutation_scale=7, lw=0.7, color=INK)
    for i in range(3):
        ax.add_patch(FancyArrowPatch((xs[i] + w + 0.5, 24.5), (xs[i + 1] - 0.5, 24.5), **kw))
        ax.add_patch(FancyArrowPatch((xs[i] + w + 0.5, 7.5), (xs[i + 1] - 0.5, 7.5), **kw))
    ax.add_patch(FancyArrowPatch((xs[1] + w / 2, 18.2), (xs[0] + w / 2 + 6, 13.8), **kw))
    ax.add_patch(FancyArrowPatch((xs[3] + w / 2, 18.2), (xs[2] + w / 2 + 6, 13.8), **kw))
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _draw_triangle(ax, edges, pos, title):
    for (u, v, w) in edges:
        ax.add_patch(FancyArrowPatch(pos[u], pos[v], arrowstyle="-|>", mutation_scale=8, lw=1.1,
                                     color=RED if w <= 0.1 else "#868e96", shrinkA=7, shrinkB=7,
                                     connectionstyle="arc3,rad=0.0"))
    for n, (x, y) in pos.items():
        ax.scatter([x], [y], s=120, c="#dde3ea", ec=INK, zorder=3, linewidths=0.6)
        ax.text(x, y, "abcd"[n], ha="center", va="center", fontsize=7, zorder=4)
    ax.set_title(title, fontsize=7, loc="left")
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.25, 1.15)
    ax.axis("off")


def _draw_bars(ax, dgms, cap):
    h1 = dgms.get(1, np.zeros((0, 2)))
    ax.set_xlim(0, cap * 1.05)
    ax.set_ylim(-0.5, 1.2)
    ax.set_yticks([])
    ax.set_xlabel("filtration value (edge weight)", fontsize=7)
    ax.tick_params(labelsize=7, length=2)
    if len(h1) == 0:
        ax.text(cap / 2, 0.35, "no H1 bar (loop filled)", ha="center", fontsize=7, color=INK)
        return
    for i, (b, d) in enumerate(h1):
        inf = not np.isfinite(d)
        end = cap if inf else d
        ax.plot([b, end], [i * 0.5, i * 0.5], color=BLUE, lw=3, solid_capstyle="butt")
        if inf:
            ax.annotate("", xy=(cap * 1.04, i * 0.5), xytext=(cap * 0.97, i * 0.5),
                        arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.0))
        ax.text(b, i * 0.5 + 0.22, f"born {b:.1f}, " + ("never dies" if inf else f"dies {d:.1f}"), fontsize=7)


def triangles(path: Path) -> None:
    """F2: three rows (graph | H1 bars), single column width (3.4 in)."""
    cases = [
        ("(a) circular triangle: a loop", cyclic_triangle(0.1), {0: (0.5, 1.0), 1: (0.0, 0.0), 2: (1.0, 0.0)}),
        ("(b) transitive triangle: one-way flow", transitive_triangle(0.1), {0: (0.5, 1.0), 1: (0.0, 0.0), 2: (1.0, 0.0)}),
        ("(c) ring + account d (w = 0.6)", shortcut_filled_triangle(0.1, 0.6),
         {0: (0.5, 1.0), 1: (0.0, 0.0), 2: (1.0, 0.0), 3: (0.5, 0.35)}),
    ]
    cap = 0.8
    fig, axes = plt.subplots(3, 2, figsize=(3.4, 4.3), gridspec_kw=dict(width_ratios=[1, 1.35], hspace=0.9, wspace=0.12))
    for row, (title, G, pos) in zip(axes, cases):
        edges = [(u, v, d["weight"]) for u, v, d in G.edges(data=True)]
        _draw_triangle(row[0], edges, pos, title)
        _draw_bars(row[1], directed_flag_persistence(G, backend="reference"), cap)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/ieee/figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pipeline(out / "f1_pipeline.png")
    triangles(out / "f2_triangles.png")


if __name__ == "__main__":
    main()
