"""Gantt chart of the project with each task's status on 3 Nov 2026.

    python experiments/make_gantt.py              # 6-week timeline -> paper/gantt_chart.png and .svg
    python experiments/make_gantt.py --weeks 12   # the v3 proposal's 12-week phase plan (section 3.7) -> paper/gantt_chart_12wk.png

Weeks are the PLAN, not commit dates. The 6-week timeline compresses the 12-week plan and lets phases overlap.
Tasks marked "after plan" were added later. Colours: the status palette (done / in progress), grey hatched for cut; every bar also
carries a symbol and word, so colour is never the only signal.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#5b5a52", "#e1e0d9", "#c3c2b7"
STATUS = {"done": ("#0ca30c", "#ffffff", "✓ Done"), "doing": ("#fab219", "#0b0b0b", "… In progress"), "cut": ("#c9c8c0", "#0b0b0b", "✕ Cut")}

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--weeks", type=int, choices=[6, 12], default=6)
args = ap.parse_args()

# (label, detail, start week, end week (inclusive), status)
TASKS12 = [
    ("1  Synthetic data", "Generator, scenarios, edge weights, optimization report", 1, 2, "done"),
    ("2  Topology pipeline", "Candidate rings, directed flag persistence, T features, tests", 3, 4, "done"),
    ("3  IBM AML study", "Loader, B and D features, audit, classification, transfer", 5, 6, "done"),
    ("4  Typing and severity", "Ring clustering, severity regression (synthetic)", 7, 7, "done"),
    ("5  NFT real data", "Cut: the free Dune tier blocks running the queries", 8, 9, "cut"),
    ("6  Adaptive threshold", "Gradient ascent, bandits, drift benchmark", 10, 10, "done"),
    ("7  Pump-and-dump", "Optional synthetic track, no topology", 11, 11, "done"),
    ("8  Write-up", "Paper, explainer, slides", 11, 12, "done"),
    ("9  Demo web app (after plan)", "Simulator, step-by-step walkthrough, results tab", 12, 12, "done"),
    ("10 IBM Medium run (after plan)", "7x more attempts, 87 positive groups; write-up pending", 12, 13, "doing"),
]
TASKS6 = [
    ("1  Synthetic data", "Generator, scenarios, edge weights, optimization report", 1, 1, "done"),
    ("2  Topology pipeline", "Candidate rings, directed flag persistence, T features, tests", 2, 2, "done"),
    ("3  IBM AML study", "Loader, labels, B and D features, audit, classification, transfer", 3, 4, "done"),
    ("4  Typing and severity", "Ring clustering, severity regression (synthetic)", 4, 4, "done"),
    ("5  NFT real data", "Cut: the free Dune tier blocks running the queries", 4, 5, "cut"),
    ("6  Adaptive threshold", "Gradient ascent, bandits, drift benchmark", 5, 5, "done"),
    ("7  Pump-and-dump", "Optional synthetic track, no topology", 5, 5, "done"),
    ("8  Write-up", "Paper, explainer, slides", 5, 6, "done"),
    ("9  Demo web app", "Simulator, step-by-step walkthrough, results tab", 6, 6, "done"),
    ("10 IBM Medium run (after plan)", "7x more attempts, 87 positive groups; write-up pending", 7, 7, "doing"),
]
TASKS = TASKS6 if args.weeks == 6 else TASKS12
PLAN = args.weeks
WEEKS = PLAN + 1  # planned weeks plus one column for work after the plan

fig, ax = plt.subplots(figsize=(13.5, 6.6), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
n = len(TASKS)
ax.set_xlim(0, WEEKS)
ax.set_ylim(n, -0.15)
ax.add_patch(Rectangle((PLAN, -0.15), 1, n + 0.15, facecolor="#f1f0ea", edgecolor="none", zorder=0))  # after-plan column
for w in range(WEEKS + 1):
    ax.axvline(w, color=GRID, lw=0.8, zorder=1)
for i in range(n + 1):
    ax.axhline(i, color=GRID, lw=0.8, zorder=1)

for i, (label, detail, a, b, st) in enumerate(TASKS):
    fill, ink, text = STATUS[st]
    x0, w = a - 1 + 0.06, (b - a + 1) - 0.12
    ax.add_patch(Rectangle((x0, i + 0.17), w, 0.66, facecolor=fill, edgecolor="#8a897f" if st == "cut" else "none", hatch="///" if st == "cut" else None, lw=0.0 if st != "cut" else 0.8, zorder=3))
    ax.text(x0 + w / 2, i + 0.5, text, ha="center", va="center", fontsize=9.5, fontweight="bold", color=ink, zorder=4,
            bbox=dict(facecolor=fill, edgecolor="none", pad=1.2) if st == "cut" else None)

ax.set_yticks([])
ax.set_xticks([w + 0.5 for w in range(WEEKS)])
ax.set_xticklabels([f"Week {w}" if PLAN == 6 else f"W{w}" for w in range(1, PLAN + 1)] + ["after\nplan"], fontsize=10 if PLAN == 6 else 9.5, color=MUTED)
ax.xaxis.tick_top()
ax.tick_params(axis="x", length=0, pad=6)
for s in ax.spines.values():
    s.set_visible(False)
ax.spines["top"].set_visible(True)
ax.spines["top"].set_color(AXIS)

LEFT = (0.012 - 0.285) / (0.985 - 0.285)  # left margin start in axes coordinates, so labels share the bars' row coordinates
for i, (label, detail, *_r) in enumerate(TASKS):  # row labels in the left margin: name (bold) and detail (secondary)
    ax.text(LEFT, i + 0.33, label, fontsize=10.5, fontweight="bold", color=INK, va="center", ha="left", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.text(LEFT, i + 0.7, detail, fontsize=8.3, color=MUTED, va="center", ha="left", transform=ax.get_yaxis_transform(), clip_on=False)

fig.suptitle("Project plan and status", x=0.012, y=0.975, ha="left", fontsize=16, fontweight="bold", color=INK)
fig.text(0.012, 0.925, "Topological ring detection for market manipulation. " + ("Six-week timeline (v3 plan compressed, phases overlap). " if PLAN == 6 else "Weeks follow the 12-week v3 plan. ") + "Status as of 3 Nov 2026.", fontsize=9.5, color=MUTED)
handles = [Patch(facecolor=STATUS["done"][0], label="Done"), Patch(facecolor=STATUS["doing"][0], label="In progress"), Patch(facecolor=STATUS["cut"][0], hatch="///", edgecolor="#8a897f", label="Cut")]
fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.975), ncol=3, frameon=False, fontsize=9.5, labelcolor=INK, handlelength=1.4)
plt.subplots_adjust(left=0.285, right=0.985, top=0.855, bottom=0.04)

out = Path("paper")
stem = "gantt_chart" if PLAN == 6 else "gantt_chart_12wk"
fig.savefig(out / f"{stem}.png", facecolor=SURFACE)
fig.savefig(out / f"{stem}.svg", facecolor=SURFACE)
print("wrote", out / f"{stem}.png")
