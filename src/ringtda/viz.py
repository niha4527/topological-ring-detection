"""Plotting helpers for notebooks. Pipeline logic does not live here."""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def draw_digraph(ax, G: nx.DiGraph, highlight_edges=None, pos=None, title: str | None = None, weight: str = "weight"):
    """Directed graph; edges coloured by weight (dark = low weight = suspicious).

    highlight_edges: edges drawn thick red (e.g. a candidate ring).
    """
    highlight_edges = set(highlight_edges or [])
    pos = pos or nx.spring_layout(G, seed=1)
    others = [e for e in G.edges() if e not in highlight_edges]
    ws = [G.edges[e].get(weight, 0.5) for e in others]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=260, node_color="#dde3ea", edgecolors="#4a5560")
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
    nx.draw_networkx_edges(
        G, pos, edgelist=others, ax=ax, edge_color=ws, edge_cmap=plt.cm.Greys_r, edge_vmin=0, edge_vmax=1.2,
        arrows=True, arrowsize=12, connectionstyle="arc3,rad=0.12", node_size=260,
    )
    if highlight_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=list(highlight_edges), ax=ax, edge_color="#d6336c", width=2.4,
            arrows=True, arrowsize=14, connectionstyle="arc3,rad=0.12", node_size=260,
        )
    if G.number_of_edges() and weight in next(iter(G.edges(data=True)))[2]:
        nx.draw_networkx_edge_labels(
            G, pos, ax=ax, font_size=7, connectionstyle="arc3,rad=0.12",
            edge_labels={e: f"{G.edges[e][weight]:.2f}" for e in G.edges()} if G.number_of_edges() <= 12 else {},
        )
    ax.set_title(title or "")
    ax.axis("off")
    return pos


def plot_diagram(ax, dgms: dict, title: str | None = None, max_value: float | None = None):
    """Persistence diagram for H0 and H1. Bars that never die are drawn on the dashed 'inf' line."""
    finite = [d[np.isfinite(d)] for d in dgms.values() if len(d)]
    top = max_value or max([1.0] + [float(f.max()) for f in finite if len(f)])
    inf_y = top * 1.1
    ax.plot([0, inf_y], [0, inf_y], color="#999", lw=0.8)
    ax.axhline(inf_y, color="#999", lw=0.8, ls="--")
    ax.text(0.01 * inf_y, inf_y * 1.01, "inf (never dies)", fontsize=7, color="#666")
    styles = {0: dict(color="#4c6ef5", marker="o", label="H0 (components)"), 1: dict(color="#d6336c", marker="^", label="H1 (loops)")}
    for dim, d in dgms.items():
        if len(d) == 0:
            continue
        y = np.where(np.isfinite(d[:, 1]), d[:, 1], inf_y)
        ax.scatter(d[:, 0], y, s=40, alpha=0.8, **styles.get(dim, {}))
    ax.set_xlim(-0.03 * inf_y, inf_y * 1.05)
    ax.set_ylim(-0.03 * inf_y, inf_y * 1.08)
    ax.set_xlabel("birth (edge weight)")
    ax.set_ylabel("death")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(title or "")
