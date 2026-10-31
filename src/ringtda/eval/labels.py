"""Candidate-ring labels from planted ground truth (proposal v3 §3.7, "ring-labeling rule").

Same rule as for IBM AML, with a planted structure playing the role of a
laundering attempt. An edge is *backed* by a structure if at least one of the
structure's own transactions is on it (background, decoy-trade and
neighbourhood "context" rows back nothing). A candidate ring is

    <structure's pattern>  if >= x of its edges are backed by one structure
                           (wash_ring -> y = 1; any other pattern -> y = 0)
    background             if none of its edges are backed (y = 0)
    ambiguous              otherwise: logged, and reported both excluded
                           and treated as negative.

For wash rings of size 2, 3 and 5 the 80% and 100% rules agree (any 4 edges of
a 5-ring already span all its accounts, so the closing edge is the ring's own),
and so do star-shaped structures (a simple cycle uses at most 2 hub edges).
They differ for structures with more edges than a single cycle, e.g. a chorded
6-ring: 1->2->3->4->5 plus a background edge 5->1 is 4/5 backed.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd

AMBIGUOUS = "ambiguous"
BACKGROUND = "background"
UNBACKED_PATTERNS = ("background", "decoy", "context")


def edge_backing(tx: pd.DataFrame) -> dict[tuple, set[int]]:
    """(src, dst) -> ids of the planted structures whose own transactions use that edge."""
    own = tx[(tx["pattern_id"] >= 0) & ~tx["pattern"].isin(UNBACKED_PATTERNS)]
    out: dict[tuple, set[int]] = defaultdict(set)
    for s, d, pid in zip(own["src"], own["dst"], own["pattern_id"]):
        out[(s, d)].add(int(pid))
    return out


def label_ring(edges, backing: dict, patterns: dict[int, str], x: float = 0.8) -> tuple[str, int]:
    """(kind, pattern_id) for one candidate; pattern_id is -1 unless kind names a structure."""
    counts = Counter(pid for e in edges for pid in backing.get(e, ()))
    if not counts:
        return BACKGROUND, -1
    pid, n = counts.most_common(1)[0]
    if n / len(edges) >= x:
        return patterns[pid], pid
    return AMBIGUOUS, -1


def label_candidates(candidates, tx: pd.DataFrame, planted, x: float = 0.8) -> pd.DataFrame:
    """One row per candidate: kind, pattern_id, y (1 = wash ring, 0 = negative, NaN = ambiguous)."""
    backing = edge_backing(tx)
    patterns = {p.pattern_id: p.pattern for p in planted}
    rows = []
    for c in candidates:
        kind, pid = label_ring(c.edges, backing, patterns, x)
        rows.append({"kind": kind, "pattern_id": pid, "y": float("nan") if kind == AMBIGUOUS else float(kind == "wash_ring")})
    return pd.DataFrame(rows, columns=["kind", "pattern_id", "y"])
