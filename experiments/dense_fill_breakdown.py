"""Why are planted dense rings filled? Breakdown of T_local_ring_inf = 0 for the dense scenario.

    python experiments/dense_fill_breakdown.py [--seeds 6] [--no-internal-chords] [--out NAME]

For each planted dense ring (a candidate at max_len 6, sparsified local graph, 1 hop) the exact
ring_survives test is run on three graphs:
    full      the ring's local graph H                       -> survives / filled
    inside    H restricted to the ring members               -> can members alone fill it?
    outside   H minus every non-ring edge between two members -> can outside accounts alone fill it?
Categories: survives; filled_inside_only; filled_outside_only; filled_both (each alone fills it);
filled_combined (neither alone, only together). Writes results/phase2_features/<NAME>.csv.
"""

import argparse
from pathlib import Path

import pandas as pd

from ringtda.data.synthetic import SyntheticConfig, generate
from ringtda.features import window_graphs
from ringtda.topology.candidates import candidate_rings, neighborhood
from ringtda.topology.vectorize import ring_survives

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=6)
ap.add_argument("--no-internal-chords", action="store_true")
ap.add_argument("--out", default="dense_fill_breakdown")
args = ap.parse_args()
kw = {"dense_internal_chords": False} if args.no_internal_chords else {}

rows = []
for seed in range(args.seeds):
    tx, planted = generate(SyntheticConfig(seed=seed, hard_negatives=False, **kw))
    _, G, Gs, tau = window_graphs(tx)
    cands = candidate_rings(G, tau=tau, max_len=6).candidates
    by_edges = {frozenset(c.edges): c for c in cands}
    for p in planted:
        if p.pattern != "wash_ring" or p.scenario != "dense":
            continue
        c = by_edges[frozenset(p.edges)]
        H = neighborhood(Gs, c.nodes, 1)
        members, ring = set(c.nodes), set(c.edges)
        inside = H.subgraph(members).copy()
        outside = H.copy()
        outside.remove_edges_from([(a, b) for a, b in H.edges if a in members and b in members and (a, b) not in ring])
        s_full, s_in, s_out = (ring_survives(g, c.edges) for g in (H, inside, outside))
        if s_full:
            cat = "survives"
        elif not s_in and s_out:
            cat = "filled_inside_only"
        elif s_in and not s_out:
            cat = "filled_outside_only"
        elif not s_in and not s_out:
            cat = "filled_both"
        else:
            cat = "filled_combined"
        rows.append({"seed": seed, "pattern_id": p.pattern_id, "ring_len": len(c.nodes), "category": cat,
                     "internal_nonring_edges": sum(1 for a, b in H.edges if a in members and b in members and (a, b) not in ring)})
df = pd.DataFrame(rows)
out = Path("results/phase2_features")
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / f"{args.out}.csv", index=False)
print(df.category.value_counts().to_string(), f"\ntotal {len(df)}")
print("rings with >=1 internal non-ring edge:", int((df.internal_nonring_edges > 0).sum()))
