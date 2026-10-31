"""Smoke check of the Phase 2 T feature table (hard_negatives=False). Writes results/phase2_features/.

    python experiments/phase2_features_smoke.py [--seeds 6] [--no-internal-chords]

Prints the table shape, mean T_local_inf_h1 / T_local_ring_inf per candidate kind and scenario, and
the fraction of planted wash rings that are candidates and have T_local_ring_inf = 1.
Descriptive only: not a performance result and not a gate.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from ringtda.data.synthetic import SyntheticConfig, generate
from ringtda.eval.labels import label_candidates
from ringtda.features import t_feature_table

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=6)
ap.add_argument("--no-internal-chords", action="store_true", help="generate with dense_internal_chords=False")
args = ap.parse_args()
OUT = Path("results") / "phase2_features"
OUT.mkdir(parents=True, exist_ok=True)
SUF = "_no_internal_chords" if args.no_internal_chords else ""
GEN = {"dense_internal_chords": False} if args.no_internal_chords else {}

t0 = time.time()
data = [generate(SyntheticConfig(seed=s, hard_negatives=False, **GEN)) for s in range(args.seeds)]
table, cands = t_feature_table([tx for tx, _ in data], return_candidates=True)  # max_len 6 (default)

meta = []
for wid, (tx, planted) in enumerate(data):
    scen = {p.pattern_id: p.scenario for p in planted}
    lab = label_candidates(cands[wid], tx, planted)
    for cid, (k, pid) in enumerate(zip(lab["kind"], lab["pattern_id"])):
        meta.append({"window_id": wid, "candidate_id": cid, "kind": k, "scenario": scen.get(pid, k)})
meta = pd.DataFrame(meta)
full = meta.merge(table, on=["window_id", "candidate_id"])
full.to_csv(OUT / f"t_table{SUF}.csv", index=False)

tcols = [c for c in table.columns if c not in ("window_id", "candidate_id")]
print(f"table shape {table.shape}  ({len(tcols)} T columns, {table.window_id.nunique()} windows, {time.time() - t0:.0f}s)")
print("NaNs:", int(table[tcols].isna().sum().sum()))
agg = {"n": ("T_local_ring_inf", "size"), "inf_h1": ("T_local_inf_h1", "mean"), "ring_inf": ("T_local_ring_inf", "mean")}
by_kind = full.groupby("kind").agg(**agg).round(3)
by_scen = full[full.kind.isin(["wash_ring", "shortcut_filled"])].groupby(["kind", "scenario"]).agg(**agg).round(3)
print("\nby candidate kind\n", by_kind)
print("\nplanted structures by scenario\n", by_scen)
print("(transitive triangles contain no directed cycle: they are never candidate rows)")

# planted wash rings: is the ring itself a candidate, and does its loop survive?
rec = []
for wid, (tx, planted) in enumerate(data):
    rows = table[table.window_id == wid].set_index("candidate_id")
    by_edges = {frozenset(c.edges): cid for cid, c in enumerate(cands[wid])}
    for p in planted:
        if p.pattern == "wash_ring":
            cid = by_edges.get(frozenset(p.edges))
            rec.append({"window_id": wid, "pattern_id": p.pattern_id, "scenario": p.scenario, "candidate": cid is not None,
                        "ring_inf": float(rows.loc[cid, "T_local_ring_inf"]) if cid is not None else float("nan")})
rec = pd.DataFrame(rec)
summ = rec.groupby("scenario").agg(rings=("candidate", "size"), candidate_frac=("candidate", "mean"), ring_inf_frac=("ring_inf", "mean")).round(3)
print("\nplanted wash rings: fraction that are candidates / with T_local_ring_inf = 1\n", summ)
bad = rec[rec.candidate & (rec.ring_inf != 1.0)]
print("failing cases (candidate with ring_inf != 1):", bad.to_dict("records") if len(bad) else "none")
rec.to_csv(OUT / f"planted_rings{SUF}.csv", index=False)
json.dump({"shape": list(table.shape), "by_kind": by_kind.to_dict("index"), "planted": summ.to_dict("index"), "seeds": args.seeds}, open(OUT / f"summary{SUF}.json", "w"), indent=1, default=float)
