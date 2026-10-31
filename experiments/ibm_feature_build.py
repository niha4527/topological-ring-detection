"""Resumable builder of the B / D / T feature tables for IBM AML windows (README "Phase 3 design decisions").

    python experiments/ibm_feature_build.py --dataset HI --tau 0.05 --max-len 6

One output set per window in results/phase3_features/ibm/<dataset>_tau<tau>/ (git-ignored):
    features_window_NN.csv.gz  keys (window_id, candidate_id) + B + D + T columns, no label columns
    ids_window_NN.csv.gz       window_id, candidate_id, length, nodes (space-separated account ids)
    window_NN.json             written LAST (so a crash leaves no half-finished window): day range, row count,
                               expected row count, per-part timings, betweenness settings
Windows whose JSON exists are skipped, so a crash or restart resumes. Betweenness is cached per window in
results/phase3_features/ibm/betweenness_cache/ (reused by another tau). Windows are exactly those of
configs/ibm.yaml (5 days, stride 1 day, full-length windows only).

Labels, attempts and groups are NOT written here. Join later on (window_id, candidate_id) with the table of
ringtda.ibm_table.ibm_candidate_tables (columns window_id, cand_id, kind, attempt, y, t_seen) and the group
columns saved by experiments/ibm_split_report.py in results/phase3_features/ibm_candidates_<ds>_<tau>.pkl.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from ringtda.data.ibm_aml import iter_ibm_windows, load_config, load_ibm
from ringtda.features import full_feature_table
from ringtda.graph.baseline_features import load_config as load_features_config

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["HI", "LI"], required=True)
ap.add_argument("--tau", type=float, default=0.05)
ap.add_argument("--max-len", type=int, default=6)
ap.add_argument("--out", default=None)
args = ap.parse_args()

cfg = load_config()
ROOT = Path("results") / "phase3_features" / "ibm"
OUT = Path(args.out) if args.out else ROOT / f"{args.dataset}_tau{args.tau}"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "betweenness_cache"
bc = load_features_config()["betweenness"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def atomic_csv(df, path):
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, compression="gzip")
    tmp.replace(path)


# expected candidates per window from the split report (cross-check only)
expected = {}
pk = Path("results/phase3_features") / f"ibm_candidates_{args.dataset}_{float(args.tau)}.pkl"
if pk.exists():
    expected = pd.read_pickle(pk).groupby("window_id").size().to_dict()
    log(f"cross-check: {sum(expected.values())} candidates expected over {len(expected)} windows from {pk.name}")

data = load_ibm(args.dataset)
log(f"{args.dataset}: loaded {len(data.tx)} transactions; out={OUT}")
total, t_start = 0, time.time()
for wid, start, end, w in iter_ibm_windows(data, cfg["window"]["length_days"], cfg["window"]["stride_days"]):
    meta_path = OUT / f"window_{wid:02d}.json"
    if meta_path.exists():
        meta = json.load(open(meta_path))
        total += meta["rows"]
        log(f"window {wid:02d}: exists ({meta['rows']} rows), skipped")
        continue
    t0 = time.time()
    cache_file = CACHE / f"{args.dataset}_w{wid:02d}.pkl"
    cache_hit = cache_file.exists()
    tm = {}
    df, cands = full_feature_table([w], args.tau, args.max_len, return_candidates=True, betweenness_cache=cache_file, timings=tm)
    df["window_id"] = wid
    ids = pd.DataFrame(
        {"window_id": wid, "candidate_id": range(len(cands[0])), "length": [c.length for c in cands[0]], "nodes": [" ".join(map(str, c.nodes)) for c in cands[0]]}
    )
    atomic_csv(df, OUT / f"features_window_{wid:02d}.csv.gz")
    atomic_csv(ids, OUT / f"ids_window_{wid:02d}.csv.gz")
    meta = {
        "dataset": args.dataset, "window_id": wid, "start_day": start, "end_day": end, "n_transactions": len(w),
        "tau_quantile": args.tau, "max_len": args.max_len, "rows": len(df), "expected_rows": expected.get(wid),
        "seconds_total": time.time() - t0, "seconds_by_part": tm, "betweenness": bc, "betweenness_cache_hit": cache_hit,
    }  # fmt: skip
    tmp = meta_path.with_name(meta_path.name + ".tmp")
    json.dump(meta, open(tmp, "w"), indent=1)
    tmp.replace(meta_path)
    total += len(df)
    flag = "" if expected.get(wid) in (None, len(df)) else f"  MISMATCH expected {expected.get(wid)}"
    log(f"window {wid:02d} days {start:.0f}-{end:.0f}: {len(df)} rows{flag}, {meta['seconds_total']:.0f}s "
        f"(graph+cands {tm['graph_candidates']:.0f}s, B {tm['B']:.0f}s, D {tm['D']:.0f}s, T {tm['T']:.0f}s)")  # fmt: skip
exp_total = sum(expected.values()) if expected else None
json.dump({"dataset": args.dataset, "tau": args.tau, "rows": total, "expected_rows": exp_total, "seconds": time.time() - t_start}, open(OUT / "done.json", "w"), indent=1)
log(f"DONE {args.dataset}: {total} rows (expected {exp_total})")
