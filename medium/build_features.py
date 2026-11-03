"""Resumable B / D / T feature builder for IBM AML Medium (HI-Medium or LI-Medium).

Adapted from experiments/ibm_feature_build.py, which only reads the Small files and writes into results/.
This script reads from medium/data/ and writes ONLY inside medium/out/, so the existing results cannot be
touched. Same pipeline code (ringtda), same settings (configs/ibm.yaml, configs/features.yaml,
configs/topology.yaml), same window rule (5 days, stride 1 day, full-length windows).

    python -I medium/build_features.py --dataset HI --tau 0.05 --max-len 6

One output set per window in medium/out/<dataset>_tau<tau>/:
    features_window_NN.csv.gz  keys (window_id, candidate_id) + B + D + T columns, no label columns
    ids_window_NN.csv.gz       window_id, candidate_id, length, nodes (space-separated account ids)
    window_NN.json             written LAST, so a crash leaves no half-finished window
Windows whose JSON exists are skipped: stop and restart at any time and it resumes.
Labels, attempts and groups are NOT written here (join later on window_id, candidate_id).
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from ringtda.data.ibm_aml import iter_ibm_windows, load_config, load_ibm_files
from ringtda.features import full_feature_table
from ringtda.graph.baseline_features import load_config as load_features_config

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["HI", "LI"], required=True)
ap.add_argument("--tau", type=float, default=0.05)
ap.add_argument("--max-len", type=int, default=6)
ap.add_argument("--max-windows", type=int, default=None, help="stop after this many windows (testing)")
args = ap.parse_args()

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "out" / f"{args.dataset}_tau{args.tau}"
CACHE = HERE / "out" / "betweenness_cache"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
cfg = load_config()
bc = load_features_config()["betweenness"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def atomic_csv(df, path):
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, compression="gzip")
    tmp.replace(path)


name = f"{args.dataset}-Medium"
data = load_ibm_files(DATA / f"{name}_Trans.csv", DATA / f"{name}_Patterns.txt", name)
log(f"{name}: loaded {len(data.tx)} transactions; out={OUT}")
total, t_start, done = 0, time.time(), 0
for wid, start, end, w in iter_ibm_windows(data, cfg["window"]["length_days"], cfg["window"]["stride_days"]):
    meta_path = OUT / f"window_{wid:02d}.json"
    if meta_path.exists():
        meta = json.load(open(meta_path))
        total += meta["rows"]
        log(f"window {wid:02d}: exists ({meta['rows']} rows), skipped")
        continue
    if args.max_windows is not None and done >= args.max_windows:
        break
    t0 = time.time()
    cache_file = CACHE / f"{args.dataset}M_w{wid:02d}.pkl"
    tm = {}
    df, cands = full_feature_table([w], args.tau, args.max_len, return_candidates=True, betweenness_cache=cache_file, timings=tm)
    df["window_id"] = wid
    ids = pd.DataFrame(
        {"window_id": wid, "candidate_id": range(len(cands[0])), "length": [c.length for c in cands[0]], "nodes": [" ".join(map(str, c.nodes)) for c in cands[0]]}
    )
    atomic_csv(df, OUT / f"features_window_{wid:02d}.csv.gz")
    atomic_csv(ids, OUT / f"ids_window_{wid:02d}.csv.gz")
    meta = {
        "dataset": name, "window_id": wid, "start_day": start, "end_day": end, "n_transactions": len(w),
        "tau_quantile": args.tau, "max_len": args.max_len, "rows": len(df),
        "seconds_total": time.time() - t0, "seconds_by_part": tm, "betweenness": bc,
    }  # fmt: skip
    tmp = meta_path.with_name(meta_path.name + ".tmp")
    json.dump(meta, open(tmp, "w"), indent=1)
    tmp.replace(meta_path)
    total += len(df)
    done += 1
    log(f"window {wid:02d} days {start:.0f}-{end:.0f}: {len(df)} rows, {meta['seconds_total']:.0f}s "
        f"(graph+cands {tm['graph_candidates']:.0f}s, B {tm['B']:.0f}s, D {tm['D']:.0f}s, T {tm['T']:.0f}s)")  # fmt: skip
if args.max_windows is None:
    json.dump({"dataset": name, "tau": args.tau, "rows": total, "seconds": time.time() - t_start}, open(OUT / "done.json", "w"), indent=1)
    log(f"DONE {name}: {total} rows")
else:
    log(f"stopped after {done} new window(s) (--max-windows)")
