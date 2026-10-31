"""Candidate-generation recall across the tau_quantile sweep (proposal v3 §3.3). Report only; no features, no models.

    nice -n 10 python experiments/candidate_recall.py --source ibm
    nice -n 10 python experiments/candidate_recall.py --source synthetic

ibm: HI-Small and LI-Small, tau_quantile {0.02, 0.05, 0.10, 0.15} x max_len {6, 8}; recall of complete CYCLE
attempts (src/ringtda/eval/recall.py) for window 0 (days 0-5) and for the whole data, plus candidates per window.
synthetic: recall of planted wash rings (isolated / dense / chorded) over a tau sweep, hard negatives on/off,
dense_internal_chords on/off, seeds 0-5.
Output: results/phase3_features/classify/candidate_recall/<source>.csv and <source>.png (drawn from the csv).
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from ringtda.eval.recall import cycle_attempts, ibm_recall, synthetic_recall

ap_ = argparse.ArgumentParser()
ap_.add_argument("--source", choices=("ibm", "synthetic"), required=True)
ap_.add_argument("--taus", nargs="+", type=float, default=None)
ap_.add_argument("--seeds", type=int, default=6)
args = ap_.parse_args()
OUT = Path("results/phase3_features/classify/candidate_recall")
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
rows = []

if args.source == "ibm":
    from ringtda.data.ibm_aml import iter_ibm_windows, load_config, load_ibm
    from ringtda.ibm_table import ibm_candidate_tables

    cfg = load_config()
    taus = args.taus or [0.02, 0.05, 0.10, 0.15]
    L = cfg["window"]["length_days"]
    for ds in ("HI", "LI"):
        data = load_ibm(ds)
        att = cycle_attempts(data.tx)
        starts = {wid: s for wid, s, _, _ in iter_ibm_windows(data, L, cfg["window"]["stride_days"])}
        print(f"{ds}: {len(att)} CYCLE attempts, {len(starts)} windows", flush=True)
        for ml in (6, 8):
            tabs = ibm_candidate_tables(data, cfg, taus, ml)
            for q, tab in tabs.items():
                for scope, wids in (("days0-5", [0]), ("whole_data", None)):
                    r = ibm_recall(att, tab, starts, L, ml, wids)
                    rows.append({"source": ds, "tau_quantile": q, "max_len": ml, "scope": scope, **r})
                    print(f"  {ds} ml={ml} q={q} {scope}: complete {r['n_complete']} (<=max_len {r['n_complete_le_maxlen']}) recovered {r['n_recovered']}, cand/window {r['candidates_per_window']:.0f}", flush=True)
else:
    from ringtda.data.synthetic import SyntheticConfig, generate
    from ringtda.features import window_graphs
    from ringtda.topology.candidates import candidate_rings

    taus = args.taus or [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    for hn in (False, True):
        for chords in (True, False):
            for seed in range(args.seeds):
                tx, pl = generate(SyntheticConfig(seed=seed, hard_negatives=hn, dense_internal_chords=chords))
                for q in taus:
                    _, G, _, tau = window_graphs(tx, q)
                    res = candidate_rings(G, tau=tau, max_len=6)
                    r = synthetic_recall(res.candidates, tx, pl)
                    for pid, sc, hit in r["rings"]:
                        rows.append({"source": "synthetic", "hard_negatives": hn, "dense_internal_chords": chords, "seed": seed, "tau_quantile": q, "max_len": 6,
                                     "planted_id": pid, "scenario": sc, "recovered": int(hit), "n_candidates": r["n_candidates"], "n_truncated_sccs": len(res.truncated_sccs)})  # fmt: skip
                print(f"hn={hn} chords={chords} seed={seed} done ({time.time() - t0:.0f}s)", flush=True)

R = pd.DataFrame(rows)
R.to_csv(OUT / f"{args.source}.csv", index=False)
print(f"done in {time.time() - t0:.0f}s -> {OUT / (args.source + '.csv')}", flush=True)
