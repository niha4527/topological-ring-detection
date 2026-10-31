"""Sanity run of the B / D / T feature tables (not a test, not a result). Writes results/phase3_features/.

    python experiments/phase3_features_sanity.py

Synthetic: seeds 0 and 1, hard negatives on and off. IBM HI-Small: window days 0-5, tau_quantile 0.05,
max_len 6. Reports row / column counts, key alignment with the T table, runtimes per part, and the mean of
D_birth and D_bal_mean for positive vs background candidates (labels used ONLY for this printout).
"""

import json
import time
from pathlib import Path

import pandas as pd

import ringtda.features as F
from ringtda.data.ibm_aml import iter_ibm_windows, load_ibm
from ringtda.data.synthetic import SyntheticConfig, generate
from ringtda.eval.labels import label_candidates, label_candidates_ibm
from ringtda.topology.vectorize import load_config

OUT = Path("results") / "phase3_features"
OUT.mkdir(parents=True, exist_ok=True)
SETS = F.feature_sets()
out = {}


def timed(parts, windows, tau=None):
    t = time.time()
    r = F._feature_table(windows, parts, tau, 6, load_config(), None, True)
    return r, time.time() - t


def signs(df, lab, source):
    d = df.assign(y=lab.y.to_numpy(), kind=lab.kind.to_numpy())
    pos, bg = d[d.y == 1], d[d.kind == "background"]
    r = {"n_pos": len(pos), "n_bg": len(bg), "D_birth_pos": pos.D_birth.mean(), "D_birth_bg": bg.D_birth.mean(), "D_bal_mean_pos": pos.D_bal_mean.mean(), "D_bal_mean_bg": bg.D_bal_mean.mean()}
    print(f"  {source}: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    return r


print("=== synthetic ===")
for seed in (0, 1):
    for hn in (False, True):
        tx, planted = generate(SyntheticConfig(seed=seed, hard_negatives=hn))
        (full, cands), t_full = timed({"B", "D", "T"}, [tx])
        (tt, _), t_t = timed({"T"}, [tx])
        (b, _), t_b = timed({"B"}, [tx])
        (d, _), t_d = timed({"D"}, [tx])
        same = set(map(tuple, full[F.KEYS].to_numpy())) == set(map(tuple, tt[F.KEYS].to_numpy()))
        print(f"seed {seed} hard_negatives={hn}: rows {len(full)}, cols B/D/T/total = {len(SETS['B'])}/{len(SETS['B+D']) - len(SETS['B'])}/{len(SETS['T'])}/{full.shape[1] - 2}, keys == T table keys: {same}; time all {t_full:.1f}s, T {t_t:.1f}s, B {t_b:.1f}s, D {t_d:.1f}s")
        lab = label_candidates(cands[0], tx, planted)
        r = signs(full, lab, "synthetic wash_ring vs background")
        out[f"synthetic_seed{seed}_hn{hn}"] = {"rows": len(full), "keys_equal_T": same, "signs": r}
        full.to_csv(OUT / f"bdt_synthetic_seed{seed}_hn{int(hn)}.csv", index=False)

print("=== IBM HI-Small, window days 0-5, tau_quantile 0.05, max_len 6 ===")
data = load_ibm("HI")
_, start, end, w = next(iter_ibm_windows(data))
(_, cands0), t_base = timed(set(), [w], 0.05)  # candidates only: graph build + enumeration
(full, cands), t_full = timed({"B", "D", "T"}, [w], 0.05)
times = {"common (graph+candidates)": t_base}
for name, parts in (("T", {"T"}), ("B", {"B"}), ("D", {"D"})):
    _, tt = timed(parts, [w], 0.05)
    times[name] = tt
print(f"rows {len(full)}, cols {full.shape[1] - 2}, time all {t_full:.1f}s; per part incl. common {times}; per part excl. common " + ", ".join(f"{k} {v - t_base:.1f}s" for k, v in times.items() if k != "common (graph+candidates)"))
attempts = dict(zip(data.attempts["attempt_id"], data.attempts["pattern"]))
lab = label_candidates_ibm(cands[0], w, attempts)
r = signs(full, lab, "IBM positive vs background")
out["ibm_hi_window0"] = {"rows": len(full), "time_all": t_full, "time_by_part": times, "signs": r}
full.to_csv(OUT / "bdt_ibm_HI_window0_q0.05.csv", index=False)
json.dump(out, open(OUT / "sanity_summary.json", "w"), indent=1, default=float)
