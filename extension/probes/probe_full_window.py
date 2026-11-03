import resource, sys, time
from pathlib import Path
import numpy as np
from ringtda.data.ibm_aml import iter_ibm_windows, load_ibm_files
from ringtda.features import full_feature_table, window_graphs
from ringtda.topology.candidates import candidate_rings

D = Path(sys.argv[1]); T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30:.2f}GB]", *a, flush=True)

d = load_ibm_files(D / "HI-Medium_Trans.csv", D / "HI-Medium_Patterns.txt", "HI-Medium")
log("loaded", {k: d.stats[k] for k in ("n_rows", "n_self_loops", "n_attempts", "t_max_day", "n_laundering")})
wins = list(iter_ibm_windows(d))
sizes = [len(w[3]) for w in wins]
log("windows:", len(wins), "tx per window min/median/max:", min(sizes), int(np.median(sizes)), max(sizes))
for wid in (0, len(wins) // 2):
    _, s, e, tx = wins[wid]
    t = time.time(); pairs, G, Gs, tau = window_graphs(tx, 0.05)
    log(f"w{wid} days {s:.0f}-{e:.0f}: {len(tx)} tx, {G.number_of_nodes()} nodes, {G.number_of_edges()} edges; graph+weights {time.time()-t:.1f}s")
    t = time.time(); res = candidate_rings(G, tau=tau, max_len=6)
    log(f"w{wid}: {len(res.candidates)} candidates, sccs {res.n_sccs}, truncated {res.truncated_sccs}; enumeration {time.time()-t:.1f}s")
    tm = {}
    t = time.time(); df = full_feature_table([tx], tau_quantile=0.05, max_len=6, timings=tm)
    log(f"w{wid}: full B+D+T table {df.shape} in {time.time()-t:.1f}s; parts {{{', '.join(f'{k}: {v:.1f}s' for k, v in tm.items())}}}")
log("done")
