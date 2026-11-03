import resource, sys, time
from pathlib import Path
import numpy as np
from ringtda.data.ibm_aml import iter_ibm_windows, load_ibm_files
from ringtda.features import window_graphs
from ringtda.graph.baseline_features import WindowBaseline, baseline_features, load_config as lf
from ringtda.graph.direction_features import direction_features, gap_cv_fill
from ringtda.topology.candidates import candidate_rings
from ringtda.topology.vectorize import load_config as lt, local_graph, local_topology_features, window_topology_features

D = Path(sys.argv[1]); T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30:.2f}GB]", *a, flush=True)
d = load_ibm_files(D / "HI-Medium_Trans.csv", D / "HI-Medium_Patterns.txt", "HI-Medium")
wins = list(iter_ibm_windows(d)); tx = wins[0][3]; del d
pairs, G, Gs, tau = window_graphs(tx, 0.05); res = candidate_rings(G, tau=tau, max_len=6); cands = res.candidates
cfg = lt(); n = len(cands); log(f"w0: {G.number_of_nodes()} nodes {G.number_of_edges()} edges, sparsified {Gs.number_of_edges()} edges, {n} candidates")
wmax = float(max(dd["weight"] for _, _, dd in G.edges(data=True)))
t = time.time(); win = window_topology_features(Gs, wmax, cfg); log(f"T_win (whole sparsified window): {time.time()-t:.1f}s")
LG = local_graph(G, Gs, cfg)
rng = np.random.default_rng(0); samp = [cands[i] for i in rng.choice(n, 150, replace=False)]
t = time.time()
for c in samp: local_topology_features(LG, c, wmax, cfg)
tl = (time.time()-t)/150; log(f"T_local: {tl*1000:.0f} ms/candidate -> x{n} = {tl*n:.0f}s")
t = time.time(); fill = gap_cv_fill(G)
for c in samp: direction_features(G, c, fill)
td = (time.time()-t)/150; log(f"D: {td*1000:.1f} ms/candidate -> x{n} = {td*n:.0f}s")
t = time.time(); ctx = WindowBaseline(G, lf()); log(f"B betweenness (k=100 sampled, once per window): {time.time()-t:.1f}s")
t = time.time()
for c in samp: baseline_features(G, c, ctx, cfg["local"]["hops"])
tb = (time.time()-t)/150; log(f"B per-candidate: {tb*1000:.0f} ms/candidate -> x{n} = {tb*n:.0f}s")
log("done")
