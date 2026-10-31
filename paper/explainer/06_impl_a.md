## 6. Implementation, stage by stage

### 6.1 Repository map and data flow

```
src/ringtda/
  data/        synthetic.py  ibm_aml.py  toy.py            transactions + ground truth
  graph/       windows.py  weights.py                       windows, per-pair aggregation, scores, weights
               baseline_features.py  direction_features.py  B and D features
  topology/    candidates.py  persistence.py  vectorize.py  rings, persistence, T features
  eval/        labels.py splits.py metrics.py audit.py recall.py
  models/      gd.py  classify.py  cluster.py  regress.py   learning and evaluation models
  threshold/   ascent.py  bandit_sim.py                     adaptive thresholds
  features.py  ibm_table.py  synthetic_table.py             table builders
configs/       topology.yaml features.yaml ibm.yaml classify.yaml bandit.yaml cluster.yaml regress.yaml cap_sensitivity.yaml
experiments/   one script per experiment (+ plot scripts, make_numbers_pack.py)
tests/         tests on tiny synthetic cases for every module
paper/         numbers_pack.md (every number, with provenance), figure_list.md, this explainer
results/       outputs (git-ignored); results/report and results/gd_metrics.json are never overwritten
```

```
transactions --> windows --> per-pair edges --> scores --> rank-normalise --> weight
                                                                   |
        window graph G (weights)                                   v
              |--> sparsify at tau --> SCC --> bounded cycles --> candidate rings
              |                                                   |
              |--> B features (full graph) <----------------------|
              |--> D features (ring edges, ring birth) <----------|
              '--> directed flag persistence (flagser | reference) of the ring's
                   neighbourhood and of the sparsified window --> cap --> landscapes --> T
labels (planted structure | IBM attempt) --> label rule X=80% --> y / kind / group
        B, D, T + y + group --> grouped CV --> classifiers --> pooled out-of-fold scores
                                              --> grouped bootstrap CIs, paired differences
```

### 6.2 Windows, aggregation, scores and weights (`graph/windows.py`, `graph/weights.py`)

`iter_windows` yields transactions in [start, start + length). `aggregate_pairs` collapses a window to one row per ordered pair (src, dst): volume (sum of amounts), count, `gap_cv` (CV of gaps between payments; **needs at least 3 payments**, otherwise undefined), and first and last time.

Scores (all in [0, 1], higher = more suspicious), with V_ab the volume from a to b in the window:

* **Pair reciprocity** `s_rec = 1 - |V_ab - V_ba| / (V_ab + V_ba)`. It is 1 for perfectly balanced two-way trade and 0 for one-way flow.
* **Node flow balance.** For a node x with in-volume in(x) and out-volume out(x): `beta(x) = 1 - |in - out| / (in + out)`; the edge score is `s_bal = min(beta(a), beta(b))`.
* **Timing regularity** `s_reg = 1 / (1 + CV)` where CV is `gap_cv`. Pairs with fewer than 3 payments have no CV; in `weights.compute_scores` they receive the window **median** regularity.

**Rank normalisation.** Each score is mapped to `(rank_min - 1) / (n - 1)` within the window; ties take the *lowest* rank. This matters because most edges have `s_rec = 0`: with average ranks that tied block would land mid-scale and look moderately suspicious. **Weight** `w = 1 - mean(normalised scores)`, so low weight = suspicious.

**Proposition 1 (the one-third floor), why alpha is learned.** In a ring of three or more accounts with no reciprocal pair, the normalised reciprocity score is 0 on every edge, so with weights `w = 1 - alpha . z` and equal alpha = (1/3, 1/3, 1/3) the weight is at least 1/3 no matter how perfect the ring's balance and timing. A multi-party ring can therefore never look maximally suspicious under equal weights. This motivates *learning* alpha on the probability simplex (section 6.10); in the optimization report's experiment the learned alpha is (0.0, 0.2067, 0.7933) for (reciprocity, balance, regularity): zero weight on reciprocity, most on timing regularity.

### 6.3 Sparsification, candidates and de-duplication (`topology/candidates.py`)

`sparsify(G, tau)` keeps edges with weight at most tau. In the table builders tau is a *quantile* of the window's edge weights (default 0.25 on synthetic data; **0.05 on IBM**, so only the most suspicious 5% of edges survive). `candidate_rings` then finds SCCs with at least 2 nodes, enumerates simple directed cycles up to `max_len` = 6 (networkx), caps enumeration at 20000 cycles per SCC and records capped components in `truncated_sccs`. Each cycle becomes a **Candidate**: its accounts in canonical order, its edges, and its **birth** (maximum edge weight on the ring).

**De-duplication, and what the code really does.** The proposal says to keep the lowest-birth cycle among near-duplicates; the code sorts shorter first, then by birth, and drops a cycle when at least 80% of its edges are covered by a kept cycle. **A k-cycle shares at most k-2 edges with any cycle of length at most k, so at 80% overlap nothing is removed for `max_len` at most 9** (first removals at 10; checked in `tests/test_candidates.py`). Dedup is therefore effectively off, near-duplicate rings all stay candidates, and cross-validation must group by ring identity (section 6.8).

### 6.4 Persistence backends (`topology/persistence.py`)

`directed_flag_persistence(G, backend)` returns H0 and H1 diagrams of the weighted directed flag complex of dimension 2: nodes at filtration 0, directed edges at their weight, ordered triples (a, b, c) with a to b, a to c, b to c at the maximum edge weight. Two backends: **flagser** (via pyflagser, fast C++, the engine of Lütgehetmann, Govc, Smith and Levi, "Computing persistent homology of directed flag complexes", Algorithms 2020) and **reference** (pure-Python Z/2 boundary-matrix reduction). They **must agree** (to a tolerance, because flagser works in float32 and differs by about 1e-7); `tests/test_directed_triangles.py` checks that, plus the canonical gates: a cyclic triangle gives one infinite H1 bar, a transitive triangle gives none, a shortcut-filled triangle dies at the fill weight, a reciprocal pair gives an H1 class. The reference backend exists as an auditable cross-check and a fallback.

### 6.5 T features (`topology/vectorize.py`, `configs/topology.yaml`)

**Local.** For each candidate take the induced subgraph on the ring plus its **1-hop** in- and out-neighbours, taken from the *sparsified* graph by default, compute H0 and H1, **cap** and vectorise. **Window.** Compute the diagrams of the whole sparsified window once and repeat the values on each ring of the window (`T_win`).

**Cap rules.** An infinite H1 bar needs a finite death for a landscape. `window_max` (default): the window's maximum edge weight. `fixed`: a constant (the legacy report used 1.0). `window_max_x`: window max times a factor. Infinite bars are *counted before capping* (`inf_h1`), so the count does not depend on the rule; a bar born at or above the cap is dropped from the landscape input but still counted.

**Vector.** 3 landscapes x 50 grid points on [0, 1.1] (150 numbers) plus scalars: H1 count, infinite H1 count, H0 count, total persistence, persistence entropy, and for T_local `ring_inf`. Total T columns: 156 local plus 155 window.

**`T_local_ring_inf`: the exact ring-survival test.** The question "is *this ring's* loop never filled?" is not answered by "there is an infinite bar born near the ring's birth" (a birth-match proxy that misses rings homologous to an earlier loop). The code (`ring_survives`) answers it exactly over Z/2: it writes the ring's loop as a vector over the directed edges of the neighbourhood, builds the boundary matrix of the directed 2-cells (a, b, c) with boundary ab + bc + ac, and checks by Gaussian elimination whether the loop vector lies in the column space. If it does the loop is filled; if not it survives. A size guard (5000 edges, 200000 triangles) falls back to the birth-match proxy and logs it.

### 6.6 D and B features (`graph/direction_features.py`, `graph/baseline_features.py`)

**D (17).** `D_length`, `D_birth`, `D_mean_w`, `D_min_w`, mean and minimum of each normalised score along the ring (`D_rec_*`, `D_bal_*`, `D_reg_*`), member degree imbalance (mean, max), `D_log_volume`, `D_vol_ratio`, `D_vol_cv`, `D_gap_cv_mean`, `D_log_count_mean`. Pairs without a defined `gap_cv` take the **window median of the defined gap CVs** (`gap_cv_fill`): a *different* fill from the weights' median-of-regularity fill (a code detail, section 11.3).

**B (12).** Member degrees (mean and max of total, in and out), betweenness (mean, max), mean clustering, variance of weights in the neighbourhood, density, neighbourhood size, all on the **full** window graph, so B sees at least as much of the graph as T (T uses the sparsified one).

**Betweenness and its sparsity.** Exact betweenness is infeasible on IBM windows (about 400k nodes), so windows with more than 3000 nodes use **k = 100 sampled sources** with a fixed seed, computed once per window and cached. Sampling leaves most nodes at exactly 0: in HI window 0 only a minority of nodes have non-zero betweenness (section 11.2).

**Leakage guard.** The feature builders take only the weighted window graph and a Candidate. They never receive planted structures, labels, attempt ids or scenario names; `tests/test_feature_leakage.py` and the feature-set tests check that no label, key, group or scenario column enters a feature set.
