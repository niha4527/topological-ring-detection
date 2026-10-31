# Topological Ring Detection for Market Manipulation

Code for the project described in `Topological_Ring_Detection_for_Market_Manipulation_v3.md`.

## One-time setup (macOS Terminal)

Python 3.11 via conda (pyflagser has no wheels for Python 3.13):

```bash
cd ~/Documents/RVU/SEM3/AIML/AIML-PROJECT
conda create -n ringtda python=3.11 -y
conda activate ringtda
pip install -e ".[dev]"
```

Later, for Phase 2 (persistence landscapes): `pip install -e ".[tda,dev]"` (adds gudhi).

## Every session

```bash
cd ~/Documents/RVU/SEM3/AIML/AIML-PROJECT
conda activate ringtda
pytest
```

In VS Code: open this folder, install the **Python** and **Jupyter** extensions, and pick `ringtda` as
the interpreter / notebook kernel. Then open `notebooks/01_sanity_visuals.ipynb` and Run All.

## Quick check that the topology library works

```bash
python sanity_check.py
```

Expected: the cyclic triangle has one H1 bar `[0.1, inf]`; the transitive triangle has none.

## Layout

```
src/ringtda/
  data/        synthetic.py (planted rings, topology-sensitive and hard negatives), toy.py (hand-built test graphs),
               ibm_aml.py (IBM AML loader, attempt join, 5-day windows)
  graph/       windows.py (sliding windows, per-pair aggregation), weights.py (s_rec, s_bal, s_reg -> weight)
  topology/    candidates.py (SCC + cycle enumeration), persistence.py (directed flag persistence),
               vectorize.py (infinite-bar cap, landscapes, scalar summaries; local and window T features)
  viz.py       plotting helpers for notebooks
  features.py  candidate table used by the optimization report (legacy topo_* columns) and t_feature_table
configs/       topology.yaml (landscape settings, cap rule, local graph)
  models/      gd.py (gradient descent, alpha learner)    threshold/  ascent.py (threshold policies)
  eval/        labels.py (§3.7 ring-labeling rule, synthetic and IBM), splits.py (groups, grouped CV, transfer)       motif/      later phases
experiments/   gd_report.py — optimization report; `--config report` reproduces the report's numbers
tests/         engineering gates: directed-triangle behaviour, weights, candidates, planted rings,
               generator scenarios (test_synthetic_scenarios.py)
notebooks/     exploration and figures only — import from src/, never define pipeline logic here
data/          datasets (git-ignored, except data/sample/)
results/       figures and metrics (git-ignored). results/report/ (+ the originals at results/gd_metrics.json,
               results/figures/) = the optimization report's numbers: do not overwrite.
               results/full/<ambiguous>/ = the same experiments on the full generator
```

## Rules of thumb

- Pipeline logic goes in `src/ringtda/*.py`; notebooks only call it and plot.
- Every new module gets a test on a tiny synthetic case first.
- Lower edge weight = more suspicious (sublevel-set filtration).
- Persistence backends: `flagser` (fast, default) and `reference` (pure Python, slow; used to
  cross-check flagser in `tests/test_directed_triangles.py` and as a fallback).

## Status

- [x] Phase 1: synthetic generator — isolated / dense-neighbourhood / chorded wash rings; transitive
      decoys and shortcut-filled pseudo-rings; hard negatives (arbitrage loops, market makers, hot
      wallets; `SyntheticConfig(hard_negatives=False)` for the "without" runs); evasion knobs
      (`wash_decoy_frac`, `mm_regularity`); windows; weights
- [x] Phase 2 (part): candidate rings; directed flag persistence; gates as tests
- [x] Candidate labels from planted ground truth (§3.7 rule, `eval/labels.py`)
- [x] Phase 2 topology features: local + window-level persistence, infinite-bar cap, landscapes and scalar
      summaries (`topology/vectorize.py`, `configs/topology.yaml`), T feature table (`features.t_feature_table`)
- [x] IBM AML loader, ring labels (§3.7), groups and fold assignments; split counts in
      `experiments/ibm_split_report.py`
- [x] Phase 3: B and D feature code, resumable IBM feature builder, IBM AML pattern-behaviour audit
- [x] Phase 3: classification harness (`experiments/ibm_classify.py`, grouped CV, grouped-bootstrap CIs, fixed
      hyperparameters); runs on HI-Small and LI-Small, T_local / T_win ablations, HI<->LI transfer
      (`experiments/ibm_transfer.py`), synthetic per-scenario runs (`experiments/synth_classify.py`)
- [x] Candidate-generation recall (`experiments/candidate_recall.py`) and the results numbers pack
      (`experiments/make_numbers_pack.py` -> `paper/numbers_pack.md`, figure list in `paper/figure_list.md`)
- [x] Phase 6: bandit threshold policies (discounted UCB, sliding-window Thompson) and the drift benchmark
      (`threshold/bandit_sim.py`, `configs/bandit.yaml`, `experiments/bandit_report.py`; results in
      `results/phase6_bandit/`, git-ignored)
- [x] Phase 4 (clustering only): Ward clustering of IBM positive rings against pattern types
      (`models/cluster.py`, `configs/cluster.yaml`, `experiments/ibm_cluster.py`; results in `results/phase4_cluster/`)
- [ ] Write-up (paper). Severity regression and Phase 5 (NFT loader and label-provenance audit) are future work,
      not part of this iteration

## Decisions and deviations from the proposal

- **Candidate de-duplication (§3.3).** The proposal keeps "the lowest-birth one"; with that rule a
  planted dense-neighbourhood ring could be replaced by a detour through a neighbour that shares 4/5
  of its edges and is born slightly earlier. Candidates are now de-duplicated shortest-first, then by
  birth. Consequence: a k-cycle shares at most k−2 edges with any cycle of length ≤ k, so at the
  default overlap 0.8 nothing is dropped unless `max_len` ≥ 10. A different overlap measure is
  needed if real de-duplication is wanted.
- **Candidate labels (§3.7).** Synthetic candidates use the IBM AML ring-labeling rule with a planted
  structure as the "attempt": the structure's kind if ≥ X% (default 80%) of the ring's edges are
  backed by its own transactions, `background` if none are backed, else `ambiguous`. Ambiguous
  candidates are counted per window and results are reported both excluding them and treating them
  as negatives. For planted wash rings of size 2/3/5 with `max_len` ≤ 6, and for star-shaped
  structures (market makers, hot wallets), X = 80% and X = 100% give identical labels, so only the
  ambiguous class changes things. The exception is structures with more edges than one cycle, such
  as chorded 6-rings: e.g. 1→2→3→4→5 plus a background edge 5→1 is 4/5 backed (labelled at 80%,
  ambiguous at 100%).
- **Dedup is effectively off (§3.3).** At the default 80% overlap nothing is removed for
  `max_len` <= 9 and the first removals happen at 10 (verified on dense windows;
  `tests/test_candidates.py::test_dedup_removes_nothing_at_default_overlap`), so
  near-duplicate rings all stay as candidates. Phase 3 cross-validation must group by ring identity,
  including near-duplicate rings, not just by window.
- **Two T pipelines.** The optimization report uses the legacy scalar `topo_*` columns of
  `window_candidates` (fixed infinite-bar cap 1.0, frozen, report-only). The Phase 3 T table
  (`features.t_feature_table`) uses the window-max cap, landscapes and window-level features from
  `topology/vectorize.py` and `configs/topology.yaml`. New code uses the T table.
- **Dense scenario fills most planted rings (default).** With `dense_internal_chords=True` (the
  default, unchanged so the report reproduces), only 7 of 24 planted dense rings (seeds 0-5, `max_len`
  6) have `T_local_ring_inf = 1`. Cause split of the 17 filled rings: 6 inside only (incidental edges
  among the ring's own accounts), 4 outside accounts only, 1 either alone, 6 only in combination
  (`results/phase2_features/dense_fill_breakdown.csv`). `SyntheticConfig(dense_internal_chords=False)`
  removes the inside edges (same random stream otherwise) and raises the fraction to 18 of 24 (75%);
  the remaining filling is by outside accounts, which is kept as realistic (§3.4). Phase 3 stratified
  results report dense rows under BOTH settings. A planted positive with `T_local_ring_inf = 0` keeps
  its positive label: the column is a feature, not a label.

## Phase 3 design decisions (IBM AML, HI-Small; LI-Small replicates with the same rules)

1. **Windows and evaluation scheme.** 5-day windows, 1-day stride (cycle attempts last up to 96 h, so a
   shorter window cuts them). The primary result is grouped cross-validation; HI<->LI transfer is the
   out-of-sample check. The time-ordered check was tried and dropped (below). The original idea (train windows ending by day 8, 4-day gap,
   test windows from day 12) was dropped: every laundering attempt in HI-Small starts on days 0-11 (cycle
   attempts on days 0-9) and laundering rows are nearly gone after day 13, so such test windows contain no
   positives (LI-Small is the same, attempts on days 0-10).
   - **Groups.** One fold unit per group, assigned once across all windows: union-find over candidates
     that share any edge, merged with the attempt id of every candidate labelled with an attempt (so a
     near-duplicate ring, which dedup does not remove, and a ring seen in several overlapping windows stay
     together). A stricter variant groups by shared account; it is reported as a sensitivity check.
   - **No leakage from tuning.** All scaling, hyperparameter and threshold choices happen inside the
     training folds only (nested where a choice is tuned).
   - **Primary estimate.** Stratified group k-fold, 5 repeats; pooled out-of-fold PR-AUC with grouped
     bootstrap CIs (resampling groups); prevalence is always reported next to PR-AUC, with the group counts.
     Measured: only 10-16 positive groups per dataset (HI-Small 13-16, LI-Small 10 across
     `tau_quantile` 0.02-0.10), and 144 of 150 fold-repeats have fewer than 5 positive groups (50 of 50 at
     `tau_quantile` 0.05; `results/phase3_features/ibm_split_counts.csv`, 3 taus x HI/LI x 5 folds x 5
     repeats). The effective sample
     size is the number of positive groups, not the number of positive candidates (e.g. 131 positive
     candidates from 43 attempts fall into 14 groups at `tau_quantile` 0.05). Expect wide CIs; they are
     reported as they are, and nothing is tuned to narrow them.
   - **Time-ordered check: tried as specified, unusable, dropped.** Train = groups first seen on days
     0-5, test = groups first seen on day >= 7, windows holding both sides dropped. HI-Small leaves 0 test
     candidates and LI-Small 1 (HI-Small: 7 of 13 windows dropped, 6 kept hold no test-side candidate),
     because rings recur: almost every window mixes early-first-seen and late-first-seen groups. It is not
     an evaluation scheme here and is reported as unusable. `eval/splits.time_ordered_split` and its tests
     stay in the code. No post-hoc variant (for example dropping individual candidates) is used.
   - **HI <-> LI transfer** (train on one, test on the other) is the out-of-sample check, labelled as
     transfer, not as a time-leak-free test. Checked: the two datasets share 3 of 705,907 LI accounts (HI has 515,088), so
     they are essentially, not perfectly, account-disjoint; both come from the same generator and cover
     the same dates.
   - **Positive counts.** Reported by pattern type, plus a cycle-only secondary subset (labels from
     CYCLE attempts only). The subset is usable on HI-Small only (35-41 cycle positives in 11-14 groups);
     LI-Small has 3-4 cycle positives (3 groups). The §3.7 labeling rule is not changed on the basis of these counts: any ring
     with >= 80% of its edges from a single attempt is positive, whatever the attempt type.
2. **Currencies.** `Amount Paid` is used as is, in its own currency (15 currencies, no exchange rates in
   the data; 72,158 rows have received != paid). Volume ratios (s_rec, s_bal) mix currencies. Limitation.
3. **Self-loops.** Rows with source == destination are dropped (591,212 of 5,078,345 in HI-Small, 11.6%);
   the count is logged by the loader. Effect: node in/out totals (hence s_bal) use inter-account flow only.
4. **Unattributed laundering rows.** 1,968 of the 5,177 laundering rows are in no attempt of the patterns
   file. A candidate whose laundering-flagged edges rest only on those rows is `ambiguous`. Results are
   reported with ambiguous excluded and with ambiguous counted positive (v3 §3.7).
5. `data/ibm_aml/ibm.zip` (8 GB) is kept. Only the `HI-Small_*` and `LI-Small_*` files are extracted;
   `data/` is git-ignored.

Measurements behind these (HI-Small, window = days 0-5, 2.2 M transactions after dropping self-loops):
aggregate + weights + graph 5.7 s, candidate enumeration <= 0.8 s, `t_feature_table` (q = 0.05,
`max_len` 6) 11.5 s for 1,113 candidates. 54 CYCLE attempts in total, all simple rings; 34 have <= 6
accounts. In that window 15 attempts are complete, 11 of them with <= 6 accounts (the recall cap at
`max_len` 6) and 12 with <= 8 (cap at 8). Recovered as candidates (a candidate whose account set equals the
attempt's): 8 for `tau_quantile` 0.02-0.10 and 9 at 0.15, identical at `max_len` 6 and 8. Under the stricter
reading that the recovering candidate must also carry a positive CYCLE label (§3.7, >= 80% of its edges from
the attempt), the counts are 6 for 0.02-0.10 and 7 at 0.15: two of the recovered rings are labelled otherwise
(`results/phase3_features/classify/candidate_recall/ibm.csv`, columns `n_recovered` and `n_recovered_labelled`).
IBM `tau_quantile` sweep range: about 0.02 to 0.15.

## Feature sets B, D, T (Phase 3)

`features.full_feature_table` builds B, D and T on the same windows and candidates, keyed by
`(window_id, candidate_id)`; `features.feature_sets()` lists the columns of B, B+D, B+D+T and T. The
builders take only graphs, pairs and candidates (never labels, planted structures or attempt ids).
- **D** (`graph/direction_features.py`, 17 columns): ring birth (`D_birth`), length, edge weights, mean/min
  of each rank-normalised score, member degree imbalance, volume and timing statistics.
- **B** (`graph/baseline_features.py`, 12 columns): member degrees, betweenness, clustering, neighbourhood
  weight variance, density and size, on the FULL window graph with the same 1-hop neighbourhood as T. T takes
  its neighbourhood from the sparsified graph by default, so B sees at least as much of the graph as T.
- **Betweenness.** Exact for windows with <= 3000 nodes (synthetic), otherwise k = 100 sampled sources with a
  fixed seed on the full window graph, cached per window (`configs/features.yaml`). Timing on an IBM HI-Small
  5-day window (404k nodes, 554k edges): 17.8 s for k = 50 and 34.8 s for k = 100. Stored k = 100 caches
  (`results/phase3_features/ibm/betweenness_cache/`): HI window 0 has 64,441 of 404,273 nodes with non-zero
  betweenness (15.9%), LI window 0 has 17,673 of 551,969 (3.2%); most nodes get 0, so B betweenness is sparse
  on IBM windows. (An earlier statement that k = 50 gave non-zero values for 166 nodes is not reproducible from
  any stored file and was removed.) The whole B step for that window is about 82 s (betweenness plus
  neighbourhood extraction and clustering around hubs).
