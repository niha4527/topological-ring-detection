# Architecture

Technical architecture of `ringtda`, the topological ring detector for market manipulation. The design source is `Topological_Ring_Detection_for_Market_Manipulation_v3.md` (section numbers such as §3.1 refer to it). This file describes what the code does today and where the remaining stages plug in.

Status legend: **[done]** implemented and tested, **[partial]** some of it exists, **[todo]** not started.

## 1. System overview

One windowed, directed transaction graph feeds a seven-stage pipeline. Each stage is a separate package under `src/ringtda/` and can be tested alone.

```
 transactions (t, sender, receiver, amount)
        |
 [1 data]      synthetic.py [done]   ibm_aml.py [done]   nft_trades.py [todo]
        |
 [2 graph]     windows.py -> weights.py -> baseline features (B) [done]
        |         sliding windows    s_rec, s_bal, s_reg -> rank-normalize -> w = 1 - mean
        |
 [3 topology]  candidates.py          persistence.py            vectorize.py [done]
        |       sparsify -> SCC ->     directed flag complex,    landscapes, infinite-bar
        |       bounded cycles          flagser | reference      cap, undirected ablation
        |
 [features]    features.py [done]: B, D and T tables per candidate (+ frozen legacy report path)
        |
 [5 models]    models/gd.py [done: logistic GD, simplex alpha learner]
        |       classify.py cluster.py regress.py [done]
        |
 [6 threshold] threshold/ascent.py [done: online gradient ascent + comparators]
        |       bandit_sim.py (discounted UCB, sliding-window Thompson) [done]
        |
 [7 eval]      eval/labels.py splits.py metrics.py audit.py recall.py [done]   ablations.py [todo]

 [4 motif]     pump-and-dump track, optional stretch [todo]
```

Rules that shape the design (from `CLAUDE.md`):
- Pipeline logic lives in `src/ringtda/`. Notebooks import and plot only.
- Lower edge weight means more suspicious (sublevel-set filtration).
- Classical ML only (scikit-learn). No deep learning.
- Engineering gates are hard tests. Model performance is reported with confidence intervals and is never a test that must pass.

## 2. Data flow

1. **Ingest** a transaction table with columns time, sender, receiver, amount. The synthetic generator also returns a list of `Planted` structures as ground truth.
2. **Window.** `iter_windows` slices the table with window length `L` and stride `S`. `aggregate_pairs` collapses each window to one row per ordered account pair: volume `V_ab`, count `n_ab`, and the CV of inter-transaction gaps.
3. **Score and weight.** `compute_scores` gives `s_rec`, `s_bal`, `s_reg`. `rank_normalize` maps each to its empirical rank in the window (ties take the lowest rank). `edge_weights` combines them (equal weights by default, or a learned `alpha`). `to_digraph` returns a `networkx.DiGraph` with a `weight` attribute.
4. **Localize.** `candidate_rings` sparsifies at cutoff `tau_s`, finds SCCs, enumerates simple directed cycles up to `max_len`, and de-duplicates. Output is `CandidateResult` holding `Candidate` objects (nodes, edges, length, birth).
5. **Persist.** `neighborhood` extracts the ring plus its 1-hop neighbourhood. `directed_flag_persistence` returns `{dim: array of (birth, death)}` for H0 and H1.
6. **Featurize.** `features.window_candidates` returns one table row per candidate plus the per-edge score arrays `Z`.
7. **Model.** Feature rows go to the classifier, clusterer, regressor. Ring birth or the classifier score goes to the threshold policy.
8. **Evaluate.** Grouped CV (groups = shared edges merged with attempt id), pooled out-of-fold PR-AUC with grouped bootstrap CIs, HI<->LI transfer, per-scenario and per-feature-set reports.

## 3. Module reference

### 3.1 `data/`
| File | Role | Status |
|---|---|---|
| `synthetic.py` | `SyntheticConfig` and `generate(cfg)`. Heavy-tailed background plus planted structures. Positives: isolated, dense-neighbourhood and chorded wash rings. Topology-sensitive negatives: transitive decoys, shortcut-filled pseudo-rings. Hard negatives: arbitrage loops, market makers, hot wallets. Knobs: `hard_negatives`, `wash_decoy_frac`, `mm_regularity`, jitter. | done |
| `toy.py` | Hand-built graphs for gate tests: cyclic triangle, transitive triangle, reciprocal pair, shortcut-filled triangle, square. | done |
| `ibm_aml.py` | AMLworld loader (`load_ibm`, `load_ibm_files`): joins laundering attempts to transactions on (timestamp, banks, accounts, amount paid) with deterministic handling of duplicate keys, attaches attempt id and pattern type, drops and counts self-loops, accounts keyed by (bank, account), time in days; `iter_ibm_windows` (5-day windows, 1-day stride) feeds the existing aggregation. Settings in `configs/ibm.yaml`. | done |
| `src/ringtda/ibm_table.py` (package root, not in `data/`) | `ibm_candidate_tables`: candidates, ring labels and first-seen day per (window, `tau_quantile`), reusing `aggregate_pairs`, `edge_weights` and `candidate_rings`. No features yet. | done |
| `nft_trades.py` | Dune / BigQuery export loader. | todo |

### 3.2 `graph/`
| File | Role | Status |
|---|---|---|
| `windows.py` | `iter_windows`, `aggregate_pairs`, `_gap_cv`. | done |
| `weights.py` | `compute_scores`, `rank_normalize`, `edge_weights(pairs, scores)`, `to_digraph`. Per-score ablations via the `scores` argument. | done |
| `baseline_features.py` | B set (12 columns) on the full window graph: member degrees, betweenness (exact up to 3000 nodes, else k = 100 sampled, cached per window), clustering, neighbourhood weight variance, density, size. | done |
| `direction_features.py` | D set (17 columns): ring birth, length, weights, rank-normalised score mean/min, degree imbalance, volume and timing statistics. Label-free by signature. | done |

Known property (Proposition 1): with equal weights, a multi-party ring has `z_rec = 0` on every edge, so its edge weight cannot fall below 1/3. This is why `alpha` is learned.

### 3.3 `topology/`
| File | Role | Status |
|---|---|---|
| `candidates.py` | `sparsify`, `candidate_rings(G, tau, max_len)`, `neighborhood(G, nodes, hops)`. Dedup is shortest-first then lowest birth (deviation from §3.3, see README). | done |
| `persistence.py` | `directed_flag_persistence(G, weight, backend)`. Backends `flagser` (default, float32) and `reference` (pure-Python Z/2 boundary-matrix reduction). They must agree within tolerance. Complex capped at dimension 2. | done |
| `vectorize.py` | T features. Local persistence of ring + 1-hop neighbourhood (`local_diagrams`), window-level persistence of the sparsified graph (`window_topology_features`, once per window), infinite-bar cap (rule in `configs/topology.yaml`, default window max), `gudhi` landscapes (3 x 50, range [0, 1.1]), scalar summaries, `T_local_ring_inf`. | done |
| undirected ablation | `gudhi` clique complex for RQ4. | todo |

### 3.4 `features.py` [done]
- `window_graphs(tx, tau_quantile)` gives the pairs, full graph, sparsified graph and `tau_s`.
- `t_feature_table(windows, tau_quantile=None, max_len=6, ...)` is the Phase 3 T table: one row per candidate keyed by `(window_id, candidate_id)`, 311 T columns (`T_local_*` from `vectorize.py`, `T_win_*` computed once per window), fixed width, no NaNs, T only. `T_local_inf_h1` counts all infinite H1 bars in the neighbourhood; `T_local_ring_inf` is exact: 1 if the ring's cycle (Z/2 vector over the directed edges of its local graph) is NOT in the column space of the Z/2 boundary matrix of the directed 2-simplices, i.e. the loop is never filled. Ring membership only picks the cycle; ring birth is not used. It ignores filtration values, is backend-independent, and falls back to a birth-match proxy (logged) if a neighbourhood exceeds `local.ring_inf_limits`.
- `window_candidates(tx, planted, tau_quantile=0.25, max_len=5, topo=True, label_x=0.8)` is the **report path**. Its D-like columns are `length`, `birth_eq`, mean and min scores. Its `topo_*` columns are **legacy: report-only, fixed cap 1.0, superseded by the window-max cap** in `vectorize.py`. They ignore `configs/topology.yaml`. New code must not use them.

`bd_feature_table` and `full_feature_table` build B, D and T on the same windows and candidates (keys `(window_id, candidate_id)`, shared loop with `t_feature_table`); `feature_sets()` gives the column lists for B, B+D, B+D+T and T. Leakage guard: `tests/test_feature_leakage.py`.

### 3.5 `eval/`
| File | Role | Status |
|---|---|---|
| `labels.py` | §3.7 ring-labeling rule. `edge_backing`, `label_ring`, `label_candidates` (synthetic: positive if at least X% (default 80) of a ring's edges are backed by one planted structure, background if none, else ambiguous, `y = NaN`) and `label_candidates_ibm` (attempts of any pattern type; laundering rows in no attempt make a ring ambiguous). | done |
| `splits.py` | `assign_groups` (union-find over shared edges merged with attempt id; strict variant by shared account), `stratified_group_cv` (k=5, fallback 3, repeats), `transfer_split`, `time_ordered_split`. The time-ordered scheme is **unusable** (HI-Small test side 0 candidates, LI-Small 1: rings recur, so almost every window mixes early and late groups) and is not used for evaluation; the code and tests are kept. | done |
| `metrics.py` | `average_precision` (PR-AUC, scikit-learn's step definition), `GroupedBootstrap` (resamples groups; the same resamples are reused across feature sets for paired differences), `ci`, `mean_ap_with_ci`, `paired_difference`. ARI, purity, RMSE, MAE, Spearman are not implemented (Phase 4). | done (PR-AUC and bootstrap) |
| `audit.py` | IBM AML pattern-behaviour audit of the edge scores (edge-level AUC, tail share, tie rate, imputation share); used by `experiments/ibm_pattern_audit.py`. | done |
| `recall.py` | Candidate-generation recall: `cycle_attempts`, `ibm_recall` (account-set and label-based recovery), `synthetic_recall`; used by `experiments/candidate_recall.py`. | done |
| `ablations.py` | Feature-set, filtration, complex, vectorization and sensitivity runners. | todo |

### 3.6 `models/`
| File | Role | Status |
|---|---|---|
| `gd.py` | Problem A: class-weighted L2 logistic regression by gradient descent (`logistic_loss_grad`, `smoothness_constant`, `gradient_descent`). Problem B: `project_simplex` and `AlphaLearner`, which learns `alpha` on the simplex with log-sum-exp smooth ring birth. | done |
| `classify.py` | scikit-learn logistic regression, random forest and RBF-SVM with fixed hyperparameters (`configs/classify.yaml`), feature-set registry (B, B+D, B+D+T, T, D, T_local, T_win, B+D+T_local, B+D+T_win), out-of-fold scoring, transfer scoring. | done |
| `cluster.py` | Ward hierarchical clustering (Euclidean) on standardized vectors; ARI, purity, silhouette, permutation and majority baselines, grouped bootstrap over positive groups; used by `experiments/ibm_cluster.py` (`configs/cluster.yaml`, results in `results/phase4_cluster/`). | done |
| `regress.py` | Next-window severity regression on SYNTHETIC rings: ring matching across consecutive windows by account set, ridge / random forest / gradient boosting, grouped folds by ring, grouped-bootstrap metrics; used by `experiments/synth_severity.py` (`configs/regress.yaml`, results in `results/phase4_severity/`). | done (synthetic only) |

### 3.7 `threshold/`
`ascent.py`: `reward`, `smoothed_reward_grad`, `best_tau`, `run_policies`. Runs online gradient ascent with label delay `d`, plus the fixed, periodically re-tuned (same lag) and hindsight-oracle comparators. `bandit_sim.py` (Phase 6, additive; `ascent.py` is unchanged): `DiscountedUCB`, `SlidingWindowTS` (Beta-Bernoulli on the normalised reward), `arm_grid`, `reward_matrix`, `run_bandit` (bandit feedback: only the played arm's reward is observed, usable `max(d, 1)` windows later; a full-information warm start on the first windows). Configuration `configs/bandit.yaml`; benchmark `experiments/bandit_report.py` on the optimization report's drift stream, results in `results/phase6_bandit/`. [done for ascent, todo for bandit]

### 3.8 `motif/` [done, optional, synthetic only]
Pump-and-dump track (v3 §3.6), **no topology**. `data/pump_dump.py` is its own generator (synthetic.py is untouched): one asset, a noisy price, planted pump-and-dump groups (accumulate in a band, exit together, price run-up and fall), organic momentum buyers and a news-spike hard negative, with `exit_spread_sd_h` as the hardness knob. `motif/pump_dump.py` finds coordinated-accumulation windows and computes M_ features (accumulation concentration, exit synchrony, position dispersion, run-up, drawdown, group size) and the baseline B_pd (group size, per-account volume); features read only the trade table and price series. `experiments/pump_dump_run.py` reuses the classification harness unchanged. The generator author decides separability.

### 3.9 Support
`viz.py` has `draw_digraph` and `plot_diagram` for notebooks. `experiments/gd_report.py` regenerates the optimization report numbers. `sanity_check.py` checks the TDA library.

Phase 3 and 6 experiment scripts (`experiments/`): `ibm_split_report.py` (counts, candidate pickles), `ibm_feature_build.py` (resumable B/D/T builder), `ibm_pattern_audit.py` (+ plot), `ibm_classify.py` (+ plot; `--append` adds ablation rows), `ibm_transfer.py`, `ibm_type_diagnostic.py`, `synth_classify.py` (+ plot), `plot_lift_per_scenario.py`, `candidate_recall.py` (+ plot), `bandit_report.py` (+ plot), and `make_numbers_pack.py`, which assembles `paper/numbers_pack.md` (every number read from a result file, provenance in `paper/numbers_pack/provenance.csv`) and `paper/figure_list.md`. Other new modules: `synthetic_table.py` (labelled B/D/T table of one synthetic window), `eval/audit.py`, `eval/recall.py`.

## 4. Key design decisions

| Decision | Reason |
|---|---|
| Directed flag complex, not undirected clique complex | An undirected complex fills every 3-cycle at once, so the canonical three-party ring has zero persistence. Undirected is kept as an ablation. |
| Localize by SCC plus cycle enumeration, not by homology | `pyflagser` returns no representative cycles. Persistence supplies features. |
| Ring birth belongs to feature set D, not T | It is a function of the ring's own edge weights. Counting it in T would overstate topology. |
| Primary test is B+D+T vs B+D | Isolates the topological contribution. A null result is reportable. |
| Candidates built once on a loose equal-weight cutoff | The candidate set must not depend on the `alpha` being learned. |
| Temporal split with a gap | Overlapping windows leak rings across random splits. |
| Persistence on candidate neighbourhoods, dimension 2 | Keeps compute bounded. |
| Two backends that must agree | `flagser` is fast; `reference` is auditable. |

## 5. Verified behaviours (engineering gates)

210 tests pass (`pytest`, a few seconds). They cover:
- cyclic triangle gives one infinite H1 bar; transitive triangle gives none; shortcut-filled triangle dies at the fill weight; reciprocal pair gives an H1 class; `flagser` and `reference` agree (`test_directed_triangles.py`);
- score and weight formulas and edge cases (`test_weights.py`);
- candidate enumeration and dedup (`test_candidates.py`);
- planted wash-ring edges land in the lowest-weight decile and every planted ring is a candidate (`test_planted_rings.py`);
- generator scenarios (`test_synthetic_scenarios.py`);
- ring-labeling rule (`test_labels.py`);
- T features and the T table (`test_vectorize.py`, `test_t_table.py`), including the exact ring-survival test;
- IBM loader, window iterator and ring labels on toy fixtures (`test_ibm_aml.py`); groups, grouped CV, time-ordered and transfer splits (`test_splits.py`);
- gradient correctness, simplex projection, GD vs SciPy (`test_gd.py`);
- B and D features and the leakage guard (`test_baseline_features.py`, `test_direction_features.py`, `test_feature_leakage.py`); PR-AUC and grouped bootstrap (`test_metrics.py`); classifiers, group disjointness, transfer (`test_classify.py`); audit (`test_audit.py`); synthetic table and recall (`test_synthetic_table.py`);
- pump-and-dump generator, candidates, features and leakage guard (`test_pump_dump.py`);
- severity target alignment, leakage guard and metrics (`test_regress.py`);
- ring clustering metrics, baselines and bootstrap (`test_cluster.py`);
- bandit policies: convergence to a known best arm, no reward used before its lag, sliding-window forgetting, determinism (`test_bandit_sim.py`).

## 6. Results provenance

`results/report/` plus `results/gd_metrics.json` and `results/figures/` hold the numbers quoted in `report/Optimization_Report`. They come from `python experiments/gd_report.py --config report`, which pins the generator to dense, chorded, shortcut-filled and hot-wallet counts of 0 and treats ambiguous candidates as negatives. Never overwrite them. The script refuses without `--force`. New experiments go to `--config full` (`results/full/<ambiguous>/`) or a new folder. Any change to generator defaults, labels, candidates or weights must keep the `report` config reproducing the same numbers; check by re-running into a scratch copy and diffing.

Headline report numbers (synthetic only): learned weights raise test average precision of the ring-birth detector from 0.568 to 0.9997 and drop reciprocity; a curvature-matched step converges in 322 iterations; online gradient ascent keeps 97% of oracle reward vs 24% for a fixed threshold.

## 7. Gaps between design and code

| Item | Design (v3) | Code now |
|---|---|---|
| Max cycle length | 6 | 6 in `t_feature_table` and `configs/topology.yaml`; 5 in `window_candidates` (report path, frozen) |
| Infinite-bar cap | window max filtration value | window max in the T table (configurable); fixed 1.0 in the legacy `topo_*` columns (report only) |
| Sparsification | swept `tau_s` | quantile, default 0.25, a parameter of `t_feature_table` and the config; candidate-recall sweep done (`experiments/candidate_recall.py`); classification run at IBM tau 0.05 only |
| Landscapes, window features | primary vectorization | done (`vectorize.py`) |
| B features | required | done (`baseline_features.py`; sampled betweenness on large windows) |
| Dedup rule | lowest birth | shortest first, then birth (documented) |
| Dedup overlap | 80% edge overlap | removes nothing for `max_len` <= 9; first removals at 10 (verified on dense windows, tests in `test_candidates.py`), so near-duplicate rings all stay; Phase 3 CV must group by ring identity |
| Ring-specific infinite bar | n/a | `T_local_ring_inf`, exact Z/2 test (`ring_survives`); birth-match proxy only as an oversize fallback |
| Threshold method | discounted UCB and Thompson | online gradient ascent (report) and, in Phase 6, discounted UCB and sliding-window Thompson (`bandit_sim.py`) |
| Repo layout | `configs/` YAML, per-phase notebooks | `configs/topology.yaml` only; one notebook |

## 8. Remaining build order

1. (done) D and B feature code, the combined B / D / T table and the resumable IBM builder for HI-Small and LI-Small at tau 0.05.
2. (done) IBM AML pattern-behaviour audit of `s_rec`, `s_bal`, `s_reg` per pattern type (no AML-specific weight variant was built).
3. (done) Classification harness (`eval/metrics.py`, `models/classify.py`) on HI-Small, LI-Small, HI<->LI transfer and synthetic per-scenario runs, with the T_local / T_win ablations. Evidence is thin by construction (10-16 positive groups): group counts and prevalence are reported, nothing is tuned to narrow CIs.
4. (not done) Sensitivity sweeps over `tau_quantile` for classification, `max_len`, infinite-bar cap, label threshold X, window length (only candidate recall is swept).
5. (done) Results numbers pack (`paper/numbers_pack.md`). Remaining: write-up (`paper/paper_content.md` is the content pack).

Phase 6 (bandit policies and drift benchmark) is done (`threshold/bandit_sim.py`). Phase 4 clustering (`models/cluster.py`) and synthetic-only severity regression (`models/regress.py`) are done as exploratory runs; infinite-bar cap sensitivity is done (`experiments/cap_sensitivity.py`, `configs/cap_sensitivity.yaml`). Phase 5 (NFT) is cut: running Dune queries is blocked on the free tier. Future work (not planned for this iteration): Phase 5 NFT loader and label-provenance audit, the pump-and-dump track, and the undirected-complex ablation.
