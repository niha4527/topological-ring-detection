# Numbers pack

Every number below was read by `experiments/make_numbers_pack.py` from the result file named under each table; nothing is typed by hand. Provenance of every table cell: `paper/numbers_pack/provenance.csv`. CSV copies of the sources: `paper/numbers_pack/csv/`. Figures: `paper/numbers_pack/figures/`. Nothing here is a pass/fail result; PR-AUC values carry 95% grouped-bootstrap CIs (resampling groups).


## 1. Setup

- Tests collected (`pytest --collect-only -q`): 189.

| dataset | transactions (self-loops dropped) | windows | window length / stride | tau_quantile | max_len | windows with candidates | candidates | negatives | positives (primary) | ambiguous | groups | positive groups |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HI | 4,487,133 | 13 | 5 days / 1 day | 0.05 | 6 | 13 | 8074 | 7424 | 131 | 519 | 2523 | 14 |
| LI | 6,119,572 | 12 | 5 days / 1 day | 0.05 | 6 | 11 | 10206 | 9682 | 38 | 486 | 3378 | 10 |

*Source:* `results/phase3_features/ibm/build.log`, `results/phase3_features/ibm/HI_tau0.05/window_*.json`, `results/phase3_features/ibm/LI_tau0.05/window_*.json`, `results/phase3_features/ibm_split_counts.csv`. Transactions from the build log; windows, tau and max_len from the per-window JSON written by the builder; counts from the split report (groups = union-find over shared edges merged with attempt id).

Classification settings (stored with the HI results, identical in the LI and transfer configs unless listed below):

| item | value |
|---|---|
| logreg | L2, C=1.0, max_iter=2000, class_weight balanced |
| random forest | n_estimators=300, n_jobs=2, class_weight balanced_subsample |
| svm | RBF, C=1.0, gamma=scale, class_weight balanced |
| CV | stratified group k-fold, k=5 (fallback 3), repeats=5, cv seed=0 |
| bootstrap | grouped, n_boot=1000, seed=12345, 95% percentile CI |
| model seed | 0 |
| runtime HI (s) | 714 (first run), 296 (ablation rows appended) |
| runtime LI (s) | 1093 |
| runtime transfer (s) | 36 |

*Source:* `results/phase3_features/classify/HI_tau0.05/config_used.yaml`, `results/phase3_features/classify/HI_tau0.05/config_used_ablation.yaml`, `results/phase3_features/classify/LI_tau0.05/config_used.yaml`, `results/phase3_features/classify/transfer/config_used.yaml`

Ablation rows (`B+D+T_local`, `B+D+T_win`) and their paired rows were appended to the HI results with the same settings (config `results/phase3_features/classify/HI_tau0.05/config_used_ablation.yaml`). The paired list stored with the config: [['B+D+T', 'B+D'], ['B+D', 'B'], ['B+D+T_local', 'B+D'], ['B+D+T_win', 'B+D'], ['B+D+T', 'B+D+T_local']].

Label rule X (§3.7): 0.8 (`configs/ibm.yaml`, current file; **not stored with the results**, flagged). Split config in the same file: k=5, fallback 3, repeats 5.

Synthetic runs: seeds [0, 1, 2, 3, 4, 5] pooled per condition, conditions hard_negatives x dense_internal_chords, CV repeats 5, bootstrap n_boot 1000, same fixed hyperparameters (stored in `results/phase3_features/classify/synthetic/hn-off_chords-default/config_used.yaml`). Synthetic generator parameters and candidate tau (0.25) are **not stored with the results** (flagged); the synthetic recall CSV stores max_len = 6.

Candidate step defaults for the synthetic table (`configs/topology.yaml`, current file): tau_quantile 0.25, max_len 6.

Betweenness on window 0 (days 0-5), sampled with k=100 sources (seed 0, exact only up to 3000 nodes):

| dataset | nodes | nodes with betweenness > 0 | share non-zero | exact |
|---|---|---|---|---|
| HI | 404,273 | 64,441 | 0.1594 | False |
| LI | 551,969 | 17,673 | 0.0320 | False |

*Source:* `results/phase3_features/ibm/betweenness_cache/HI_w00.pkl`, `results/phase3_features/ibm/betweenness_cache/LI_w00.pkl`


## 2. Optimization report headline numbers

Copied unchanged (values as stored, full precision; lists as JSON). Generator pinned to the report configuration (dense / chorded / shortcut-filled / hot-wallet counts = 0).

| key | value |
|---|---|
| data.train_jitter | [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6] |
| data.test_jitter | [0.1, 0.19, 0.28, 0.37, 0.46, 0.55] |
| data.train_windows | 12 |
| data.test_windows | 6 |
| data.train_candidates | 2235 |
| data.test_candidates | 1370 |
| data.train_by_kind.background | 1960 |
| data.train_by_kind.market_maker | 120 |
| data.train_by_kind.wash_ring | 108 |
| data.train_by_kind.arbitrage_loop | 47 |
| data.test_by_kind.background | 1234 |
| data.test_by_kind.market_maker | 60 |
| data.test_by_kind.wash_ring | 54 |
| data.test_by_kind.arbitrage_loop | 22 |
| problem_A.lambda | 0.01 |
| problem_A.L | 6.887705869715905 |
| problem_A.eta_1_over_L | 0.14518622294788072 |
| problem_A.L_local_at_optimum | 0.2205079575528847 |
| problem_A.J_star_lbfgs | 0.042958848803096736 |
| problem_A.final_gap.0.2/L | 0.007066348826610017 |
| problem_A.final_gap.1/L | 0.00032581514866465017 |
| problem_A.final_gap.1.9/L | 3.0540141188979686e-05 |
| problem_A.final_gap.1/L_loc | -1.3780643293159756e-14 |
| problem_A.final_gap.2.2/L_loc | 0.007405561651085031 |
| problem_A.iters_to_1e-6.0.2/L | null |
| problem_A.iters_to_1e-6.1/L | null |
| problem_A.iters_to_1e-6.1.9/L | null |
| problem_A.iters_to_1e-6.1/L_loc | 322 |
| problem_A.iters_to_1e-6.2.2/L_loc | null |
| problem_A.max_abs_theta_diff_vs_lbfgs | 6.505325776950954e-07 |
| problem_A.test_AP.D (behavioural) | 0.9984306340238543 |
| problem_A.test_AP.D + T (behavioural + topological) | 0.9968660968660967 |
| problem_A.coefficients_D+T.min_reg | 1.084 |
| problem_A.coefficients_D+T.topo_nbhd_edges | -0.802 |
| problem_A.coefficients_D+T.mean_reg | 0.559 |
| problem_A.coefficients_D+T.mean_rec | -0.415 |
| problem_A.coefficients_D+T.topo_total_pers | -0.372 |
| problem_A.coefficients_D+T.birth_eq | -0.368 |
| problem_A.coefficients_D+T.topo_n_infinite | -0.364 |
| problem_A.coefficients_D+T.mean_bal | 0.341 |
| problem_A.coefficients_D+T.topo_max_pers | -0.332 |
| problem_A.coefficients_D+T.min_bal | 0.306 |
| problem_A.coefficients_D+T.topo_h1_count | -0.306 |
| problem_A.coefficients_D+T.min_rec | -0.28 |
| problem_A.coefficients_D+T.length | 0.157 |
| problem_B.alpha_learned | [0.0, 0.2067, 0.7933] |
| problem_B.c | 5.082687275467421 |
| problem_B.k | 9.818131894382558 |
| problem_B.loss_start | 0.882672265454568 |
| problem_B.loss_end | 0.0794075929695794 |
| problem_B.restarts.equal.alpha | [0.0, 0.2067, 0.7933] |
| problem_B.restarts.equal.loss | 0.0794075929695794 |
| problem_B.restarts.rec vertex.alpha | [0.0, 0.2053, 0.7947] |
| problem_B.restarts.rec vertex.loss | 0.07948467189068216 |
| problem_B.restarts.bal vertex.alpha | [0.0, 0.2073, 0.7927] |
| problem_B.restarts.bal vertex.loss | 0.07937311467627156 |
| problem_B.restarts.reg vertex.alpha | [0.0, 0.2061, 0.7939] |
| problem_B.restarts.reg vertex.loss | 0.07944357842400052 |
| problem_B.grad_check_max_abs_error | 9.317147053877761e-11 |
| problem_B.grad_alpha_at_solution | [0.08476654878972688, -0.05110694691236644, -0.05045746186466852] |
| problem_B.test_AP_ring_birth_equal_alpha | 0.5675129878917745 |
| problem_B.test_AP_ring_birth_learned_alpha | 0.9996632996632996 |
| problem_C.windows | 40 |
| problem_C.lambda | 1.0 |
| problem_C.delay | 2 |
| problem_C.cumulative_reward_after_warmup.fixed | 71.0 |
| problem_C.cumulative_reward_after_warmup.retune | 258.0 |
| problem_C.cumulative_reward_after_warmup.ascent | 291.0 |
| problem_C.cumulative_reward_after_warmup.oracle | 300.0 |
| problem_C.alpha_pre_drift | [0.0, 0.0, 1.0] |
| problem_C.share_of_oracle.fixed | 0.23666666666666666 |
| problem_C.share_of_oracle.retune | 0.86 |
| problem_C.share_of_oracle.ascent | 0.97 |
| problem_C.share_of_oracle.oracle | 1.0 |
| problem_C.robust_detector_cumulative_reward.fixed | 292.0 |
| problem_C.robust_detector_cumulative_reward.retune | 287.0 |
| problem_C.robust_detector_cumulative_reward.ascent | 291.0 |
| problem_C.robust_detector_cumulative_reward.oracle | 306.0 |
| problem_C.robust_detector_share_of_oracle.fixed | 0.954248366013072 |
| problem_C.robust_detector_share_of_oracle.retune | 0.9379084967320261 |
| problem_C.robust_detector_share_of_oracle.ascent | 0.9509803921568627 |
| problem_C.robust_detector_share_of_oracle.oracle | 1.0 |
| problem_C.drift | "wash timing jitter 0.05 -> 0.60 and amount noise 0.01 -> 0.16 over 40 windows" |
| problem_C.candidate_recall_first5_last5 | [1.0, 1.0] |

*Source:* `results/gd_metrics.json`. Identical original at `results/report/gd_metrics.json`; not modified.


## 3. IBM HI-Small


### 3.1 HI-Small: label counts by variant and type

| variant | candidates | positives | prevalence | positive groups | fewest positive groups on a train or test side |
|---|---|---|---|---|---|
| primary | 7555 | 131 | 0.0173 | 14 | not recorded |
| ambiguous_positive | 8074 | 650 | 0.0805 | 142 | not recorded |
| cycle_only | 7460 | 36 | 0.0048 | 12 | not recorded |
| strict_groups | 7555 | 131 | 0.0173 | 14 | not recorded |

| candidates | negatives | ambiguous | positives | pos CYCLE | pos GATHER-SCATTER | pos SCATTER-GATHER | pos STACK | cycle positives | cycle positive groups |
|---|---|---|---|---|---|---|---|---|---|
| 8074 | 7424 | 519 | 131 | 36 | 44 | 18 | 33 | 36 | 12 |

*Source:* `results/phase3_features/classify/HI_tau0.05/results.csv`, `results/phase3_features/ibm_split_counts.csv`


### 3.2 HI-Small: primary PR-AUC [95% CI], ambiguous excluded

Prevalence 0.0173, positives 131, positive groups 14, k per repeat 5,5,5,5,5, folds without positive 0. Pooled out-of-fold, mean over 5 repeats.

| feature set | logreg | rf | svm |
|---|---|---|---|
| B | 0.077 [0.033, 0.167] | 0.080 [0.033, 0.157] | 0.094 [0.040, 0.187] |
| B+D | 0.477 [0.240, 0.657] | 0.652 [0.328, 0.830] | 0.326 [0.126, 0.536] |
| B+D+T | 0.581 [0.300, 0.733] | 0.676 [0.351, 0.868] | 0.329 [0.135, 0.509] |
| B+D+T_local | 0.618 [0.385, 0.747] | 0.692 [0.355, 0.881] | 0.363 [0.173, 0.543] |
| B+D+T_win | 0.511 [0.255, 0.700] | 0.659 [0.349, 0.846] | 0.414 [0.174, 0.618] |
| T | 0.552 [0.333, 0.696] | 0.400 [0.181, 0.662] | 0.295 [0.126, 0.483] |
| D | 0.581 [0.301, 0.774] | 0.682 [0.363, 0.867] | 0.131 [0.038, 0.256] |
| T_local | 0.512 [0.276, 0.658] | 0.376 [0.177, 0.597] | 0.209 [0.074, 0.348] |
| T_win | 0.121 [0.056, 0.198] | 0.130 [0.061, 0.202] | 0.030 [0.015, 0.051] |

`-D_birth` alone (single-feature detector, no model, scores within this dataset): 0.314 [0.097, 0.487]

Mean of per-window PR-AUC over the 13 windows with at least one positive:

| feature set | logreg | rf | svm |
|---|---|---|---|
| B | 0.322 | 0.311 | 0.345 |
| B+D | 0.644 | 0.737 | 0.560 |
| B+D+T | 0.695 | 0.760 | 0.558 |
| B+D+T_local | 0.752 | 0.771 | 0.559 |
| B+D+T_win | 0.637 | 0.732 | 0.598 |
| T | 0.651 | 0.553 | 0.549 |
| D | 0.703 | 0.764 | 0.393 |
| T_local | 0.669 | 0.591 | 0.547 |
| T_win | 0.257 | 0.257 | 0.261 |

*Source:* `results/phase3_features/classify/HI_tau0.05/results.csv`


### 3.3 HI-Small: paired differences (primary), same resamples

**Pre-specified primary comparison: B+D+T minus B+D.** Second: B+D minus B. The remaining comparisons are **exploratory** (T_local / T_win ablations, added after the first run).

Pre-specified primary:

| comparison | logreg | rf | svm | n_boot |
|---|---|---|---|---|
| B+D+T minus B+D | +0.104 [-0.002, +0.171] | +0.025 [-0.050, +0.102] | +0.003 [-0.111, +0.077] | 1000 |

Second:

| comparison | logreg | rf | svm | n_boot |
|---|---|---|---|---|
| B+D minus B | +0.399 [+0.139, +0.571] | +0.572 [+0.276, +0.720] | +0.233 [+0.057, +0.379] | 1000 |

Exploratory:

| comparison | logreg | rf | svm | n_boot |
|---|---|---|---|---|
| B+D+T_local minus B+D | +0.141 [+0.031, +0.224] | +0.041 [-0.037, +0.106] | +0.037 [-0.081, +0.126] | 1000 |
| B+D+T_win minus B+D | +0.034 [-0.005, +0.094] | +0.007 [-0.025, +0.059] | +0.088 [+0.006, +0.176] | 1000 |
| B+D+T minus B+D+T_local | -0.037 [-0.099, -0.001] | -0.016 [-0.037, +0.010] | -0.035 [-0.076, -0.002] | 1000 |

**Exploratory comparisons made on HI: 9 in the primary variant (3 comparisons x 3 models), 36 over all variants in this file. No multiplicity correction is applied.**

*Source:* `results/phase3_features/classify/HI_tau0.05/paired_differences.csv`


### 3.4 HI-Small: other variants

**ambiguous_positive**: prevalence 0.0805, positives 650, positive groups 142.

| feature set | logreg | rf | svm |
|---|---|---|---|
| B | 0.168 [0.090, 0.261] | 0.178 [0.083, 0.303] | 0.212 [0.097, 0.331] |
| B+D | 0.446 [0.155, 0.678] | 0.486 [0.184, 0.697] | 0.231 [0.087, 0.400] |
| B+D+T | 0.473 [0.172, 0.674] | 0.490 [0.147, 0.720] | 0.408 [0.094, 0.656] |
| B+D+T_local | 0.458 [0.168, 0.651] | 0.507 [0.159, 0.729] | 0.409 [0.110, 0.654] |
| B+D+T_win | 0.454 [0.154, 0.688] | 0.497 [0.181, 0.711] | 0.431 [0.117, 0.673] |
| T | 0.394 [0.135, 0.583] | 0.218 [0.093, 0.368] | 0.352 [0.083, 0.588] |

| paired difference (exploratory rows last) | logreg | rf | svm |
|---|---|---|---|
| B+D+T minus B+D | +0.027 [-0.015, +0.086] | +0.003 [-0.056, +0.032] | +0.176 [-0.033, +0.297] |
| B+D minus B | +0.277 [+0.033, +0.432] | +0.308 [+0.090, +0.407] | +0.019 [-0.035, +0.082] |
| B+D+T_local minus B+D | +0.012 [-0.037, +0.096] | +0.020 [-0.039, +0.044] | +0.177 [-0.005, +0.294] |
| B+D+T_win minus B+D | +0.008 [-0.011, +0.021] | +0.011 [-0.018, +0.023] | +0.200 [+0.012, +0.309] |
| B+D+T minus B+D+T_local | +0.015 [-0.014, +0.032] | -0.017 [-0.024, -0.007] | -0.001 [-0.042, +0.015] |

**cycle_only**: prevalence 0.0048, positives 36, positive groups 12.

| feature set | logreg | rf | svm |
|---|---|---|---|
| B | 0.104 [0.011, 0.323] | 0.036 [0.009, 0.078] | 0.121 [0.008, 0.359] |
| B+D | 0.252 [0.072, 0.509] | 0.295 [0.091, 0.573] | 0.041 [0.017, 0.080] |
| B+D+T | 0.190 [0.042, 0.407] | 0.301 [0.082, 0.612] | 0.029 [0.014, 0.052] |
| B+D+T_local | 0.211 [0.049, 0.433] | 0.335 [0.097, 0.661] | 0.044 [0.020, 0.085] |
| B+D+T_win | 0.205 [0.059, 0.431] | 0.252 [0.072, 0.518] | 0.036 [0.017, 0.077] |
| T | 0.225 [0.046, 0.434] | 0.209 [0.052, 0.462] | 0.016 [0.006, 0.045] |

| paired difference (exploratory rows last) | logreg | rf | svm |
|---|---|---|---|
| B+D+T minus B+D | -0.062 [-0.196, +0.027] | +0.007 [-0.056, +0.074] | -0.012 [-0.044, +0.010] |
| B+D minus B | +0.148 [-0.127, +0.403] | +0.259 [+0.063, +0.519] | -0.080 [-0.322, +0.044] |
| B+D+T_local minus B+D | -0.040 [-0.182, +0.050] | +0.040 [-0.032, +0.115] | +0.003 [-0.020, +0.035] |
| B+D+T_win minus B+D | -0.047 [-0.106, +0.011] | -0.042 [-0.098, +0.013] | -0.005 [-0.032, +0.014] |
| B+D+T minus B+D+T_local | -0.021 [-0.050, +0.001] | -0.034 [-0.074, +0.000] | -0.015 [-0.038, -0.000] |

**strict_groups**: prevalence 0.0173, positives 131, positive groups 14.

| feature set | logreg | rf | svm |
|---|---|---|---|
| B | 0.082 [0.034, 0.173] | 0.076 [0.030, 0.156] | 0.111 [0.051, 0.211] |
| B+D | 0.478 [0.230, 0.672] | 0.671 [0.372, 0.846] | 0.310 [0.117, 0.501] |
| B+D+T | 0.600 [0.337, 0.749] | 0.688 [0.393, 0.871] | 0.322 [0.134, 0.516] |
| B+D+T_local | 0.607 [0.365, 0.748] | 0.702 [0.397, 0.883] | 0.347 [0.168, 0.538] |
| B+D+T_win | 0.504 [0.246, 0.693] | 0.678 [0.388, 0.853] | 0.407 [0.163, 0.609] |
| T | 0.550 [0.323, 0.696] | 0.405 [0.189, 0.644] | 0.294 [0.134, 0.465] |

| paired difference (exploratory rows last) | logreg | rf | svm |
|---|---|---|---|
| B+D+T minus B+D | +0.122 [-0.013, +0.218] | +0.016 [-0.059, +0.085] | +0.013 [-0.106, +0.097] |
| B+D minus B | +0.396 [+0.143, +0.569] | +0.596 [+0.318, +0.731] | +0.198 [+0.021, +0.346] |
| B+D+T_local minus B+D | +0.129 [+0.015, +0.219] | +0.031 [-0.045, +0.090] | +0.037 [-0.084, +0.135] |
| B+D+T_win minus B+D | +0.026 [-0.016, +0.083] | +0.006 [-0.025, +0.049] | +0.097 [+0.009, +0.177] |
| B+D+T minus B+D+T_local | -0.007 [-0.041, +0.012] | -0.014 [-0.034, +0.008] | -0.024 [-0.072, +0.013] |

*Source:* `results/phase3_features/classify/HI_tau0.05/results.csv`, `results/phase3_features/classify/HI_tau0.05/paired_differences.csv`


### 3.5 HI-Small pattern-behaviour audit (pooled over the audited windows)

Audited windows: w0 (days 0-5), w2 (days 2-7) and pooled. **Tail share** = share of a group's pairs in the most suspicious 10% tail of the window (for `weight`: the lowest 10% of weights); the all-pairs baseline is 0.100 for `weight`, `s_bal`, `s_reg` and is the `baseline` column. AUC = edge-level AUC of the group against background pairs (0.5 = chance, higher = the score ranks the group as more suspicious). `s_rec` is almost always 0 (ties), so its tail share and baseline are tie-driven.

| group | pairs | AUC weight | AUC s_bal | AUC s_rec | AUC s_reg | tail share weight | tail share s_bal |
|---|---|---|---|---|---|---|---|
| BIPARTITE | 238 | 0.562 | 0.564 | 0.515 | 0.557 | 0.038 | 0.176 |
| CYCLE | 208 | 0.794 | 0.795 | 0.584 | 0.601 | 0.173 | 0.433 |
| FAN-IN | 234 | 0.583 | 0.541 | 0.536 | 0.603 | 0.081 | 0.103 |
| FAN-OUT | 263 | 0.528 | 0.513 | 0.526 | 0.574 | 0.057 | 0.106 |
| GATHER-SCATTER | 408 | 0.617 | 0.591 | 0.543 | 0.595 | 0.091 | 0.125 |
| RANDOM | 122 | 0.785 | 0.791 | 0.551 | 0.614 | 0.107 | 0.393 |
| SCATTER-GATHER | 470 | 0.619 | 0.593 | 0.517 | 0.593 | 0.038 | 0.132 |
| STACK | 365 | 0.653 | 0.635 | 0.525 | 0.597 | 0.055 | 0.255 |
| unattributed | 1915 | 0.476 | 0.460 | 0.523 | 0.524 | 0.065 | 0.069 |
| background (baseline) | 873803 | - | - | - | - | 0.100 | 0.100 |

*Source:* `results/phase3_features/audit/HI_audit_summary.csv`

Share of pairs with fewer than 3 transactions, whose timing-regularity score is imputed (`gap_cv` fill):

| group | pairs | share imputed |
|---|---|---|
| BIPARTITE | 238 | 0.966 |
| CYCLE | 208 | 0.899 |
| FAN-IN | 234 | 0.923 |
| FAN-OUT | 263 | 0.947 |
| GATHER-SCATTER | 408 | 0.939 |
| RANDOM | 122 | 0.893 |
| SCATTER-GATHER | 470 | 0.964 |
| STACK | 365 | 0.956 |
| unattributed | 1915 | 0.685 |
| background | 873803 | 0.467 |

*Source:* `results/phase3_features/audit/HI_audit_imputation.csv`


## 4. IBM LI-Small and HI<->LI transfer


### 4.1 LI-Small: label counts by variant and type

| variant | candidates | positives | prevalence | positive groups | fewest positive groups on a train or test side |
|---|---|---|---|---|---|
| primary | 9720 | 38 | 0.0039 | 10 | 1 |
| ambiguous_positive | 10206 | 524 | 0.0513 | 189 | 33 |
| cycle_only | unusable (see note) or not run |  |  |  |  |
| strict_groups | unusable (see note) or not run |  |  |  |  |

**cycle_only is unusable on LI**: 3 positives in 3 groups; fewest positive groups on a train or test side = 1 (< 2). Source: `results/phase3_features/classify/LI_tau0.05/config_used.yaml`.

| candidates | negatives | ambiguous | positives | pos CYCLE | pos GATHER-SCATTER | pos SCATTER-GATHER | pos STACK | cycle positives | cycle positive groups |
|---|---|---|---|---|---|---|---|---|---|
| 10206 | 9682 | 486 | 38 | 3 | 1 | 17 | 17 | 3 | 3 |

*Source:* `results/phase3_features/classify/LI_tau0.05/results.csv`, `results/phase3_features/ibm_split_counts.csv`


### 4.2 LI-Small: primary PR-AUC [95% CI], ambiguous excluded

Prevalence 0.0039, positives 38, positive groups 10, k per repeat 5,5,5,5,5, folds without positive 0. Pooled out-of-fold, mean over 5 repeats.

| feature set | logreg | rf | svm |
|---|---|---|---|
| B | 0.020 [0.007, 0.045] | 0.010 [0.002, 0.031] | 0.042 [0.008, 0.167] |
| B+D | 0.093 [0.030, 0.214] | 0.142 [0.049, 0.306] | 0.148 [0.030, 0.321] |
| B+D+T | 0.126 [0.032, 0.268] | 0.130 [0.042, 0.259] | 0.103 [0.036, 0.227] |
| B+D+T_local | 0.094 [0.030, 0.216] | 0.137 [0.046, 0.285] | 0.127 [0.040, 0.287] |
| B+D+T_win | 0.120 [0.029, 0.259] | 0.206 [0.069, 0.400] | 0.113 [0.037, 0.246] |
| T | 0.021 [0.002, 0.064] | 0.005 [0.002, 0.015] | 0.023 [0.002, 0.152] |
| D | 0.091 [0.028, 0.221] | 0.077 [0.026, 0.169] | 0.040 [0.014, 0.082] |
| T_local | 0.009 [0.002, 0.035] | 0.004 [0.001, 0.007] | 0.012 [0.002, 0.049] |
| T_win | 0.006 [0.002, 0.015] | 0.004 [0.002, 0.009] | 0.006 [0.002, 0.014] |

`-D_birth` alone (single-feature detector, no model, scores within this dataset): 0.006 [0.002, 0.025]

Mean of per-window PR-AUC over the 11 windows with at least one positive:

| feature set | logreg | rf | svm |
|---|---|---|---|
| B | 0.169 | 0.114 | 0.173 |
| B+D | 0.278 | 0.336 | 0.376 |
| B+D+T | 0.267 | 0.322 | 0.291 |
| B+D+T_local | 0.283 | 0.339 | 0.378 |
| B+D+T_win | 0.262 | 0.378 | 0.296 |
| T | 0.119 | 0.155 | 0.187 |
| D | 0.272 | 0.217 | 0.177 |
| T_local | 0.119 | 0.097 | 0.118 |
| T_win | 0.096 | 0.096 | 0.101 |

*Source:* `results/phase3_features/classify/LI_tau0.05/results.csv`


### 4.3 LI-Small: paired differences (primary), same resamples

**Pre-specified primary comparison: B+D+T minus B+D.** Second: B+D minus B. The remaining comparisons are **exploratory** (T_local / T_win ablations, added after the first run).

Pre-specified primary:

| comparison | logreg | rf | svm | n_boot |
|---|---|---|---|---|
| B+D+T minus B+D | +0.033 [-0.007, +0.097] | -0.012 [-0.119, +0.077] | -0.044 [-0.181, +0.091] | 1000 |

Second:

| comparison | logreg | rf | svm | n_boot |
|---|---|---|---|---|
| B+D minus B | +0.072 [+0.020, +0.173] | +0.132 [+0.045, +0.283] | +0.105 [-0.060, +0.276] | 1000 |

Exploratory:

| comparison | logreg | rf | svm | n_boot |
|---|---|---|---|---|
| B+D+T_local minus B+D | +0.002 [-0.033, +0.042] | -0.005 [-0.082, +0.066] | -0.020 [-0.127, +0.094] | 1000 |
| B+D+T_win minus B+D | +0.027 [-0.008, +0.078] | +0.065 [-0.019, +0.173] | -0.035 [-0.136, +0.056] | 1000 |
| B+D+T minus B+D+T_local | +0.032 [-0.002, +0.085] | -0.007 [-0.053, +0.029] | -0.024 [-0.111, +0.043] | 1000 |

**Exploratory comparisons made on LI: 9 in the primary variant (3 comparisons x 3 models), 18 over all variants in this file. No multiplicity correction is applied.**

*Source:* `results/phase3_features/classify/LI_tau0.05/paired_differences.csv`


### 4.4 LI-Small: other variants

**ambiguous_positive**: prevalence 0.0513, positives 524, positive groups 189.

| feature set | logreg | rf | svm |
|---|---|---|---|
| B | 0.077 [0.061, 0.097] | 0.061 [0.050, 0.079] | 0.076 [0.061, 0.094] |
| B+D | 0.091 [0.068, 0.124] | 0.064 [0.052, 0.078] | 0.081 [0.064, 0.107] |
| B+D+T | 0.090 [0.068, 0.120] | 0.067 [0.054, 0.082] | 0.088 [0.068, 0.120] |
| B+D+T_local | 0.090 [0.068, 0.123] | 0.066 [0.054, 0.081] | 0.091 [0.070, 0.120] |
| B+D+T_win | 0.089 [0.068, 0.118] | 0.066 [0.054, 0.081] | 0.085 [0.066, 0.113] |
| T | 0.052 [0.043, 0.063] | 0.054 [0.046, 0.065] | 0.053 [0.043, 0.066] |

| paired difference (exploratory rows last) | logreg | rf | svm |
|---|---|---|---|
| B+D+T minus B+D | -0.001 [-0.014, +0.008] | +0.003 [-0.003, +0.008] | +0.007 [-0.009, +0.028] |
| B+D minus B | +0.014 [-0.003, +0.036] | +0.002 [-0.012, +0.011] | +0.005 [-0.010, +0.025] |
| B+D+T_local minus B+D | -0.001 [-0.008, +0.006] | +0.002 [-0.002, +0.007] | +0.009 [-0.003, +0.026] |
| B+D+T_win minus B+D | -0.002 [-0.013, +0.003] | +0.002 [-0.001, +0.006] | +0.004 [-0.007, +0.018] |
| B+D+T minus B+D+T_local | -0.000 [-0.009, +0.005] | +0.000 [-0.002, +0.003] | -0.002 [-0.013, +0.008] |

cycle_only: not run or unusable.

strict_groups: not run or unusable.

*Source:* `results/phase3_features/classify/LI_tau0.05/results.csv`, `results/phase3_features/classify/LI_tau0.05/paired_differences.csv`


### 4.5 LI-Small pattern-behaviour audit (pooled over the audited windows)

Audited windows: w0 (days 0-5), w2 (days 2-7) and pooled. **Tail share** = share of a group's pairs in the most suspicious 10% tail of the window (for `weight`: the lowest 10% of weights); the all-pairs baseline is 0.100 for `weight`, `s_bal`, `s_reg` and is the `baseline` column. AUC = edge-level AUC of the group against background pairs (0.5 = chance, higher = the score ranks the group as more suspicious). `s_rec` is almost always 0 (ties), so its tail share and baseline are tie-driven.

| group | pairs | AUC weight | AUC s_bal | AUC s_rec | AUC s_reg | tail share weight | tail share s_bal |
|---|---|---|---|---|---|---|---|
| BIPARTITE | 94 | 0.615 | 0.589 | 0.498 | 0.596 | 0.000 | 0.181 |
| CYCLE | 45 | 0.771 | 0.793 | 0.498 | 0.599 | 0.000 | 0.444 |
| FAN-IN | 84 | 0.489 | 0.503 | 0.498 | 0.525 | 0.000 | 0.107 |
| FAN-OUT | 145 | 0.522 | 0.501 | 0.501 | 0.569 | 0.007 | 0.083 |
| GATHER-SCATTER | 90 | 0.634 | 0.615 | 0.515 | 0.584 | 0.033 | 0.267 |
| RANDOM | 63 | 0.746 | 0.778 | 0.498 | 0.542 | 0.000 | 0.381 |
| SCATTER-GATHER | 146 | 0.708 | 0.705 | 0.529 | 0.587 | 0.062 | 0.260 |
| STACK | 212 | 0.647 | 0.635 | 0.510 | 0.580 | 0.024 | 0.222 |
| unattributed | 2408 | 0.484 | 0.466 | 0.526 | 0.526 | 0.075 | 0.073 |
| background (baseline) | 1193760 | - | - | - | - | 0.100 | 0.100 |

*Source:* `results/phase3_features/audit/LI_audit_summary.csv`

Share of pairs with fewer than 3 transactions, whose timing-regularity score is imputed (`gap_cv` fill):

| group | pairs | share imputed |
|---|---|---|
| BIPARTITE | 94 | 1.000 |
| CYCLE | 45 | 0.978 |
| FAN-IN | 84 | 1.000 |
| FAN-OUT | 145 | 1.000 |
| GATHER-SCATTER | 90 | 1.000 |
| RANDOM | 63 | 1.000 |
| SCATTER-GATHER | 146 | 1.000 |
| STACK | 212 | 0.995 |
| unattributed | 2408 | 0.678 |
| background | 1193760 | 0.467 |

*Source:* `results/phase3_features/audit/LI_audit_imputation.csv`


### 4.6 HI<->LI transfer

Labelled as **transfer, not a time-leak-free test**: both datasets come from the same generator and cover the same dates; 3 accounts are shared between HI and LI, used by 0 training and 0 test candidates (HI->LI; source `results/phase3_features/ibm_split_counts.csv`). Primary variant (ambiguous excluded in train and test), scaling fitted on the training dataset only, **one model fitted per direction (no repeats)**, CI = grouped bootstrap over the TEST groups.

**Train HI -> test LI**: 9720 test candidates, 38 positives (prevalence 0.0039) in 10 test groups; training positives 131.

| feature set | logreg | rf | svm |
|---|---|---|---|
| B | 0.030 [0.008, 0.075] | 0.019 [0.006, 0.063] | 0.014 [0.005, 0.033] |
| B+D | 0.133 [0.050, 0.318] | 0.149 [0.027, 0.393] | 0.124 [0.055, 0.238] |
| B+D+T | 0.115 [0.019, 0.311] | 0.083 [0.009, 0.232] | 0.074 [0.019, 0.182] |
| B+D+T_local | 0.154 [0.044, 0.363] | 0.095 [0.012, 0.250] | 0.231 [0.036, 0.515] |
| T | 0.046 [0.004, 0.138] | 0.039 [0.002, 0.141] | 0.017 [0.003, 0.112] |
| D | 0.163 [0.027, 0.382] | 0.107 [0.008, 0.339] | 0.026 [0.008, 0.074] |

`-D_birth` row: 0.006 [0.002, 0.025]. **This is the within-dataset score of a no-model detector on the LI candidates, not a transfer result** (nothing is fitted on HI).

| paired difference (B+D+T minus B+D is the pre-specified primary; the T_local row is exploratory) | logreg | rf | svm |
|---|---|---|---|
| B+D+T minus B+D | -0.017 [-0.147, +0.092] | -0.066 [-0.177, -0.012] | -0.050 [-0.142, +0.042] |
| B+D+T_local minus B+D | +0.021 [-0.069, +0.129] | -0.054 [-0.147, -0.009] | +0.107 [-0.051, +0.319] |

**Train LI -> test HI**: 7555 test candidates, 131 positives (prevalence 0.0173) in 14 test groups; training positives 38.

| feature set | logreg | rf | svm |
|---|---|---|---|
| B | 0.078 [0.020, 0.222] | 0.056 [0.021, 0.132] | 0.063 [0.015, 0.208] |
| B+D | 0.405 [0.200, 0.607] | 0.385 [0.179, 0.603] | 0.108 [0.041, 0.192] |
| B+D+T | 0.129 [0.033, 0.228] | 0.366 [0.185, 0.551] | 0.052 [0.028, 0.096] |
| B+D+T_local | 0.044 [0.016, 0.101] | 0.333 [0.187, 0.509] | 0.070 [0.033, 0.121] |
| T | 0.030 [0.007, 0.074] | 0.125 [0.068, 0.207] | 0.063 [0.027, 0.128] |
| D | 0.314 [0.148, 0.549] | 0.344 [0.132, 0.541] | 0.068 [0.023, 0.127] |

`-D_birth` row: 0.314 [0.097, 0.487]. **This is the within-dataset score of a no-model detector on the HI candidates, not a transfer result** (nothing is fitted on LI).

| paired difference (B+D+T minus B+D is the pre-specified primary; the T_local row is exploratory) | logreg | rf | svm |
|---|---|---|---|
| B+D+T minus B+D | -0.276 [-0.550, -0.036] | -0.019 [-0.079, +0.046] | -0.056 [-0.123, +0.025] |
| B+D+T_local minus B+D | -0.361 [-0.566, -0.147] | -0.052 [-0.130, +0.041] | -0.038 [-0.085, +0.007] |

*Source:* `results/phase3_features/classify/transfer/results.csv`, `results/phase3_features/classify/transfer/paired_differences.csv`


## 5. Synthetic data


### 5.1 Condition hn-off_chords-default

Primary (ambiguous excluded): 5071 candidates, 103 positives, prevalence 0.0203, 46 positive groups; seeds pooled, 5 CV repeats. Cell = pooled PR-AUC [95% CI] (mean +- std of the per-seed PR-AUC, 6 seeds).

| feature set | logreg | rf | svm |
|---|---|---|---|
| B | 0.530 [0.431, 0.644] (0.554 +- 0.069) | 0.576 [0.467, 0.694] (0.625 +- 0.080) | 0.482 [0.385, 0.630] (0.526 +- 0.059) |
| B+D | 0.970 [0.930, 0.985] (0.971 +- 0.014) | 0.959 [0.896, 0.990] (0.962 +- 0.055) | 0.892 [0.791, 0.950] (0.914 +- 0.061) |
| B+D+T | 0.988 [0.978, 0.999] (0.990 +- 0.013) | 0.996 [0.989, 1.000] (0.998 +- 0.005) | 0.969 [0.940, 0.996] (0.971 +- 0.037) |
| B+D+T_local | 0.993 [0.985, 1.000] (0.994 +- 0.008) | 0.997 [0.991, 1.000] (0.998 +- 0.004) | 0.987 [0.970, 1.000] (0.984 +- 0.023) |
| T | 0.339 [0.193, 0.666] (0.517 +- 0.188) | 0.617 [0.500, 0.818] (0.663 +- 0.061) | 0.398 [0.273, 0.610] (0.491 +- 0.092) |
| T_local | 0.417 [0.232, 0.714] (0.554 +- 0.156) | 0.617 [0.498, 0.821] (0.669 +- 0.067) | 0.388 [0.230, 0.642] (0.506 +- 0.140) |
| D | 0.968 [0.929, 0.984] (0.971 +- 0.013) | 0.952 [0.882, 0.990] (0.960 +- 0.053) | 0.932 [0.874, 0.967] (0.939 +- 0.015) |

`-D_birth` alone (no model): 0.424 [0.341, 0.640] (0.454 +- 0.077).

| paired difference, overall (B+D+T minus B+D pre-specified; T_local row exploratory) | logreg | rf | svm |
|---|---|---|---|
| B+D+T minus B+D | +0.018 [+0.002, +0.057] | +0.038 [+0.010, +0.095] | +0.077 [+0.019, +0.175] |
| B+D minus B | +0.440 [+0.324, +0.535] | +0.383 [+0.265, +0.477] | +0.411 [+0.213, +0.520] |
| B+D+T_local minus B+D | +0.022 [+0.006, +0.064] | +0.038 [+0.010, +0.097] | +0.094 [+0.032, +0.194] |

Ambiguous counted positive: 8065 candidates, 3097 positives, prevalence 0.384, 47 groups; models run: logreg, rf, svm (limited for runtime).

| feature set | logreg | rf | svm |
|---|---|---|---|
| B | 0.459 [0.425, 0.539] | 0.427 [0.380, 0.512] | 0.453 [0.426, 0.518] |
| B+D | 1.000 [0.999, 1.000] | 1.000 [0.998, 1.000] | 0.999 [0.997, 1.000] |
| B+D+T | 0.963 [0.924, 1.000] | 1.000 [0.999, 1.000] | 0.951 [0.863, 0.998] |
| B+D+T_local | 0.966 [0.935, 1.000] | 1.000 [0.999, 1.000] | 0.993 [0.958, 1.000] |
| T | 0.405 [0.358, 0.455] | 0.430 [0.404, 0.491] | 0.383 [0.373, 0.431] |

*Source:* `results/phase3_features/classify/synthetic/hn-off_chords-default/results.csv`, `results/phase3_features/classify/synthetic/hn-off_chords-default/paired_differences.csv`

Per scenario: positives of ONE scenario against ALL negatives from the same out-of-fold scores (**prevalence changes per scenario**).

*chorded*: 25 positives, prevalence 0.0050, 8 positive groups.

| feature set / paired difference | logreg | rf | svm |
|---|---|---|---|
| B | 0.37 [0.14, 0.59] | 0.55 [0.31, 0.76] | 0.22 [0.12, 0.38] |
| B+D | 0.98 [0.93, 1.00] | 0.90 [0.76, 0.98] | 0.61 [0.36, 0.82] |
| B+D+T | 0.96 [0.91, 1.00] | 0.99 [0.94, 1.00] | 0.89 [0.77, 0.98] |
| B+D+T_local | 0.99 [0.96, 1.00] | 0.99 [0.95, 1.00] | 0.98 [0.93, 1.00] |
| T | 0.25 [0.06, 0.58] | 0.26 [0.09, 0.69] | 0.37 [0.10, 0.75] |
| B+D+T minus B+D | -0.023 [-0.052, +0.014] | +0.085 [+0.021, +0.205] | +0.283 [+0.095, +0.521] |
| B+D+T_local minus B+D | +0.007 [-0.004, +0.039] | +0.087 [+0.021, +0.211] | +0.366 [+0.138, +0.620] |
| B+D minus B | +0.612 [+0.396, +0.840] | +0.351 [+0.184, +0.511] | +0.386 [+0.147, +0.569] |

*dense*: 24 positives, prevalence 0.0048, 10 positive groups.

| feature set / paired difference | logreg | rf | svm |
|---|---|---|---|
| B | 0.02 [0.01, 0.04] | 0.02 [0.01, 0.06] | 0.02 [0.01, 0.03] |
| B+D | 0.94 [0.81, 0.99] | 0.90 [0.70, 0.99] | 0.78 [0.54, 0.91] |
| B+D+T | 0.93 [0.86, 1.00] | 0.95 [0.83, 1.00] | 0.82 [0.59, 0.98] |
| B+D+T_local | 0.94 [0.85, 1.00] | 0.97 [0.89, 1.00] | 0.89 [0.74, 1.00] |
| T | 0.02 [0.01, 0.03] | 0.01 [0.01, 0.03] | 0.01 [0.01, 0.02] |
| B+D+T minus B+D | -0.004 [-0.069, +0.109] | +0.055 [-0.021, +0.205] | +0.038 [-0.134, +0.265] |
| B+D+T_local minus B+D | -0.001 [-0.065, +0.108] | +0.074 [+0.001, +0.229] | +0.112 [-0.045, +0.341] |
| B+D minus B | +0.917 [+0.782, +0.965] | +0.876 [+0.673, +0.964] | +0.759 [+0.518, +0.890] |

*isolated*: 54 positives, prevalence 0.0108, 39 positive groups.

| feature set / paired difference | logreg | rf | svm |
|---|---|---|---|
| B | 0.57 [0.44, 0.71] | 0.58 [0.44, 0.73] | 0.54 [0.42, 0.70] |
| B+D | 0.92 [0.86, 0.96] | 0.90 [0.80, 0.97] | 0.81 [0.69, 0.91] |
| B+D+T | 0.99 [0.97, 1.00] | 1.00 [0.99, 1.00] | 0.97 [0.94, 1.00] |
| B+D+T_local | 0.99 [0.98, 1.00] | 1.00 [0.99, 1.00] | 0.99 [0.97, 1.00] |
| T | 0.31 [0.18, 0.72] | 0.87 [0.78, 0.96] | 0.42 [0.32, 0.66] |
| B+D+T minus B+D | +0.064 [+0.027, +0.125] | +0.096 [+0.031, +0.195] | +0.162 [+0.062, +0.287] |
| B+D+T_local minus B+D | +0.073 [+0.038, +0.133] | +0.096 [+0.031, +0.195] | +0.178 [+0.073, +0.305] |
| B+D minus B | +0.350 [+0.227, +0.469] | +0.318 [+0.194, +0.428] | +0.265 [+0.088, +0.404] |

*Source:* `results/phase3_features/classify/synthetic/hn-off_chords-default/per_scenario.csv`, `results/phase3_features/classify/synthetic/hn-off_chords-default/per_scenario_paired.csv`


### 5.2 Condition hn-off_chords-off

Primary (ambiguous excluded): 3860 candidates, 103 positives, prevalence 0.0267, 46 positive groups; seeds pooled, 5 CV repeats. Cell = pooled PR-AUC [95% CI] (mean +- std of the per-seed PR-AUC, 6 seeds).

| feature set | logreg | rf | svm |
|---|---|---|---|
| B | 0.546 [0.447, 0.664] (0.570 +- 0.084) | 0.567 [0.423, 0.711] (0.625 +- 0.106) | 0.507 [0.389, 0.657] (0.550 +- 0.056) |
| B+D | 0.960 [0.915, 0.980] (0.961 +- 0.016) | 0.956 [0.888, 0.988] (0.959 +- 0.049) | 0.874 [0.774, 0.943] (0.899 +- 0.075) |
| B+D+T | 0.990 [0.977, 0.999] (0.989 +- 0.018) | 0.999 [0.996, 1.000] (1.000 +- 0.001) | 0.970 [0.946, 0.998] (0.985 +- 0.018) |
| B+D+T_local | 0.997 [0.990, 1.000] (0.997 +- 0.006) | 0.999 [0.998, 1.000] (1.000 +- 0.000) | 0.988 [0.969, 1.000] (0.983 +- 0.025) |
| T | 0.230 [0.182, 0.450] (0.354 +- 0.066) | 0.605 [0.479, 0.811] (0.657 +- 0.064) | 0.248 [0.177, 0.465] (0.397 +- 0.156) |
| T_local | 0.309 [0.226, 0.548] (0.405 +- 0.076) | 0.615 [0.488, 0.823] (0.669 +- 0.053) | 0.436 [0.318, 0.620] (0.499 +- 0.079) |
| D | 0.956 [0.914, 0.975] (0.957 +- 0.015) | 0.954 [0.887, 0.988] (0.958 +- 0.044) | 0.932 [0.882, 0.965] (0.937 +- 0.030) |

`-D_birth` alone (no model): 0.452 [0.365, 0.666] (0.480 +- 0.069).

| paired difference, overall (B+D+T minus B+D pre-specified; T_local row exploratory) | logreg | rf | svm |
|---|---|---|---|
| B+D+T minus B+D | +0.030 [+0.010, +0.074] | +0.043 [+0.011, +0.110] | +0.096 [+0.029, +0.210] |
| B+D minus B | +0.414 [+0.290, +0.509] | +0.389 [+0.248, +0.519] | +0.368 [+0.184, +0.487] |
| B+D+T_local minus B+D | +0.037 [+0.017, +0.079] | +0.043 [+0.012, +0.111] | +0.113 [+0.041, +0.218] |

Ambiguous counted positive: 6026 candidates, 2269 positives, prevalence 0.377, 47 groups; models run: logreg, rf, svm (limited for runtime).

| feature set | logreg | rf | svm |
|---|---|---|---|
| B | 0.503 [0.474, 0.585] | 0.526 [0.495, 0.576] | 0.527 [0.492, 0.577] |
| B+D | 1.000 [0.999, 1.000] | 0.999 [0.998, 1.000] | 0.999 [0.996, 1.000] |
| B+D+T | 0.994 [0.989, 1.000] | 0.999 [0.998, 1.000] | 0.911 [0.761, 0.999] |
| B+D+T_local | 0.995 [0.989, 1.000] | 1.000 [0.999, 1.000] | 0.984 [0.928, 1.000] |
| T | 0.448 [0.417, 0.539] | 0.479 [0.430, 0.587] | 0.423 [0.366, 0.543] |

*Source:* `results/phase3_features/classify/synthetic/hn-off_chords-off/results.csv`, `results/phase3_features/classify/synthetic/hn-off_chords-off/paired_differences.csv`

Per scenario: positives of ONE scenario against ALL negatives from the same out-of-fold scores (**prevalence changes per scenario**).

*chorded*: 25 positives, prevalence 0.0066, 8 positive groups.

| feature set / paired difference | logreg | rf | svm |
|---|---|---|---|
| B | 0.38 [0.15, 0.60] | 0.53 [0.31, 0.75] | 0.28 [0.12, 0.51] |
| B+D | 0.98 [0.93, 1.00] | 0.90 [0.76, 0.98] | 0.60 [0.37, 0.81] |
| B+D+T | 0.96 [0.89, 1.00] | 0.99 [0.98, 1.00] | 0.92 [0.84, 0.99] |
| B+D+T_local | 0.99 [0.96, 1.00] | 1.00 [0.99, 1.00] | 0.98 [0.95, 1.00] |
| T | 0.12 [0.04, 0.34] | 0.25 [0.09, 0.63] | 0.21 [0.02, 0.55] |
| B+D+T minus B+D | -0.018 [-0.082, +0.024] | +0.090 [+0.019, +0.219] | +0.319 [+0.130, +0.535] |
| B+D+T_local minus B+D | +0.010 [-0.022, +0.054] | +0.094 [+0.020, +0.235] | +0.385 [+0.174, +0.616] |
| B+D minus B | +0.598 [+0.390, +0.831] | +0.374 [+0.174, +0.548] | +0.321 [+0.055, +0.521] |

*dense*: 24 positives, prevalence 0.0063, 10 positive groups.

| feature set / paired difference | logreg | rf | svm |
|---|---|---|---|
| B | 0.03 [0.02, 0.05] | 0.04 [0.03, 0.07] | 0.03 [0.02, 0.09] |
| B+D | 0.83 [0.60, 0.92] | 0.87 [0.66, 0.97] | 0.70 [0.45, 0.87] |
| B+D+T | 0.96 [0.89, 1.00] | 0.99 [0.96, 1.00] | 0.83 [0.61, 1.00] |
| B+D+T_local | 0.97 [0.92, 1.00] | 0.99 [0.97, 1.00] | 0.91 [0.77, 1.00] |
| T | 0.03 [0.02, 0.06] | 0.02 [0.01, 0.04] | 0.02 [0.01, 0.08] |
| B+D+T minus B+D | +0.130 [+0.038, +0.330] | +0.123 [+0.029, +0.329] | +0.130 [-0.082, +0.440] |
| B+D+T_local minus B+D | +0.147 [+0.056, +0.348] | +0.126 [+0.030, +0.330] | +0.217 [+0.013, +0.492] |
| B+D minus B | +0.797 [+0.556, +0.888] | +0.829 [+0.617, +0.921] | +0.663 [+0.388, +0.826] |

*isolated*: 54 positives, prevalence 0.0142, 39 positive groups.

| feature set / paired difference | logreg | rf | svm |
|---|---|---|---|
| B | 0.57 [0.43, 0.71] | 0.53 [0.37, 0.70] | 0.53 [0.40, 0.69] |
| B+D | 0.91 [0.85, 0.95] | 0.90 [0.80, 0.97] | 0.79 [0.67, 0.90] |
| B+D+T | 0.98 [0.96, 1.00] | 1.00 [1.00, 1.00] | 0.98 [0.95, 1.00] |
| B+D+T_local | 1.00 [0.98, 1.00] | 1.00 [1.00, 1.00] | 0.99 [0.98, 1.00] |
| T | 0.18 [0.12, 0.43] | 0.82 [0.71, 0.94] | 0.20 [0.14, 0.45] |
| B+D+T minus B+D | +0.071 [+0.031, +0.136] | +0.101 [+0.032, +0.202] | +0.191 [+0.082, +0.313] |
| B+D+T_local minus B+D | +0.082 [+0.042, +0.144] | +0.101 [+0.032, +0.203] | +0.203 [+0.095, +0.326] |
| B+D minus B | +0.342 [+0.208, +0.469] | +0.372 [+0.203, +0.512] | +0.253 [+0.085, +0.393] |

*Source:* `results/phase3_features/classify/synthetic/hn-off_chords-off/per_scenario.csv`, `results/phase3_features/classify/synthetic/hn-off_chords-off/per_scenario_paired.csv`


### 5.3 Condition hn-on_chords-default

Primary (ambiguous excluded): 2477 candidates, 102 positives, prevalence 0.0412, 55 positive groups; seeds pooled, 5 CV repeats. Cell = pooled PR-AUC [95% CI] (mean +- std of the per-seed PR-AUC, 6 seeds).

| feature set | logreg | rf | svm |
|---|---|---|---|
| B | 0.441 [0.363, 0.566] (0.481 +- 0.064) | 0.608 [0.500, 0.739] (0.633 +- 0.078) | 0.339 [0.262, 0.478] (0.395 +- 0.080) |
| B+D | 0.983 [0.971, 0.992] (0.985 +- 0.011) | 0.997 [0.992, 1.000] (0.996 +- 0.004) | 0.937 [0.904, 0.967] (0.945 +- 0.033) |
| B+D+T | 0.988 [0.975, 1.000] (0.989 +- 0.017) | 0.994 [0.984, 1.000] (0.992 +- 0.013) | 0.976 [0.954, 0.995] (0.976 +- 0.031) |
| B+D+T_local | 0.993 [0.987, 1.000] (0.993 +- 0.009) | 0.994 [0.986, 1.000] (0.993 +- 0.011) | 0.982 [0.965, 0.998] (0.980 +- 0.021) |
| T | 0.546 [0.427, 0.752] (0.599 +- 0.072) | 0.676 [0.568, 0.837] (0.698 +- 0.038) | 0.420 [0.333, 0.628] (0.523 +- 0.160) |
| T_local | 0.584 [0.458, 0.780] (0.620 +- 0.054) | 0.683 [0.573, 0.845] (0.702 +- 0.046) | 0.486 [0.379, 0.699] (0.550 +- 0.131) |
| D | 0.983 [0.969, 0.993] (0.983 +- 0.012) | 0.996 [0.990, 0.999] (0.995 +- 0.005) | 0.955 [0.935, 0.976] (0.959 +- 0.014) |

`-D_birth` alone (no model): 0.266 [0.178, 0.459] (0.276 +- 0.011).

| paired difference, overall (B+D+T minus B+D pre-specified; T_local row exploratory) | logreg | rf | svm |
|---|---|---|---|
| B+D+T minus B+D | +0.005 [-0.009, +0.021] | -0.003 [-0.012, +0.004] | +0.038 [-0.004, +0.080] |
| B+D minus B | +0.542 [+0.418, +0.621] | +0.389 [+0.258, +0.494] | +0.598 [+0.458, +0.676] |
| B+D+T_local minus B+D | +0.010 [-0.000, +0.024] | -0.003 [-0.011, +0.004] | +0.045 [+0.009, +0.082] |

Ambiguous counted positive: 67757 candidates, 65382 positives, prevalence 0.965, 57 groups; models run: logreg (limited for runtime).

| feature set | logreg | rf | svm |
|---|---|---|---|
| B | 1.000 [0.999, 1.000] | not run | not run |
| B+D | 1.000 [1.000, 1.000] | not run | not run |
| B+D+T | 1.000 [1.000, 1.000] | not run | not run |
| B+D+T_local | 1.000 [1.000, 1.000] | not run | not run |
| T | 0.999 [0.998, 1.000] | not run | not run |

*Source:* `results/phase3_features/classify/synthetic/hn-on_chords-default/results.csv`, `results/phase3_features/classify/synthetic/hn-on_chords-default/paired_differences.csv`

Per scenario: positives of ONE scenario against ALL negatives from the same out-of-fold scores (**prevalence changes per scenario**).

*chorded*: 24 positives, prevalence 0.0100, 9 positive groups.

| feature set / paired difference | logreg | rf | svm |
|---|---|---|---|
| B | 0.28 [0.17, 0.47] | 0.68 [0.48, 0.84] | 0.14 [0.07, 0.25] |
| B+D | 1.00 [0.98, 1.00] | 1.00 [1.00, 1.00] | 0.70 [0.50, 0.87] |
| B+D+T | 0.97 [0.91, 1.00] | 0.99 [0.95, 1.00] | 0.93 [0.82, 0.99] |
| B+D+T_local | 0.99 [0.97, 1.00] | 0.99 [0.96, 1.00] | 0.97 [0.89, 1.00] |
| T | 0.36 [0.19, 0.60] | 0.40 [0.25, 0.65] | 0.19 [0.10, 0.39] |
| B+D+T minus B+D | -0.023 [-0.081, +0.008] | -0.011 [-0.047, +0.004] | +0.229 [+0.049, +0.449] |
| B+D+T_local minus B+D | -0.004 [-0.026, +0.010] | -0.010 [-0.041, +0.004] | +0.271 [+0.125, +0.462] |
| B+D minus B | +0.714 [+0.524, +0.828] | +0.324 [+0.157, +0.521] | +0.569 [+0.354, +0.717] |

*dense*: 24 positives, prevalence 0.0100, 10 positive groups.

| feature set / paired difference | logreg | rf | svm |
|---|---|---|---|
| B | 0.03 [0.02, 0.05] | 0.03 [0.02, 0.06] | 0.03 [0.02, 0.04] |
| B+D | 0.99 [0.97, 1.00] | 1.00 [1.00, 1.00] | 0.92 [0.83, 0.97] |
| B+D+T | 0.95 [0.87, 0.99] | 0.90 [0.74, 1.00] | 0.87 [0.78, 0.95] |
| B+D+T_local | 0.96 [0.91, 1.00] | 0.94 [0.83, 1.00] | 0.88 [0.80, 0.94] |
| T | 0.02 [0.02, 0.05] | 0.02 [0.01, 0.04] | 0.04 [0.02, 0.09] |
| B+D+T minus B+D | -0.043 [-0.122, +0.014] | -0.101 [-0.262, +0.001] | -0.056 [-0.167, +0.081] |
| B+D+T_local minus B+D | -0.026 [-0.077, +0.017] | -0.062 [-0.172, +0.002] | -0.047 [-0.123, +0.060] |
| B+D minus B | +0.955 [+0.933, +0.971] | +0.969 [+0.941, +0.981] | +0.896 [+0.799, +0.941] |

*isolated*: 54 positives, prevalence 0.0222, 47 positive groups.

| feature set / paired difference | logreg | rf | svm |
|---|---|---|---|
| B | 0.40 [0.31, 0.54] | 0.57 [0.44, 0.75] | 0.33 [0.24, 0.51] |
| B+D | 0.95 [0.91, 0.98] | 0.99 [0.97, 1.00] | 0.88 [0.81, 0.94] |
| B+D+T | 0.98 [0.95, 1.00] | 0.99 [0.97, 1.00] | 0.97 [0.93, 1.00] |
| B+D+T_local | 0.99 [0.97, 1.00] | 0.99 [0.97, 1.00] | 0.98 [0.94, 1.00] |
| T | 0.62 [0.46, 0.82] | 0.87 [0.77, 0.96] | 0.42 [0.29, 0.71] |
| B+D+T minus B+D | +0.032 [-0.007, +0.068] | +0.002 [-0.017, +0.016] | +0.096 [+0.018, +0.170] |
| B+D+T_local minus B+D | +0.042 [+0.009, +0.077] | +0.002 [-0.017, +0.017] | +0.100 [+0.031, +0.166] |
| B+D minus B | +0.549 [+0.424, +0.631] | +0.421 [+0.249, +0.539] | +0.551 [+0.393, +0.632] |

*Source:* `results/phase3_features/classify/synthetic/hn-on_chords-default/per_scenario.csv`, `results/phase3_features/classify/synthetic/hn-on_chords-default/per_scenario_paired.csv`


### 5.4 Condition hn-on_chords-off

Primary (ambiguous excluded): 1933 candidates, 93 positives, prevalence 0.0481, 52 positive groups; seeds pooled, 5 CV repeats. Cell = pooled PR-AUC [95% CI] (mean +- std of the per-seed PR-AUC, 6 seeds).

| feature set | logreg | rf | svm |
|---|---|---|---|
| B | 0.466 [0.374, 0.604] (0.534 +- 0.076) | 0.599 [0.476, 0.754] (0.640 +- 0.117) | 0.315 [0.237, 0.447] (0.372 +- 0.058) |
| B+D | 0.976 [0.959, 0.987] (0.980 +- 0.012) | 0.994 [0.986, 1.000] (0.995 +- 0.006) | 0.928 [0.884, 0.964] (0.937 +- 0.034) |
| B+D+T | 0.990 [0.975, 1.000] (0.991 +- 0.016) | 0.997 [0.992, 1.000] (0.996 +- 0.006) | 0.980 [0.956, 0.998] (0.977 +- 0.034) |
| B+D+T_local | 0.993 [0.979, 1.000] (0.995 +- 0.013) | 0.998 [0.992, 1.000] (0.997 +- 0.006) | 0.991 [0.975, 1.000] (0.992 +- 0.017) |
| T | 0.629 [0.512, 0.803] (0.684 +- 0.053) | 0.656 [0.533, 0.838] (0.687 +- 0.097) | 0.340 [0.267, 0.566] (0.490 +- 0.134) |
| T_local | 0.642 [0.525, 0.823] (0.683 +- 0.059) | 0.682 [0.563, 0.851] (0.716 +- 0.095) | 0.430 [0.345, 0.679] (0.546 +- 0.132) |
| D | 0.975 [0.956, 0.988] (0.974 +- 0.014) | 0.991 [0.980, 0.999] (0.992 +- 0.008) | 0.944 [0.913, 0.970] (0.946 +- 0.020) |

`-D_birth` alone (no model): 0.278 [0.176, 0.495] (0.297 +- 0.045).

| paired difference, overall (B+D+T minus B+D pre-specified; T_local row exploratory) | logreg | rf | svm |
|---|---|---|---|
| B+D+T minus B+D | +0.014 [-0.006, +0.039] | +0.003 [-0.004, +0.012] | +0.052 [+0.001, +0.109] |
| B+D minus B | +0.510 [+0.376, +0.600] | +0.395 [+0.244, +0.513] | +0.614 [+0.468, +0.698] |
| B+D+T_local minus B+D | +0.017 [-0.002, +0.040] | +0.003 [-0.004, +0.012] | +0.063 [+0.019, +0.114] |

Ambiguous counted positive: 68363 candidates, 66523 positives, prevalence 0.973, 54 groups; models run: logreg (limited for runtime).

| feature set | logreg | rf | svm |
|---|---|---|---|
| B | 1.000 [1.000, 1.000] | not run | not run |
| B+D | 1.000 [1.000, 1.000] | not run | not run |
| B+D+T | 1.000 [1.000, 1.000] | not run | not run |
| B+D+T_local | 1.000 [1.000, 1.000] | not run | not run |
| T | 1.000 [0.999, 1.000] | not run | not run |

*Source:* `results/phase3_features/classify/synthetic/hn-on_chords-off/results.csv`, `results/phase3_features/classify/synthetic/hn-on_chords-off/paired_differences.csv`

Per scenario: positives of ONE scenario against ALL negatives from the same out-of-fold scores (**prevalence changes per scenario**).

*chorded*: 22 positives, prevalence 0.0118, 8 positive groups.

| feature set / paired difference | logreg | rf | svm |
|---|---|---|---|
| B | 0.31 [0.15, 0.54] | 0.55 [0.34, 0.76] | 0.14 [0.07, 0.24] |
| B+D | 0.99 [0.98, 1.00] | 1.00 [0.98, 1.00] | 0.70 [0.50, 0.87] |
| B+D+T | 0.99 [0.97, 1.00] | 1.00 [0.99, 1.00] | 0.95 [0.79, 1.00] |
| B+D+T_local | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [0.99, 1.00] |
| T | 0.48 [0.27, 0.74] | 0.29 [0.10, 0.59] | 0.17 [0.05, 0.45] |
| B+D+T minus B+D | -0.004 [-0.024, +0.017] | +0.003 [-0.008, +0.017] | +0.248 [+0.017, +0.483] |
| B+D+T_local minus B+D | +0.006 [+0.000, +0.019] | +0.003 [-0.003, +0.017] | +0.299 [+0.126, +0.498] |
| B+D minus B | +0.686 [+0.453, +0.840] | +0.448 [+0.243, +0.654] | +0.556 [+0.328, +0.729] |

*dense*: 21 positives, prevalence 0.0113, 9 positive groups.

| feature set / paired difference | logreg | rf | svm |
|---|---|---|---|
| B | 0.05 [0.04, 0.09] | 0.10 [0.04, 0.27] | 0.03 [0.03, 0.06] |
| B+D | 0.95 [0.88, 0.99] | 0.98 [0.97, 1.00] | 0.88 [0.77, 0.95] |
| B+D+T | 0.98 [0.93, 1.00] | 1.00 [0.97, 1.00] | 0.93 [0.83, 0.99] |
| B+D+T_local | 1.00 [0.99, 1.00] | 1.00 [0.98, 1.00] | 0.98 [0.96, 1.00] |
| T | 0.07 [0.05, 0.13] | 0.06 [0.04, 0.14] | 0.06 [0.02, 0.17] |
| B+D+T minus B+D | +0.035 [-0.040, +0.109] | +0.014 [-0.014, +0.028] | +0.053 [-0.068, +0.185] |
| B+D+T_local minus B+D | +0.051 [+0.007, +0.114] | +0.017 [-0.007, +0.030] | +0.106 [+0.040, +0.214] |
| B+D minus B | +0.894 [+0.828, +0.934] | +0.883 [+0.717, +0.940] | +0.842 [+0.731, +0.903] |

*isolated*: 50 positives, prevalence 0.0265, 45 positive groups.

| feature set / paired difference | logreg | rf | svm |
|---|---|---|---|
| B | 0.39 [0.29, 0.56] | 0.50 [0.35, 0.71] | 0.26 [0.17, 0.42] |
| B+D | 0.93 [0.89, 0.97] | 0.99 [0.96, 1.00] | 0.87 [0.80, 0.94] |
| B+D+T | 0.98 [0.94, 1.00] | 0.99 [0.97, 1.00] | 0.97 [0.93, 1.00] |
| B+D+T_local | 0.98 [0.95, 1.00] | 0.99 [0.97, 1.00] | 0.98 [0.94, 1.00] |
| T | 0.64 [0.46, 0.83] | 0.80 [0.65, 0.92] | 0.28 [0.16, 0.54] |
| B+D+T minus B+D | +0.046 [-0.007, +0.097] | +0.007 [-0.017, +0.038] | +0.106 [+0.016, +0.187] |
| B+D+T_local minus B+D | +0.050 [-0.002, +0.102] | +0.008 [-0.018, +0.038] | +0.115 [+0.029, +0.194] |
| B+D minus B | +0.545 [+0.389, +0.633] | +0.482 [+0.281, +0.613] | +0.610 [+0.456, +0.700] |

*Source:* `results/phase3_features/classify/synthetic/hn-on_chords-off/per_scenario.csv`, `results/phase3_features/classify/synthetic/hn-on_chords-off/per_scenario_paired.csv`


### 5.5 Dense scenario under both chord settings

| condition | dense positive candidates | with T_local_ring_inf = 0 | fraction | planted dense rings with a positive candidate | of those with T_local_ring_inf = 1 |
|---|---|---|---|---|---|
| hn-off_chords-default | 24 | 17 | 0.708 | 24 | 7 |
| hn-off_chords-off | 24 | 6 | 0.250 | 24 | 18 |
| hn-on_chords-default | 24 | 15 | 0.625 | 24 | 9 |
| hn-on_chords-off | 21 | 3 | 0.143 | 21 | 18 |

Positives with `T_local_ring_inf = 0` keep their positive label (the column is a feature). Dense-scenario PR-AUC and paired differences for each setting are in the per-scenario tables above.

| condition | dense paired difference | logreg | rf | svm |
|---|---|---|---|---|
| hn-off_chords-default | B+D+T minus B+D | -0.004 [-0.069, +0.109] | +0.055 [-0.021, +0.205] | +0.038 [-0.134, +0.265] |
| hn-off_chords-default | B+D+T_local minus B+D | -0.001 [-0.065, +0.108] | +0.074 [+0.001, +0.229] | +0.112 [-0.045, +0.341] |
| hn-off_chords-off | B+D+T minus B+D | +0.130 [+0.038, +0.330] | +0.123 [+0.029, +0.329] | +0.130 [-0.082, +0.440] |
| hn-off_chords-off | B+D+T_local minus B+D | +0.147 [+0.056, +0.348] | +0.126 [+0.030, +0.330] | +0.217 [+0.013, +0.492] |
| hn-on_chords-default | B+D+T minus B+D | -0.043 [-0.122, +0.014] | -0.101 [-0.262, +0.001] | -0.056 [-0.167, +0.081] |
| hn-on_chords-default | B+D+T_local minus B+D | -0.026 [-0.077, +0.017] | -0.062 [-0.172, +0.002] | -0.047 [-0.123, +0.060] |
| hn-on_chords-off | B+D+T minus B+D | +0.035 [-0.040, +0.109] | +0.014 [-0.014, +0.028] | +0.053 [-0.068, +0.185] |
| hn-on_chords-off | B+D+T_local minus B+D | +0.051 [+0.007, +0.114] | +0.017 [-0.007, +0.030] | +0.106 [+0.040, +0.214] |

*Source:* `results/phase3_features/classify/synthetic/hn-off_chords-default/positives_ring_inf.csv`, `results/phase3_features/classify/synthetic/hn-off_chords-off/positives_ring_inf.csv`, `results/phase3_features/classify/synthetic/hn-on_chords-default/positives_ring_inf.csv`, `results/phase3_features/classify/synthetic/hn-on_chords-off/positives_ring_inf.csv`, `results/phase3_features/classify/synthetic/hn-off_chords-default/per_scenario_paired.csv`, `results/phase3_features/classify/synthetic/hn-off_chords-off/per_scenario_paired.csv`, `results/phase3_features/classify/synthetic/hn-on_chords-default/per_scenario_paired.csv`, `results/phase3_features/classify/synthetic/hn-on_chords-off/per_scenario_paired.csv`


### 5.6 Mean out-of-fold score rank percentile by candidate kind

Used instead of a false-positive rate at 80% recall. 1 = top of the ranking, 0.5 = middle. Primary variant, mean over CV repeats; `n` is the number of candidates of that kind.

*hn-off_chords-default, logreg* (n: wash_ring 103, shortcut_filled 18, background 4950)

| feature set | wash_ring | shortcut_filled | background |
|---|---|---|---|
| B | 0.942 | 0.990 | 0.489 |
| B+D | 0.989 | 0.982 | 0.488 |
| B+D+T | 0.990 | 0.909 | 0.488 |
| B+D+T_local | 0.990 | 0.816 | 0.489 |
| T | 0.906 | 0.639 | 0.491 |

*hn-off_chords-default, rf* (n: wash_ring 103, shortcut_filled 18, background 4950)

| feature set | wash_ring | shortcut_filled | background |
|---|---|---|---|
| B | 0.928 | 0.988 | 0.489 |
| B+D | 0.989 | 0.981 | 0.488 |
| B+D+T | 0.990 | 0.960 | 0.488 |
| B+D+T_local | 0.990 | 0.946 | 0.488 |
| T | 0.892 | 0.258 | 0.493 |

*Source:* `results/phase3_features/classify/synthetic/hn-off_chords-default/negatives_rank.csv`

*hn-off_chords-off, logreg* (n: wash_ring 103, shortcut_filled 18, background 3739)

| feature set | wash_ring | shortcut_filled | background |
|---|---|---|---|
| B | 0.948 | 0.984 | 0.485 |
| B+D | 0.986 | 0.977 | 0.484 |
| B+D+T | 0.986 | 0.772 | 0.485 |
| B+D+T_local | 0.987 | 0.575 | 0.486 |
| T | 0.888 | 0.230 | 0.491 |

*hn-off_chords-off, rf* (n: wash_ring 103, shortcut_filled 18, background 3739)

| feature set | wash_ring | shortcut_filled | background |
|---|---|---|---|
| B | 0.940 | 0.980 | 0.486 |
| B+D | 0.986 | 0.976 | 0.484 |
| B+D+T | 0.987 | 0.939 | 0.484 |
| B+D+T_local | 0.987 | 0.948 | 0.484 |
| T | 0.903 | 0.148 | 0.491 |

*Source:* `results/phase3_features/classify/synthetic/hn-off_chords-off/negatives_rank.csv`

*hn-on_chords-default, logreg* (n: wash_ring 102, shortcut_filled 18, background 1284, arbitrage_loop 17, market_maker 60, hot_wallet 996)

| feature set | wash_ring | shortcut_filled | background | arbitrage_loop | market_maker | hot_wallet |
|---|---|---|---|---|---|---|
| B | 0.928 | 0.975 | 0.639 | 0.988 | 0.888 | 0.237 |
| B+D | 0.979 | 0.959 | 0.564 | 0.746 | 0.763 | 0.340 |
| B+D+T | 0.979 | 0.928 | 0.552 | 0.626 | 0.846 | 0.353 |
| B+D+T_local | 0.979 | 0.949 | 0.549 | 0.577 | 0.847 | 0.358 |
| T | 0.888 | 0.644 | 0.662 | 0.751 | 0.783 | 0.228 |

*hn-on_chords-default, rf* (n: wash_ring 102, shortcut_filled 18, background 1284, arbitrage_loop 17, market_maker 60, hot_wallet 996)

| feature set | wash_ring | shortcut_filled | background | arbitrage_loop | market_maker | hot_wallet |
|---|---|---|---|---|---|---|
| B | 0.925 | 0.969 | 0.628 | 0.940 | 0.829 | 0.256 |
| B+D | 0.979 | 0.956 | 0.474 | 0.715 | 0.532 | 0.471 |
| B+D+T | 0.979 | 0.895 | 0.560 | 0.730 | 0.777 | 0.346 |
| B+D+T_local | 0.979 | 0.884 | 0.518 | 0.663 | 0.708 | 0.405 |
| T | 0.894 | 0.538 | 0.651 | 0.418 | 0.765 | 0.250 |

*Source:* `results/phase3_features/classify/synthetic/hn-on_chords-default/negatives_rank.csv`

*hn-on_chords-off, logreg* (n: wash_ring 93, shortcut_filled 17, background 829, arbitrage_loop 17, market_maker 55, hot_wallet 922)

| feature set | wash_ring | shortcut_filled | background | arbitrage_loop | market_maker | hot_wallet |
|---|---|---|---|---|---|---|
| B | 0.933 | 0.965 | 0.655 | 0.983 | 0.856 | 0.279 |
| B+D | 0.975 | 0.954 | 0.544 | 0.811 | 0.808 | 0.380 |
| B+D+T | 0.974 | 0.909 | 0.594 | 0.364 | 0.800 | 0.345 |
| B+D+T_local | 0.975 | 0.926 | 0.587 | 0.356 | 0.826 | 0.349 |
| T | 0.923 | 0.586 | 0.697 | 0.402 | 0.843 | 0.260 |

*hn-on_chords-off, rf* (n: wash_ring 93, shortcut_filled 17, background 829, arbitrage_loop 17, market_maker 55, hot_wallet 922)

| feature set | wash_ring | shortcut_filled | background | arbitrage_loop | market_maker | hot_wallet |
|---|---|---|---|---|---|---|
| B | 0.939 | 0.960 | 0.631 | 0.937 | 0.790 | 0.304 |
| B+D | 0.976 | 0.949 | 0.479 | 0.658 | 0.528 | 0.458 |
| B+D+T | 0.976 | 0.882 | 0.589 | 0.589 | 0.782 | 0.346 |
| B+D+T_local | 0.976 | 0.860 | 0.554 | 0.628 | 0.759 | 0.379 |
| T | 0.908 | 0.533 | 0.693 | 0.391 | 0.689 | 0.276 |

*Source:* `results/phase3_features/classify/synthetic/hn-on_chords-off/negatives_rank.csv`


### 5.7 Candidate kinds and ambiguity per condition (all seeds pooled)

| condition | candidates | ambiguous | ambiguous share | wash_ring | shortcut_filled | arbitrage_loop | market_maker | hot_wallet | background |
|---|---|---|---|---|---|---|---|---|---|
| hn-off_chords-default | 8065 | 2994 | 0.371 | 103 | 18 | 0 | 0 | 0 | 4950 |
| hn-off_chords-off | 6026 | 2166 | 0.359 | 103 | 18 | 0 | 0 | 0 | 3739 |
| hn-on_chords-default | 67757 | 65280 | 0.963 | 102 | 18 | 17 | 60 | 996 | 1284 |
| hn-on_chords-off | 68363 | 66430 | 0.972 | 93 | 17 | 17 | 55 | 922 | 829 |

Sources: per-seed candidate tables `results/phase3_features/classify/synthetic/cache/<condition>_seed<k>.pkl` (written by `experiments/synth_classify.py`).


## 6. Candidate-generation recall


### 6.1 IBM AML

Complete = CYCLE attempts with all transactions inside a window; reachable = complete attempts with at most `max_len` accounts. **Recovered (account set)** = a candidate of the same window has exactly the attempt's account set (the README measurement). **Recovered (labelled)** = in addition the candidate carries a positive CYCLE label under the §3.7 rule. The difference between the two columns is the account-set vs label-based gap. Scope `days0-5` is window 0 only; `whole_data` is the union over all windows.

| dataset | scope | max_len | tau_quantile | complete | reachable | recovered (account set) | recovered (labelled) | recall of complete | recall of reachable | candidates per window |
|---|---|---|---|---|---|---|---|---|---|---|
| HI | days0-5 | 6 | 0.02 | 15 | 11 | 8 | 6 | 0.533 | 0.727 | 1102 |
| HI | days0-5 | 6 | 0.05 | 15 | 11 | 8 | 6 | 0.533 | 0.727 | 1113 |
| HI | days0-5 | 6 | 0.1 | 15 | 11 | 8 | 6 | 0.533 | 0.727 | 1115 |
| HI | days0-5 | 6 | 0.15 | 15 | 11 | 9 | 7 | 0.600 | 0.818 | 1117 |
| HI | days0-5 | 8 | 0.02 | 15 | 12 | 8 | 6 | 0.533 | 0.667 | 1102 |
| HI | days0-5 | 8 | 0.05 | 15 | 12 | 8 | 6 | 0.533 | 0.667 | 1113 |
| HI | days0-5 | 8 | 0.1 | 15 | 12 | 8 | 6 | 0.533 | 0.667 | 1115 |
| HI | days0-5 | 8 | 0.15 | 15 | 12 | 9 | 7 | 0.600 | 0.750 | 1117 |
| HI | whole_data | 6 | 0.02 | 54 | 34 | 18 | 14 | 0.333 | 0.529 | 620 |
| HI | whole_data | 6 | 0.05 | 54 | 34 | 19 | 15 | 0.352 | 0.559 | 621 |
| HI | whole_data | 6 | 0.1 | 54 | 34 | 21 | 18 | 0.389 | 0.618 | 622 |
| HI | whole_data | 6 | 0.15 | 54 | 34 | 27 | 24 | 0.500 | 0.794 | 628 |
| HI | whole_data | 8 | 0.02 | 54 | 43 | 18 | 14 | 0.333 | 0.419 | 620 |
| HI | whole_data | 8 | 0.05 | 54 | 43 | 20 | 16 | 0.370 | 0.465 | 621 |
| HI | whole_data | 8 | 0.1 | 54 | 43 | 23 | 19 | 0.426 | 0.535 | 622 |
| HI | whole_data | 8 | 0.15 | 54 | 43 | 34 | 30 | 0.630 | 0.791 | 629 |
| LI | days0-5 | 6 | 0.02 | 1 | 1 | 0 | 0 | 0.000 | 0.000 | 1357 |
| LI | days0-5 | 6 | 0.05 | 1 | 1 | 0 | 0 | 0.000 | 0.000 | 1373 |
| LI | days0-5 | 6 | 0.1 | 1 | 1 | 0 | 0 | 0.000 | 0.000 | 1373 |
| LI | days0-5 | 6 | 0.15 | 1 | 1 | 0 | 0 | 0.000 | 0.000 | 1374 |
| LI | days0-5 | 8 | 0.02 | 1 | 1 | 0 | 0 | 0.000 | 0.000 | 1357 |
| LI | days0-5 | 8 | 0.05 | 1 | 1 | 0 | 0 | 0.000 | 0.000 | 1373 |
| LI | days0-5 | 8 | 0.1 | 1 | 1 | 0 | 0 | 0.000 | 0.000 | 1373 |
| LI | days0-5 | 8 | 0.15 | 1 | 1 | 0 | 0 | 0.000 | 0.000 | 1374 |
| LI | whole_data | 6 | 0.02 | 12 | 6 | 3 | 3 | 0.250 | 0.500 | 849 |
| LI | whole_data | 6 | 0.05 | 12 | 6 | 3 | 3 | 0.250 | 0.500 | 850 |
| LI | whole_data | 6 | 0.1 | 12 | 6 | 3 | 3 | 0.250 | 0.500 | 851 |
| LI | whole_data | 6 | 0.15 | 12 | 6 | 6 | 6 | 0.500 | 1.000 | 859 |
| LI | whole_data | 8 | 0.02 | 12 | 7 | 3 | 3 | 0.250 | 0.429 | 849 |
| LI | whole_data | 8 | 0.05 | 12 | 7 | 3 | 3 | 0.250 | 0.429 | 850 |
| LI | whole_data | 8 | 0.1 | 12 | 7 | 3 | 3 | 0.250 | 0.429 | 851 |
| LI | whole_data | 8 | 0.15 | 12 | 7 | 6 | 6 | 0.500 | 0.857 | 859 |

*Source:* `results/phase3_features/classify/candidate_recall/ibm.csv`


### 6.2 Synthetic planted wash rings

Recall = share of planted wash rings (6 seeds) with a candidate labelled as that ring; `max_len` 6. **†** marks cells where the per-SCC enumeration cap (20000 cycles) was hit in at least one of the pooled windows; the *truncated share* column is the fraction of the pooled windows in the row's cell with at least one truncated SCC. Recall there is limited by the cap, not by tau.

*hard negatives off, dense_internal_chords default*

| scenario \ tau_quantile | 0.05 | 0.1 | 0.15 | 0.2 | 0.25 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|---|---|---|
| isolated | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| dense | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| chorded | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| mean candidates per window | 116 | 166 | 448 | 966 | 1344 | 2002 | 3832 | 5888 |
| truncated share of windows | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

*hard negatives off, dense_internal_chords off*

| scenario \ tau_quantile | 0.05 | 0.1 | 0.15 | 0.2 | 0.25 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|---|---|---|
| isolated | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| dense | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| chorded | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| mean candidates per window | 95 | 137 | 333 | 688 | 1004 | 1584 | 3341 | 5321 |
| truncated share of windows | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

*hard negatives on, dense_internal_chords default*

| scenario \ tau_quantile | 0.05 | 0.1 | 0.15 | 0.2 | 0.25 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|---|---|---|
| isolated | 0.52 | 0.91 | 1.00 | 1.00 | 1.00 | 0.72† | 0.20† | 0.07† |
| dense | 0.21 | 0.92 | 1.00 | 1.00 | 1.00 | 0.50† | 0.00† | 0.00† |
| chorded | 0.42 | 0.83 | 1.00 | 1.00 | 1.00 | 0.50† | 0.00† | 0.00† |
| mean candidates per window | 85 | 551 | 1939 | 5836 | 11293 | 16981 | 20002 | 20001 |
| truncated share of windows | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | 1.00 | 1.00 |

*hard negatives on, dense_internal_chords off*

| scenario \ tau_quantile | 0.05 | 0.1 | 0.15 | 0.2 | 0.25 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|---|---|---|
| isolated | 0.52 | 0.93 | 1.00 | 1.00 | 0.93† | 0.72† | 0.20† | 0.07† |
| dense | 0.25 | 0.92 | 1.00 | 1.00 | 0.88† | 0.50† | 0.00† | 0.00† |
| chorded | 0.50 | 0.83 | 1.00 | 1.00 | 0.92† | 0.50† | 0.00† | 0.00† |
| mean candidates per window | 87 | 549 | 1969 | 6045 | 11394 | 16944 | 20002 | 20001 |
| truncated share of windows | 0.00 | 0.00 | 0.00 | 0.00 | 0.17 | 0.50 | 1.00 | 1.00 |

Planted wash rings per window (per condition, per seed): isolated 9, dense 4, chorded 2 (seed 0).

*Source:* `results/phase3_features/classify/candidate_recall/synthetic.csv`


## 7. Limitations and caveats

1. Few positive groups: 14 on HI and 10 on LI (positives 131 and 38 candidates); the effective sample size is the group count, so CIs are wide.
2. LI has 3 cycle positives in 3 groups: the cycle-only variant is unusable there (fewest positive groups on a train or test side: 1); LI primary has 1 on its thinnest side.
3. LI contradicts HI on the ring-birth signal: `-D_birth` alone scores 0.006 on LI (prevalence 0.0039) against 0.314 on HI (prevalence 0.0173).
4. `T_win` alone on HI: PR-AUC 0.121 (logreg) and 0.130 (rf) at prevalence 0.0173: window-constant features rank well above chance (consistent with a time proxy, not tested directly); on LI the same sets score 0.006 / 0.004 at prevalence 0.0039.
5. The synthetic generator is near-separable by D: D alone scores 0.952 to 0.996 (logreg and rf, four conditions), so there is little room for B+D+T to add; the other models and `-D_birth` alone are in section 5.
6. Imputed timing: on HI, 0.685 to 0.966 of each laundering group's pairs have fewer than 3 transactions (`gap_cv` imputed), against 0.467 for background pairs, so the timing-regularity score of most laundering pairs is the imputed constant.
7. Betweenness is sampled (k=100) on IBM windows: 0.1594 of the 404,273 nodes of HI window 0 have non-zero betweenness; B is exact only up to 3000 nodes (synthetic).
8. Recall caps: HI days 0-5 has 15 complete CYCLE attempts, only 11 with at most 6 accounts (12 at max_len 8); 8 recovered by account set but 6 also labelled positive. LI days 0-5 has 1 complete attempt (not a usable rate).
9. Synthetic recall with hard negatives on falls at large tau, where the enumeration cap (20000 cycles per SCC) is hit: rows with a truncated SCC recover 0.13 of planted rings (465 ring-cells), untruncated rows at tau >= 0.15 recover 1.00; at tau 0.05 untruncated recall is 0.43.
10. Hard-negative ambiguity: with hard negatives on, 0.963 (hn-on_chords-default) and 0.972 (hn-on_chords-off) of all candidates are ambiguous, so the ambiguous-positive variant has a prevalence near 1 and is run for logreg only.
11. Smallest per-scenario positive-group count on synthetic: 8; chorded and dense scenarios have few groups, so scenario-level CIs are wide.
12. Transfer fits one model per direction (no repeats); test positive groups: 10 (LI) and 14 (HI); both datasets come from the same generator and cover the same dates.
13. Exploratory comparisons (T_local / T_win ablations) were added after the first HI run; 36 such paired rows exist in the HI file and 18 in the LI file; no multiplicity correction.
14. The RBF SVM (fixed C=1, gamma=scale, untuned) is weaker than rf on HI: B+D 0.326 vs 0.652, B+D+T 0.329 vs 0.676.
15. Currencies are mixed (`Amount Paid` as is); the IBM data are synthetic AML data, not market transactions; the optimisation report numbers come from the pinned report generator (section 2).


## 8. Claims the data supports / does not support

**Supported:**

- Adding D to B raises PR-AUC on HI: the paired CI excludes 0 in 3 of 3 models (logreg +0.399, rf +0.572, svm +0.233).
- On LI, adding D to B: the paired CI excludes 0 in 2 of 3 models (logreg +0.072 [+0.020, +0.173], rf +0.132 [+0.045, +0.283], svm +0.105 [-0.060, +0.276]).
- On synthetic data B+D+T improves on B+D by a paired CI above 0 in 7 of 12 condition-model cells (hn-off_chords-default: 3/3; hn-off_chords-off: 3/3; hn-on_chords-default: 0/3; hn-on_chords-off: 1/3); B+D itself scores 0.960 to 0.997 (logreg minimum, rf maximum over the four conditions).
- Exploratory: the local topology columns alone (B+D+T_local) add to B+D on HI for logreg only (CI excludes 0 in 1 of 3 models: logreg +0.141, rf +0.041, svm +0.037); this is one of the exploratory comparisons and is not corrected for multiplicity.
- Candidate recall is partial and tau-dependent (section 6): HI days 0-5, tau 0.05, max_len 6 recovers 8 of 11 reachable complete CYCLE attempts (15 complete); synthetic recall is 1.00 with hard negatives off at every tau tested.

**Not supported:**

- Topology adds to B+D on IBM HI (pre-specified primary comparison): the CI excludes 0 in 0 of 3 models (logreg +0.104 [-0.002, +0.171], rf +0.025 [-0.050, +0.102], svm +0.003 [-0.111, +0.077]).
- Topology adds to B+D on IBM LI: the CI excludes 0 in 0 of 3 models (logreg +0.033, rf -0.012, svm -0.044).
- Transfer HI->LI: B+D+T does not beat B+D (CI above 0 in 0, below 0 in 1 of 3 models: logreg -0.017, rf -0.066, svm -0.050).
- Transfer LI->HI: B+D+T does not beat B+D (CI above 0 in 0, below 0 in 1 of 3 models: logreg -0.276, rf -0.019, svm -0.056).
- That the ring-birth signal carries over from HI to LI: `-D_birth` alone is 0.006 on LI at prevalence 0.0039 (HI: 0.314).
- That the synthetic topology gains predict IBM results: B+D+T minus B+D has a CI above 0 in 7 of 12 synthetic cells and in 0 IBM cells (HI primary, LI primary); IBM has few positive groups, so absence of a detected gain is not evidence of no gain.
- That the `T_win` contribution on HI is topological: `T_win` columns are constant within a window; alone they score 0.121 (logreg) on HI at prevalence 0.0173 and 0.006 on LI at prevalence 0.0039; a time proxy is consistent with this but was not tested directly.


## 9. Deviations from the v3 proposal

1. Candidate de-duplication order: the proposal keeps the lowest-birth ring among near-duplicates; the code de-duplicates shortest-first, then by birth. At overlap 0.8 nothing is removed for max_len <= 9 (first removals at 10; README), so near-duplicate rings all stay candidates and CV groups by ring identity.
2. max_len = 6 for the Phase 3 tables (the optimization report keeps 5).
3. Infinite-bar cap: Phase 3 T features use the window-maximum cap; the optimization report uses the frozen legacy `topo_*` scalar features with a fixed cap of 1.0 (two T pipelines coexist).
4. Windows are 5 days with a 1-day stride on IBM (cycle attempts last up to 96 h; README).
5. Time-ordered evaluation was tried as specified and dropped: HI leaves 0 test candidates and LI 1 (`eval/splits.time_ordered_split` is kept, unused); grouped CV is primary and HI<->LI transfer the out-of-sample check.
6. Cycle-only subset: positives are candidates labelled from CYCLE attempts; other-type positives and ambiguous candidates are removed, background kept (not defined in the proposal). Usable on HI only.
7. Ambiguity: candidates whose laundering edges rest only on unattributed rows, or whose backing is split across structures, are `ambiguous`; results are reported with them excluded (primary) and counted positive.
8. Hard-negative false-positive rate at 80% recall was replaced by the mean out-of-fold score rank percentile of each negative kind.
9. Recall of complete cycle attempts uses the account-set definition (README measurement); the label-based count is reported next to it.
10. Phases 4-6 are not done (architecture.md: Phase 4 clustering and severity regression, Phase 5 NFT loader and label-provenance audit, Phase 6 bandit policies and drift benchmark). The adaptive-threshold study exists only in the optimization report (section 2).
11. Learned-weight (Problem B), classifier (Problem A) and threshold (Problem C) results in section 2 come from the pinned report generator, not from IBM AML.


## Expected files that were missing

None.
