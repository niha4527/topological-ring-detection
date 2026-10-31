
### 6.7 Labels (`eval/labels.py`)

**The ring-labelling rule (v3 section 3.7, X = 80%).** A transaction *backs* an edge for a structure if it is one of the structure's *own* transactions (decoy, context and background rows back nothing). For a candidate ring with edges E:

* if at least X = 80% of E is backed by **one** structure, the candidate gets that structure's pattern as its kind (synthetic: `wash_ring` gives y = 1, any other pattern y = 0; IBM: any laundering attempt type gives y = 1);
* if **no** edge is backed it is `background` (y = 0); on IBM, a ring whose flagged edges rest only on laundering rows that belong to no attempt is `ambiguous` instead;
* otherwise it is `ambiguous` (y undefined).

For planted rings of size 2, 3 and 5 the 80% and 100% rules agree; they differ only for structures with more edges than one cycle (a chorded 6-ring: four of five edges backed counts as positive at 80%).

**Both ways of counting ambiguity.** Ambiguous candidates are reported **excluded** (the *primary* variant) and **counted positive** (the *ambiguous-positive* variant). Synthetic runs with hard negatives on show why this matters: most candidates are ambiguous (section 8.6).

**Cycle-only subset.** Positives from CYCLE attempts only, with other-type positives and ambiguous candidates removed and background kept. The proposal does not define it; this is our definition (section 11.3).

### 6.8 Groups, and why rows are not independent (`eval/splits.py`)

Because dedup is effectively off and rings recur across overlapping windows, many candidates are near-copies of each other. A **group** is one unit that must stay together in cross-validation: assigned once across all windows by union-find over candidates that **share any edge**, additionally merged by **attempt id** (all candidates labelled with the same IBM attempt, or the same planted structure). A stricter **strict-groups** variant merges candidates sharing an *account*. On synthetic data groups are prefixed by seed so seeds never merge.

### 6.9 The classification harness (`models/classify.py`, `eval/metrics.py`, `experiments/ibm_classify.py`)

* **Feature sets.** B, B+D, B+D+T, T; controls D, T_local, T_win; ablations B+D+T_local and B+D+T_win (added after the first run).
* **Models, hyperparameters fixed in `configs/classify.yaml`, no search of any kind.** Logistic regression (L2, C = 1.0, 2000 iterations, class weights balanced); random forest (300 trees, balanced_subsample); RBF-SVM (C = 1.0, gamma = scale, balanced). Each is a pipeline that includes a standard scaler, so **scaling is fitted on the training fold only**.
* **Cross-validation.** Stratified *group* k-fold (k = 5; falls back to 3 if some fold would hold no positive), **5 repeats** with different seeds, so every candidate gets 5 out-of-fold scores.
* **Metric: PR-AUC** (average precision) of the **pooled out-of-fold scores**, averaged over the 5 repeats. Prevalence is always printed beside it, because the PR-AUC of a random scorer equals the prevalence.
* **Confidence intervals: grouped bootstrap.** Resample *groups* with replacement (a group's candidates get its multiplicity as weight), recompute the AP of every repeat on each resample, average over repeats, take the 2.5th and 97.5th percentiles of 1000 resamples (seed 12345).
* **Paired differences.** To compare feature sets A and B, the *same* resamples are used for both and the CI is of the difference A minus B. This is much tighter than comparing two separate intervals.
* **Single-feature reference.** `-D_birth` alone (low birth = suspicious) with no model.

**A bug that was found and fixed.** The first harness run computed AP with 0/0 at the head of the ranking when the top-ranked candidates' groups were absent from a bootstrap resample; the resulting NaNs were silently dropped, which kept only resamples containing the top items and **biased the CIs upward** (one paired CI printed [+0.032, +0.172] with "share of resamples above 0" of 0.26, which was inconsistent). The fix makes the AP zero-safe, only resamples with no positive at all are undefined; the point estimates did not change, the CIs got wider (section 12). All reported CIs are from the corrected code, with no NaN resamples.

### 6.10 The optimization report (`models/gd.py`, `threshold/ascent.py`, `experiments/gd_report.py`)

This is the earlier, self-contained part of the project: three small optimization problems written out in NumPy so each line maps to an equation of the report. It is run on a pinned synthetic generator and **its numbers are never overwritten** (`results/report/`, `results/gd_metrics.json`, `results/figures/`).

* **Problem A, a classifier by gradient descent.** Class-weighted, L2-regularised logistic regression solved by plain gradient descent. The objective is convex, so every stationary point is global; the step size 1/L (L = smoothness constant) is safe, and a step matched to the *local* curvature at the optimum converges in 322 iterations.
* **Problem B, learning alpha on the simplex.** Choose the score weights alpha (non-negative, summing to 1) to minimise a weighted logistic loss of `sigmoid(c - k * ring_birth(alpha))`, where ring birth is smoothed by a softmax (beta = 30) so it is differentiable. Each step moves alpha by a gradient step and **projects back onto the simplex**; the gradient check agrees with finite differences to about 1e-10. This is non-convex, so four starts (equal weights and the three vertices) were run and they land on nearly the same alpha.
* **Problem C, online threshold ascent.** Smooth the step-function reward `TP - lambda * FP` with a sigmoid of width T, and move tau uphill on the newest labelled window (20 small steps per window), with labels arriving d windows late. Comparators: fixed, re-tuned every M windows with the same lag, hindsight oracle.

### 6.11 Bandit simulation (`threshold/bandit_sim.py`, `experiments/bandit_report.py`)

Reuses the report's drift stream and reward. Arms are K = 10 evenly spaced thresholds on the score (0.05 to 0.95). **Bandit feedback:** only the played arm's reward is observed, and it becomes usable `max(d, 1)` windows after its window. **Warm start:** the first 5 windows are observed for all arms (the same windows that tune the fixed comparator).

* **Discounted UCB** (Garivier and Moulines, ALT 2011): an observation from window s counts with weight gamma^(t - s); index = mean + c * sqrt(ln(1 + n) / N), with gamma = 0.9, c = 0.5.
* **Sliding-window Thompson sampling [citation needed]:** a Beta(1 + successes, 1 + failures) belief per arm over the last W = 10 windows only; the normalised reward in [0, 1] becomes a Bernoulli trial when revealed; one sample per arm, play the argmax.

Rewards are clipped at +-10 and mapped to [0, 1] for the policies. **All hyperparameters were fixed in `configs/bandit.yaml` before the first run.** Comparators and the oracle come from the existing `ascent.run_policies`, unchanged; 6 seeds, seed 0 is exactly the report's stream.

### 6.12 Severity regression (`models/regress.py`, `experiments/synth_severity.py`)

**Synthetic only.** IBM has no next-window wash volume. The generator makes one week in which each planted ring runs all week, so consecutive **one-day windows** contain the same ring. A candidate at window t is matched to the same ring at t+1 by *account set and planted id*. **Target:** log1p of the ring's own wash volume in window t+1 (the sum of its own `wash_ring` transactions). **Excluded inputs:** the D columns that encode the ring's own current volume or activity (`D_log_volume`, `D_vol_ratio`, `D_vol_cv`, `D_log_count_mean`), so a persistence shortcut is not handed to the model. Ridge, random forest and gradient boosting with fixed hyperparameters; grouped CV by ring identity across 6 seeds, 5 repeats; RMSE, MAE and Spearman with grouped-bootstrap CIs.

### 6.13 Clustering (`models/cluster.py`, `experiments/ibm_cluster.py`)

**Exploratory.** Ward hierarchical clustering (Euclidean) of the *positive* IBM rings on standardised vectors, with k equal to the number of pattern types; ARI (adjusted Rand index) and purity against the pattern type, silhouette, and baselines: random relabelling of the same cluster sizes (1000 permutations) and majority-class purity. The same grouped bootstrap (500 resamples over positive groups) gives CIs; ARI against the *group id* is a check that clusters are not just recovering attempts.

### 6.14 Cap sensitivity (`experiments/cap_sensitivity.py`)

The cap only enters when diagrams are vectorised, so each window's and each ring's persistence is computed once and vectorised under every cap rule. The default `window_max` reproduces the stored T columns exactly (max absolute difference 0.0); the stored classification protocol is rerun with the same folds and bootstrap resamples. B and D columns and labels are unchanged, so B+D rows do not depend on the cap.

### 6.15 Candidate recall (`eval/recall.py`, `experiments/candidate_recall.py`)

Recall asks how many true rings the *candidate generator* returns, separately from classification. IBM: a CYCLE attempt is **complete** in a window if all its transactions fall inside it and **recovered** if a candidate has exactly the attempt's account set (the README measurement); the label-based count (the candidate also carries a positive label) is reported next to it. Synthetic: a planted wash ring is recovered if some candidate is labelled as that ring. Recall needs no features.

### 6.16 Pump-and-dump track (`motif/pump_dump.py`, `experiments/pump_dump_run.py`; synthetic only, no topology)

**Candidates** are *accumulation windows* found from trades alone: a 12-hour band in which at least 4 accounts each buy at least a set volume. **M features** (8): accumulation concentration (share of buy volume in the busiest quarter of the band), **exit synchrony** (spread of the members' first-sell times after the band), share selling within 24 hours, median exit delay, position-size dispersion, price run-up, drawdown, and group size. **B_pd** (3): group size and per-account volume, with no timing or price. Features read only the trade table and the price series (a test scrambles labels and ids and checks nothing changes); they use sells and prices *after* the band, so this is retrospective detection, not early warning. The run reuses the classification harness unchanged (same fixed hyperparameters, grouped stratified CV, grouped bootstrap, paired differences). The generator thresholds were calibrated on candidate counts before any model run, and the configuration was fixed in `configs/pump_dump.yaml` before the run.
