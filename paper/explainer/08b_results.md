
### 8.6 Synthetic data: four conditions, per scenario, hard negatives, dense chords

*Question.* On the generator that plants exactly the cases where topology should matter, does B+D+T beat B+D, overall and per scenario, with and without hard negatives and under both dense-chord settings? *Setup.* Seeds 0 to 5 pooled per condition; conditions are hard negatives off or on times `dense_internal_chords` default or off; candidate quantile tau 0.25, `max_len` 6; stratified group CV (5 repeats), fixed hyperparameters, 1000 bootstrap resamples; groups are prefixed by seed. Cells are pooled PR-AUC [CI] with the mean +- std of the per-seed PR-AUC in brackets. **This is the generator's own distribution; the generator decides the difficulty (section 11.1).**

**Condition hn-off, chords default** (5071 candidates, prevalence 0.0203, 46 positive groups):

<<TABLE 5.1|1>>
<<TABLE 5.1|2>>

**Condition hn-off, chords off:**

<<TABLE 5.2|1>>
<<TABLE 5.2|2>>

**Condition hn-on, chords default:**

<<TABLE 5.3|1>>
<<TABLE 5.3|2>>

**Condition hn-on, chords off:**

<<TABLE 5.4|1>>
<<TABLE 5.4|2>>

![Figure 8.4. Synthetic data, PR-AUC by feature set and model in the four conditions; primary variant; prevalence 0.0203, 0.0267, 0.0412 and 0.0481 (in the order hn-off default, hn-off chords off, hn-on default, hn-on chords off). Error bars: 95% grouped-bootstrap CI over six seeds pooled.](numbers_pack/figures/synthetic_pr_auc.png)

**Per scenario** (positives of one scenario against all negatives, so the prevalence changes per scenario). Condition hn-off, chords default; the three tables are chorded, dense and isolated:

<<TABLE 5.1|4>>
<<TABLE 5.1|5>>
<<TABLE 5.1|6>>

![Figure 8.5. Synthetic data, paired PR-AUC difference B+D+T minus B+D per scenario by model; hard negatives off (left) and on (right); filled = chords default, hollow = chords off. Error bars: 95% grouped-bootstrap CI of the paired difference. Prevalence differs per scenario (isolated 0.0108-0.0265, dense 0.0048-0.0113, chorded 0.0050-0.0118).](numbers_pack/figures/synthetic_lift_per_scenario.png)

**The dense scenario under both chord settings.** Share of the dense positives whose own loop is filled (`T_local_ring_inf` = 0; they still keep their positive label):

<<TABLE 5.5|1>>
<<TABLE 5.5|2>>

**Where the negatives rank** (mean out-of-fold score percentile by candidate kind; 1 = top of the ranking), hn-off default with logistic regression and hn-on default with logistic regression:

<<TABLE 5.6|1>>
<<TABLE 5.6|5>>

**Candidate kinds and ambiguity.** With hard negatives on, the large majority of candidates are ambiguous:

<<TABLE 5.7|1>>

*How to read it.* (i) D carries most of the signal: B+D minus B is positive with a CI above 0 in all four conditions for every model: logistic regression +0.414 to +0.542 (lowest CI lower bound +0.290), random forest +0.383 to +0.395 (lowest +0.244) and SVM +0.368 to +0.614 (lowest +0.184). (ii) B+D+T beats B+D by a paired CI above 0 in 7 of 12 condition-model cells (all three models with hard negatives off; only one cell with hard negatives on), but the gains are small because B+D is already high (B+D scores logistic regression 0.960-0.983, random forest 0.956-0.997, SVM 0.874-0.937 over the conditions). (iii) The gain is largest for isolated and chorded rings with SVM, and the T_local-only set usually does at least as well as the full T set. (iv) In the dense scenario the gain depends on the chord setting: with chords default it is not distinguishable from 0 for most models; with chords off it is positive for logistic regression and random forest. (v) The shortcut-filled negatives still rank high under B+D (their mean percentile is close to the wash rings'), and adding T lowers it (section 8.6 tables, shortcut_filled column): this is the intended topological effect, a ring-looking loop that is filled gets pushed down.

*Conclusion.* **On the synthetic generator topology gives a small lift in most hard-negative-free cells, a smaller or null one when hard negatives are on, and the generator is nearly separable by D alone**, so this does not show that topology helps on real data.

### 8.7 Candidate recall (separate from classification)

*Question.* How many true rings does the candidate step return? *Setup.* `max_len` 6 and 8; tau quantile 0.02 to 0.15 on IBM and 0.05 to 0.5 on synthetic. **Recovered (account set)** = a candidate has exactly the attempt's account set; **recovered (labelled)** adds that the candidate carries a positive CYCLE label; the gap between the columns is the account-set versus label-based difference. IBM, complete CYCLE attempts:

<<TABLE 6.1|1>>

Synthetic planted wash rings (6 seeds), hard negatives off, chords default; then hard negatives on, chords default. A **dagger** marks a cell where the per-SCC enumeration cap (20000 cycles) was hit in at least one window; recall there is limited by the cap, not by tau:

<<TABLE 6.2|1>>
<<TABLE 6.2|3>>

![Figure 8.6. Candidate recall of complete CYCLE attempts against tau_quantile (reachable attempts, at most max_len accounts), HI and LI, days 0-5 and whole data, max_len 6 and 8. Counts, not a sample: no CI, prevalence not applicable. LI days 0-5 has one complete attempt.](numbers_pack/figures/recall_ibm.png)

![Figure 8.7. Recall of planted wash rings against tau_quantile, six seeds, hard negatives off and on. Mean over rings, no CI. With hard negatives on, cells where the enumeration cap is hit are limited by the cap.](numbers_pack/figures/recall_synthetic.png)

*How to read it.* On IBM the cap on recall comes from `max_len`: in HI days 0-5, 15 attempts are complete but only 11 have at most 6 accounts (12 at `max_len` 8), and 8 of the 11 are recovered at tau 0.05; recall rises at tau 0.15. **The account-set count (8) exceeds the label-based count (6): two recovered rings carry a label other than a positive CYCLE under the 80% rule.** On synthetic data without hard negatives recall is 1.00 at every tau; with hard negatives it is low at tau 0.05, perfect at 0.15 to 0.2, and collapses above 0.25 where the enumeration cap is hit.

*Conclusion.* Candidate generation is partial and tau-dependent on IBM, near-perfect on easy synthetic data, and cap-limited on hard-negative synthetic data at large tau.

### 8.8 Cap sensitivity of the T features

*Question.* Do the conclusions depend on how infinite bars are capped? *Setup.* Variants in `configs/cap_sensitivity.yaml`: `window_max` (the **pre-set primary**), `fixed_1.0`, `max_x1.1`, and `ring_local_max` (T_local only). **The `ring_local_max` SVM cells are a post hoc finding in a single model, uncorrected for multiplicity, and are not a result.**

IBM HI-Small (prevalence 0.0173, 14 groups), B+D+T, B+D+T_local and the paired differences:

<<TABLE 6C|1>>
<<TABLE 6C|2>>
<<TABLE 6C|3>>

Synthetic hn-off, chords default (prevalence 0.0203, 46 groups):

<<TABLE 6C|4>>
<<TABLE 6C|5>>
<<TABLE 6C|6>>

*How to read it.* Among `window_max`, `fixed_1.0` and `max_x1.1` the PR-AUC barely moves (largest absolute paired difference to `window_max` 0.024, B+D+T with SVM under `fixed_1.0`, on HI). `ring_local_max`, a post hoc variant, changes several cells, mainly SVM on HI (+0.190 [+0.097, +0.240] for B+D+T), while the logistic regression cell goes the other way; because it was examined after seeing results, it is not adopted.

*Conclusion.* The IBM conclusion (no detectable lift) does not hinge on the cap among the pre-set rules.

### 8.9 The adaptive-threshold benchmark (bandits versus re-tuning)

*Question.* Under drift with delayed labels, do discounted UCB and sliding-window Thompson sampling beat periodic re-tuning at the same lag? *Setup.* The optimization report's drift stream (40 windows; wash timing jitter 0.05 to 0.6 and amount noise 0.01 to 0.16), 6 stream seeds (seed 0 is the report's stream), K = 10 arms, reward TP minus lambda times FP, lambda 0.5/1/2 (primary 1), lag d in {0, 1, 2, 5} (primary 2); hyperparameters fixed before the run. **Seed-0 reproduction of the report:**

<<TABLE 6A|7>>

**Share of the hindsight-oracle cumulative reward, lambda 1** (mean +- std over seeds; `oracle_arms` is the best of the 10 arms per window):

<<TABLE 6A|2>>

**Cumulative reward, lambda 1:**

<<TABLE 6A|3>>

**Bandit minus comparator, paired per seed (negative = the bandit loses):**

<<TABLE 6A|5>>

![Figure 8.8. Phase 6, synthetic drift stream (6 seeds, 40 windows). Left: share of oracle reward by policy and lag at lambda 1, mean +- std over seeds (not a CI). Middle and right: rolling 5-window precision and recall, stream seed 0, d = 2, lambda 1. No bootstrap CI.](numbers_pack/figures/bandit_drift.png)

**Sensitivity (exploratory; the pre-set gamma 0.9 and window 10 are the headline, and gamma 0.95 is NOT adopted):**

<<TABLE 6A|6>>

*How to read it.* At the primary setting both bandits lose to re-tuning every 5 windows (discounted UCB -1.245, 0 of 6 seeds better; Thompson -0.771, 2 of 6 seeds better). The only positive cell is discounted UCB at lag 5 (+0.258, all 6 seeds), and it is not a skill result: on stream seed 0 the discounted UCB at lag 5 never played the last arm within its 35 windows while at lag 2 it did (seen on one seed, not tested further). In this stream arms 0 to 8 (thresholds up to 0.85) earn about the same reward per window (mean 8.22 on seed 0), while the last arm (threshold 0.95, which flags almost everything) earns mean -306.4 per window and as little as -1983; **one exploratory play of that arm costs about as much as the whole oracle total**, but the policies see the reward clipped at -10, so its size is invisible to them. Online gradient ascent, which moves a continuous threshold, reaches 0.981 of the oracle at lambda 1 for every lag.

*Conclusion.* **The bandits lose to periodic re-tuning at the primary setting**; online gradient ascent is the best adaptive method here. This is a simulation on one synthetic drift stream, and it does not say bandits are bad in general (the arm grid and the reward clipping matter).

### 8.10 Severity regression (SYNTHETIC ONLY; descriptive, not a claim)

*Question.* Can features of a ring in window t predict its wash volume in window t+1? *Setup.* The target is log1p of the ring's own wash volume in t+1; one-day windows of one generated week; **the generator decides that rings persist**; the volume-encoding D columns are excluded; 6 seeds, grouped CV by ring. **This is synthetic only, because IBM has no next-window wash volume.** Population:

<<TABLE 6B|1>>

RMSE [95% CI] (log1p units, lower is better):

<<TABLE 6B|2>>

Paired B+D+T minus B+D (RMSE and MAE: negative = B+D+T better):

<<TABLE 6B|5>>

**Diagnostic.** B+D (no volume) beats the mean predictor and a fair persistence fit, which looks like skill, so we added structure-only predictors, **labelled as diagnostics, not features** (the scenario label comes from the generator):

<<TABLE 6B|6>>
<<TABLE 6B|7>>
<<TABLE 6B|8>>

*How to read it.* **`D_length` alone (ring size) has a lower RMSE than B+D for random forest and gradient boosting, with a CI excluding 0, and carries most of the random-forest importance of B+D.** So the model recovers how the generator ties volume to ring size and structure; it is not forecasting. T adds nothing (B+D+T minus B+D is about 0 for random forest and gradient boosting; ridge on T is unstable).

*Conclusion.* Synthetic, generator-decided persistence; recovers generator structure, not forecasting skill; no topological contribution.

### 8.11 Clustering of positive rings (EXPLORATORY)

*Question.* Do rings of different IBM pattern types have distinguishable fingerprints? *Setup.* Ward clustering of the positive rings (131 on HI in 14 groups; 37 on LI in 9 groups, three types with at least 3 groups each) on standardised vectors; k = number of types, plus one silhouette-chosen row. **The grouped-bootstrap percentile CIs are biased and several exclude their own point estimate, so they are not used as CIs: read the point estimates against the baselines.**

<<TABLE 6D|1>>
<<TABLE 6D|2>>
<<TABLE 6D|3>>

![Figure 8.9. Phase 4 (exploratory), HI-Small positive rings (131 rings, 14 groups): 2D PCA of the standardised vectors, coloured by pattern type (top) and by Ward cluster (bottom). Descriptive, no CI.](numbers_pack/figures/cluster_pca_HI.png)

![Figure 8.10. Phase 4 (exploratory), LI-Small positive rings (37 rings, 9 groups): same layout. Descriptive, no CI.](numbers_pack/figures/cluster_pca_LI.png)

*How to read it.* On HI with k = 4, ARI against pattern type is 0.059 (D), 0.073 (T_local landscape), 0.097 (D + T_local landscape) and 0.031 (all features) against a random-labelling ARI of about 0; purity 0.443-0.489 against a random purity of 0.358-0.365 and a majority-class purity of 0.336. **ARI against the group id is larger than against the type (0.169, 0.296, 0.398, 0.303): the clusters track attempts more than pattern types.** On LI (T_local landscape, k = 3, 9 groups with 3 to 5 groups per type) ARI is 0.260 and purity 0.730 against a random purity of 0.536 and a majority purity of 0.459, a small-sample point estimate that cannot be separated from group structure.

*Conclusion (exploratory).* Weak agreement with pattern types, stronger agreement with attempts; no evidence of type recovery.

### 8.12 Comparisons made across the project

**Definition.** *Pre-specified primary* = `B+D+T minus B+D`, one row per model, dataset and variant at overall scope (severity: three metrics; bandits: the re-tuned comparison at lambda 1, lag 2). *Pre-specified secondary* = `B+D minus B`. All else is exploratory or sensitivity. **The primary rows are not independent tests**: they include every variant, and the same candidates and resamples recur across models and variants. **The headline test per dataset is B+D+T minus B+D on the primary variant.**

<<TABLE 6E|1>>

### 8.14 Synthetic pump-and-dump track (minor, exploratory, no topology)

*Question.* Does the harness work on a second pattern type, and do timing features separate planted pump-and-dump groups from a group-size baseline? *Setup.* 6 seeds, 8 pump / 8 organic / 5 news groups per seed, candidates and features as in section 6.16, fixed hyperparameters, grouped CV and bootstrap. **Synthetic only; no topology; the generator author decides separability.** Candidates (ambiguous excluded from the primary variant):

<<TABLE 6F|1>>

Default condition (prevalence 0.390, 47 positive groups):

<<TABLE 6F|2>>
<<TABLE 6F|3>>

Harder exits (a wider spread of the planted exit times; the candidates and groups are identical):

<<TABLE 6F|4>>
<<TABLE 6F|5>>
<<TABLE 6F|6>>

![Figure 8.11. Synthetic pump-and-dump (no topology): PR-AUC of M and B_pd features by model in the default and harder-exit conditions; prevalence 0.390 (dashed line), 47 positive groups. Error bars: 95% grouped-bootstrap CI.](numbers_pack/figures/pump_dump_pr_auc.png)

*How to read it.* M beats B_pd in both conditions: the paired M minus B_pd difference is +0.382 to +0.502 for logistic regression (lowest CI lower bound +0.260), +0.547 to +0.602 for random forest (lowest +0.437) and +0.413 to +0.514 for SVM (lowest +0.299), and widening the exit spread lowers M's PR-AUC (the last table is a difference of two separate point estimates, with no CI). **One feature, `-M_exit_spread_h` alone, reaches 0.906 [0.860, 0.945] in the default condition and 0.643 [0.513, 0.752] with wider exits**: the generator is nearly separable by exit timing. B_pd is weak by construction, because every group is drawn from the same size and volume distributions, so the gain is large by design.

*Conclusion.* The harness runs unchanged on a second pattern type and timing features separate the planted groups; this says nothing about real pump-and-dump data or about topology.

### 8.13 Where the planned figures and tables went

Planned in `paper/paper_content.md` section 7 versus what exists: **F6 (PR curves per feature set) was not made; PR-AUC bars with CIs (Figures 8.3 and 8.4) are used instead.** **F10 is a PCA scatter (Figures 8.9, 8.10), not a dendrogram.**

| ID | Planned | What exists |
|---|---|---|
| F1 | pipeline diagram | ASCII diagram in section 6.1; not drawn |
| F2 | triangles and bars | ASCII sketch in section 4.5; `viz.py` helper exists, not drawn |
| F3 | birth by loop kind, equal vs learned weights | `fig3_birth_distributions.png` (optimization report) |
| F4 | PGD paths on the simplex | `fig2_alpha_simplex.png` (optimization report) |
| F5 | GD optimality gap | `fig1_gd_convergence.png` (optimization report) |
| F6 | PR curves per feature set | **not made**; PR-AUC bars (Fig. 8.3, 8.4) |
| F7 | topological lift per scenario | Fig. 8.5 |
| F8 | candidate recall against tau | Fig. 8.6, 8.7 |
| F9 | threshold trajectories and reward | Fig. 8.1 (report) and Fig. 8.8 (bandits) |
| F10 | ring types vs pattern types | **PCA scatter** (Fig. 8.9, 8.10), not a dendrogram |
| T2 | dataset statistics | pack sections 1 and 3.1 (tables in 8.3) |
| T4 | main classification | tables in 8.3 to 8.6 |
| T5 | ablations | T_local/T_win in 8.3; cap in 8.8; undirected, persistence-image and vectorization ablations **not run** |
| T6 | clustering | 8.11 |
| T7 | severity | 8.10 |
| T8 | adaptive threshold | 8.9 |
| T9 | hyperparameter sensitivity | only the bandit table in 8.9 |
