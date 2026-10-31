
## 8. Results

Each subsection gives the **question**, the **setup**, the table(s), **how to read it** and a **one-line honest conclusion**. All tables are copied by the build script from `paper/numbers_pack.md`, which cites the source CSV, config and seeds. PR-AUC cells are `PR-AUC [95% grouped-bootstrap CI]`. Prevalence is stated with every table. CIs that include 0 (for differences) or overlap heavily (for levels) mean "not distinguishable at this sample size".

### 8.1 The optimization report (synthetic, pinned generator)

*Question.* Do the three optimization problems of section 6.10 behave as theory predicts, and how much does adapting the threshold matter? *Setup.* Pinned report generator (dense, chorded, shortcut-filled and hot-wallet counts 0, ambiguous counted as negatives); 12 training and 6 test windows for A and B; a 40-window drift stream for C. Source: `results/gd_metrics.json` (copied unchanged in the pack, section 2).

* **Problem A.** Step 1/L with L = 6.8877; the local-curvature step reaches a gap of 1e-6 in 322 iterations while the others have not by the end. The gradient-descent solution agrees with L-BFGS to 6.5e-7 in the parameters. Test AP of the D classifier is 0.9984 and of D plus the legacy topology columns 0.9969: **topology does not help the report's detector either.**
* **Problem B.** Test AP of the ring-birth detector rises from 0.5675 with equal weights to 0.9997 with the learned alpha (0.0, 0.2067, 0.7933): *learning the weights matters a great deal, because of Proposition 1.* The ring-birth detector is then close to ceiling on this generator.
* **Problem C.** Cumulative reward after warm-up as a share of the hindsight oracle: fixed 0.2367, re-tuned every 5 windows 0.86, online ascent 0.97 (figure below).

![Figure 8.1. Optimization report, Problem C: threshold trajectories and cumulative reward under drift (pinned synthetic stream, one stream, 40 windows). Deterministic, no CI; reward counts TP minus lambda times FP with lambda 1.](numbers_pack/figures/fig5_threshold_drift.png)

*Conclusion.* On this synthetic generator weight learning and threshold adaptation each matter; the optimization report is a synthetic-only demonstration, not evidence about real data.

### 8.2 Pattern-behaviour audit on IBM AML

*Question.* The three scores encode wash-trading behaviour. Do IBM laundering patterns actually look suspicious under them? *Setup.* HI-Small windows 0 and 2 (5 days each) pooled; for every pattern group, the edge-level AUC of that group's pairs against all background pairs for each score (0.5 = chance), and the **tail share**: the share of a group's pairs in the most suspicious 10% tail of the window (for `weight`, the lowest 10% of weights); the baseline for a score with no ties is 0.100. Source: `results/phase3_features/audit/HI_audit_summary.csv`.

<<TABLE 3.5|1>>

*How to read it.* CYCLE and RANDOM pairs are ranked as more suspicious than background by the balance score and the combined weight (AUC about 0.8); the other types are closer to chance, and unattributed laundering pairs are at or below chance for balance. `s_rec` is almost always 0, so its AUC and tail share are tie-driven.

**Imputed timing.** The timing score needs at least 3 payments in a pair. Share of pairs with fewer than 3 payments, whose regularity score is therefore imputed:

<<TABLE 3.5|2>>

![Figure 8.2. HI-Small audit: box plots of the rank-normalised scores and the combined weight by pattern group and background pairs, pooled over the audited windows. Descriptive; no CI, no class prevalence.](numbers_pack/figures/hi_audit_scores.png)

*Conclusion.* The wash-trading weights describe IBM cycles only partly (the balance score does most of the work), and the timing score is mostly an imputed constant for laundering pairs. This is why IBM is used to validate the machinery, not the wash-trading weights.

### 8.3 IBM HI-Small classification (the main real-structure result)

*Question.* Does B+D+T beat B+D? *Setup.* `tau_quantile` 0.05, `max_len` 6, 5-day windows with a 1-day stride; primary variant (ambiguous excluded); grouped CV (5 folds, 5 repeats), fixed hyperparameters, 1000 bootstrap resamples. **Candidates and labels:**

<<TABLE 3.1|1>>
<<TABLE 3.1|2>>

**PR-AUC, primary variant (prevalence 0.0173, 131 positives in 14 positive groups):**

<<TABLE 3.2|1>>

![Figure 8.3. IBM HI-Small, PR-AUC of every feature set and control by model; primary variant, prevalence 0.0173 (dashed line), 14 positive groups. Error bars: 95% grouped-bootstrap CI. Pooled out-of-fold, mean of 5 repeats.](numbers_pack/figures/hi_pr_auc.png)

**Paired differences** (same resamples). First the pre-specified primary comparison, then the secondary, then the exploratory ones:

<<TABLE 3.3|1>>
<<TABLE 3.3|2>>
<<TABLE 3.3|3>>

*How to read it.* **Direction and behaviour matter:** B+D minus B is positive with a CI excluding 0 for all three models. **No detectable topological lift; the CIs cannot rule out a small effect either way:** B+D+T minus B+D is +0.104 [-0.002, +0.171] for logistic regression, with a CI that just includes 0, and about 0 for the other two. The exploratory ablation says the local topology columns help logistic regression (B+D+T_local minus B+D = +0.141, CI above 0) while adding the window-constant T_win columns lowers it again (B+D+T minus B+D+T_local is -0.037, CI below 0 for logistic regression and svm). Both are exploratory, one model each, and not corrected for multiplicity. **Controls:** `T_win` alone scores far above the prevalence on HI (0.121 and 0.130) although it is constant within a window, consistent with a time proxy; D alone matches or beats B+D for two models. `-D_birth` alone scores 0.314 [0.097, 0.487] with no model at all.

**Other variants** (ambiguous counted positive: prevalence 0.0805, 142 groups; cycle-only: prevalence 0.0048, 12 groups; strict account groups: 14 groups):

<<TABLE 3.4|1>>
<<TABLE 3.4|2>>
<<TABLE 3.4|3>>
<<TABLE 3.4|4>>
<<TABLE 3.4|5>>
<<TABLE 3.4|6>>

*Conclusion (HI).* B+D is much better than B; **there is no detectable lift from adding topology to B+D** (primary paired CIs include 0 for all three models, with 14 positive groups). The point estimates are mostly small and positive, so a small effect cannot be excluded, and a small negative one cannot either. This is the central real-structure result.

### 8.4 IBM LI-Small

*Question.* Does the HI pattern replicate on the low-illicit dataset with the same code? *Setup.* identical; **38 positives in 10 groups at prevalence 0.0039**; only 3 of the positives are CYCLE (3 groups), so the cycle-only variant is **unusable** (fewest positive groups on a train or test side is 1), and the primary variant has only 1 on its thinnest side.

<<TABLE 4.1|1>>
<<TABLE 4.1|2>>
<<TABLE 4.2|1>>
<<TABLE 4.3|1>>
<<TABLE 4.3|2>>
<<TABLE 4.3|3>>

Ambiguous counted positive (prevalence 0.0513, 189 groups):

<<TABLE 4.4|1>>
<<TABLE 4.4|2>>

*How to read it.* Scores are far lower than on HI and the intervals are wide. B+D beats B for two of three models, B+D+T versus B+D is again not distinguishable from 0, and `-D_birth` alone is at 0.006, i.e. no better than chance at prevalence 0.0039, **contradicting HI**. `T_win` alone is also at chance here, so the HI time-proxy pattern does not appear on LI.

*Conclusion (LI).* The direction-feature gain partly replicates (two of three models); there is again no detectable topological lift (CIs cannot rule out a small effect either way); the ring-birth detector does not replicate. With 10 positive groups these are weak constraints in both directions.

### 8.5 HI to LI transfer (labelled as transfer, not a time-leak-free test)

*Question.* Does a model trained on one dataset work on the other? *Setup.* Train on all primary-variant candidates of one dataset, test on the other; scaling fitted on the training dataset only; **one model per direction, no repeats**; CI = grouped bootstrap over the TEST groups. Both datasets come from one generator and cover the same dates; 3 accounts are shared and used by no candidate. **The `-D_birth` row is a within-dataset score of a no-model detector and is not a transfer result.**

<<TABLE 4.6|1>>
<<TABLE 4.6|2>>
<<TABLE 4.6|3>>
<<TABLE 4.6|4>>

*Conclusion.* Transfer is weak and asymmetric: HI to LI reaches PR-AUC around 0.1 to 0.2 for B+D at prevalence 0.0039, and in both directions there is no detectable lift of B+D+T over B+D (in several cells it is lower, with the CI excluding 0 for random forest HI to LI and logistic regression LI to HI; CIs cannot rule out a small effect either way for the rest).
