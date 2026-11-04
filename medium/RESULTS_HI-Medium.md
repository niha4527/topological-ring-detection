# HI-Medium results (3 Nov 2026)

Status: **extension run, not in the paper, the numbers pack or the slides.** Everything lives in `medium/`; nothing in `results/`, `paper/` or `experiments/` was touched.
SVM finished (section 3b).

Setup is identical to the HI-Small study: 5-day windows with a 1-day stride, tau quantile 0.05, ring length at most 6, same B / D / T feature code,
same ring-labelling rule (at least 80% of edges from one laundering attempt), ambiguous candidates excluded (primary variant), fixed hyperparameters
(`configs/classify.yaml`), stratified group 5-fold cross-validation with 5 repeats, pooled out-of-fold PR-AUC, grouped bootstrap (1,000 resamples) with 95%
percentile intervals, paired differences on the same resamples.
Scripts: `medium/build_features.py`, `medium/make_labels.py`, `medium/classify.py`. Raw outputs: `medium/out/classify/HI-Medium_tau0.05/{lr_full,rf,svm}/`.

## 1. Data compared with HI-Small

| | HI-Small | HI-Medium |
|---|---|---|
| Transactions (self-loops dropped) | 4,487,133 | 29,336,378 |
| Windows | 13 | 23 (last 2 have no candidates) |
| Candidates | 8,074 | 56,902 |
| Primary rows (ambiguous excluded) | 7,555 | 52,125 |
| Positive rings | 131 | 852 |
| **Positive groups** | **14** | **87** (73 under the stricter account-level grouping) |
| Prevalence (primary) | 0.0173 | 0.0163 |
| Smallest positive-group count on a train or test side | below 5 in 144 of 150 fold-repeats on Small | 17 |
| Positives by type | mixed | GATHER-SCATTER 262, SCATTER-GATHER 242, STACK 196, CYCLE 152 |

The candidate ids and account sequences of the 56,902 labelled candidates were checked against the feature files and match exactly.

## 2. Logistic regression (full protocol, 241 s)

PR-AUC [95% grouped-bootstrap CI]; prevalence 0.0163.

| Feature set | PR-AUC |
|---|---|
| B | 0.091 [0.060, 0.134] |
| D | 0.644 [0.533, 0.734] |
| B+D | 0.702 [0.608, 0.782] |
| B+D+T | 0.810 [0.737, 0.864] |
| B+D+T_local | 0.784 [0.706, 0.846] |
| B+D+T_win | 0.740 [0.651, 0.810] |
| T | 0.556 [0.456, 0.643] |
| T_local | 0.519 [0.416, 0.609] |
| T_win | 0.071 [0.044, 0.104] |
| -D_birth alone (no model) | 0.279 [0.198, 0.356] |

Paired differences (same resamples):

| Comparison | Difference | 95% CI |
|---|---|---|
| **B+D+T minus B+D (primary)** | **+0.108** | **[+0.049, +0.174]** |
| B+D minus B (secondary) | +0.611 | [+0.522, +0.677] |
| B+D+T_local minus B+D | +0.081 | [+0.022, +0.147] |
| B+D+T_win minus B+D | +0.037 | [+0.022, +0.058] |
| B+D+T minus B+D+T_local | +0.026 | [+0.008, +0.044] |

## 3. Random forest (full protocol, 338 s)

| Feature set | PR-AUC |
|---|---|
| B | 0.422 [0.303, 0.538] |
| B+D | 0.897 [0.843, 0.934] |
| B+D+T | 0.882 [0.823, 0.924] |
| B+D+T_local | 0.893 [0.837, 0.932] |

| Comparison | Difference | 95% CI |
|---|---|---|
| **B+D+T minus B+D (primary)** | **-0.014** | **[-0.036, +0.001]** |
| B+D minus B (secondary) | +0.475 | [+0.379, +0.575] |
| B+D+T_local minus B+D | -0.004 | [-0.020, +0.010] |
| B+D+T minus B+D+T_local | -0.011 | [-0.018, -0.006] |

## 3b. SVM (RBF; B+D and B+D+T only, 719 s)

| Feature set | PR-AUC |
|---|---|
| B+D | 0.659 [0.567, 0.738] |
| B+D+T | 0.709 [0.621, 0.782] |

| Comparison | Difference | 95% CI |
|---|---|---|
| **B+D+T minus B+D (primary)** | **+0.050** | **[-0.004, +0.107]** (share of resamples above 0: 0.96) |

## 4. HI-Small versus HI-Medium, primary comparison (B+D+T minus B+D)

| Model | HI-Small (14 groups) | HI-Medium (87 groups) |
|---|---|---|
| Logistic regression | +0.104 [-0.002, +0.171] | **+0.108 [+0.049, +0.174]** |
| Random forest | +0.025 [-0.050, +0.102] | **-0.014 [-0.036, +0.001]** |
| SVM | +0.003 [-0.111, +0.077] | +0.050 [-0.004, +0.107] |

Interval width fell as projected: logistic regression 0.173 to 0.125 (a smaller drop than the 0.065 I estimated), random forest 0.152 to 0.037.

## 5. What it shows

1. **The logistic-regression gain is real in this data.** The point estimate hardly moved (+0.104 to +0.108), and with 87 groups the interval now excludes 0.
   The small-data result was a low-power one, not a null.
2. **The random forest shows no gain.** The interval is tight, about -0.036 to +0.001, so a gain of any useful size is excluded for this model.
3. **The result depends on the model.** The random forest with B+D (0.897) beats logistic regression with B+D+T (0.810) by a wide margin, and the RBF SVM
   (B+D 0.659, B+D+T 0.709) sits in between. The strongest model gets nothing from topology. One reading is that the forest already captures non-linear structure
   that the topology features give the weaker models. The SVM is awkward for that reading: its kernel is non-linear, yet it shows a borderline gain
   (+0.050, interval [-0.004, +0.107]). The reading is a hypothesis and has not been tested (see the plan in the chat of 3 Nov, five experiments).
4. **Behaviour features still carry the signal.** B+D minus B is +0.611 (LR) and +0.475 (RF), both far from 0.
5. **Time proxy.** T_win alone scores 0.071 here (against 0.121 on Small), about 4 times prevalence. B+D+T_win still beats B+D by +0.037 for logistic
   regression, so part of the whole-window set carries something beyond a clock, or still acts as a weak time proxy. The per-ring set T_local gives +0.081 on its own,
   so most of the logistic-regression gain comes from the per-ring features.
6. **Ring birth alone** scores 0.279 [0.198, 0.356], close to its HI-Small value (0.314), so that signal replicates across HI sizes.

## 6. Caveats

- One dataset and one generator. HI-Medium comes from the same simulator as HI-Small, so this is a larger sample, not an independent replication, and it is still
  synthetic and not real wash trading.
- The SVM was run on B+D and B+D+T only (not B, controls or ablations), because it is slow on 52k rows.
- The primary comparison was fixed in advance; the T_local, T_win and B+D+T_local rows are exploratory ablations, uncorrected for multiplicity.
- 87 groups is much better than 14, but the folds still hold only 10 to 25 positive groups, so the intervals are not tight for every cell.
- No LI-Medium run yet, and no cycle-only or strict-group variants.

## 7. How to describe it honestly

"On the larger HI-Medium data, with 87 positive groups instead of 14, topology gives a detectable gain for logistic regression (+0.108, interval above 0) and no
gain for the random forest, which is the stronger model. So topology helps a linear model but does not improve the best model we tested."

## 8. Timing (for the record)

| Job | Time |
|---|---|
| Feature build, 23 windows | about 5 h 20 min including a 2 h sleep (about 3 h 20 min of real compute) |
| Labels and groups | 8 min |
| Logistic regression, 9 feature sets, 5x5 CV, 1,000 bootstraps | 4 min |
| Random forest, 4 feature sets | 6 min |
| SVM, 2 feature sets | 12 min |

## 9. Time-aware features (exploratory, 3 Nov 2026)

Script `medium/temporal.py`; raw outputs `medium/out/temporal/{lr_hgb,rf,tp_only,wid_control}/`. Idea: a ring recurs in consecutive overlapping windows, so add
**causal** features built only from earlier windows (how many earlier windows it appeared in, its streak, its previous and lowest birth, the change in birth, volume,
balance and regularity). History missing for a first appearance is filled with the current value (neutral). 65.1% of candidates and 743 of 852 positive rings have a
history. No labels, no window index and no absolute time enter the features. Gradient boosting is scikit-learn's `HistGradientBoostingClassifier` with defaults and a fixed
seed (no search). Same grouped CV and bootstrap as above. Not pre-specified, so treat every row as exploratory. No LSTM was run (the project is classical ML only).

PR-AUC [95% CI]; prevalence 0.0163, 87 positive groups.

| Features | Logistic regression | Random forest | Gradient boosting |
|---|---|---|---|
| B+D | 0.702 [0.608, 0.782] | 0.897 [0.843, 0.934] | 0.900 [0.849, 0.935] |
| B+D+TP (history added) | 0.738 [0.647, 0.815] | 0.910 [0.860, 0.944] | 0.911 [0.863, 0.943] |
| B+D+T | 0.810 [0.737, 0.864] | 0.882 [0.823, 0.924] | 0.903 [0.851, 0.938] |
| B+D+T+TP | 0.824 [0.751, 0.880] | 0.900 [0.845, 0.938] | 0.917 [0.871, 0.949] |

Paired differences (same resamples):

| Comparison | Logistic regression | Random forest | Gradient boosting |
|---|---|---|---|
| B+D+TP minus B+D | +0.036 [+0.016, +0.058] | +0.013 [+0.005, +0.023] | +0.011 [+0.001, +0.024] |
| B+D+T+TP minus B+D+T | +0.014 [+0.005, +0.026] | +0.017 [+0.010, +0.029] | +0.014 [+0.004, +0.027] |
| B+D+T minus B+D (no history) | +0.108 [+0.049, +0.174] | -0.014 [-0.036, +0.001] | +0.003 [-0.006, +0.011] |

Controls: the history features alone score 0.436 [0.320, 0.545] (LR) and 0.519 [0.400, 0.623] (gradient boosting), while the window index alone scores 0.068 and 0.062, so the
history features are not just a clock.

What it shows:
1. History helps a little and consistently: every model gains, and every interval excludes 0. The gain is large for logistic regression (+0.036) and about +0.01 for
   the tree ensembles. The best score overall is gradient boosting on B+D+T+TP, 0.917 [0.871, 0.949], against 0.897 for the earlier best (random forest on B+D). That
   difference between two models was not tested with a paired interval, so it is not a claim.
2. A second tree ensemble (gradient boosting) shows no topology gain either (+0.003 [-0.006, +0.011]), like the random forest and unlike logistic regression. That fits the
   earlier reading that tree ensembles already capture what topology adds to a linear model, though it is still a hypothesis and not tested directly.
3. Caveats: exploratory and uncorrected for multiplicity; the gains for tree models are small (about 0.01); consecutive windows share most of their transactions, so a ring's past
   is strongly related to its present; synthetic data; no tuning of the gradient boosting.
