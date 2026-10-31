
## 7. Evaluation protocol: why it looks the way it does

**Why random splits leak.** Rings recur across overlapping windows, and with dedup effectively off a ring and its slight variants are all candidates. A random split would put one copy in training and a near-identical copy in testing, and the classifier would memorise rings instead of learning to detect them. All our primary evaluations therefore use **grouped** cross-validation, with groups defined by shared edges and attempt ids (section 6.8).

**Why a time-ordered split was dropped.** A split that trains on early rings and tests on late ones was tried as specified and was **unusable**: HI-Small left 0 test candidates and LI-Small 1, because rings recur and almost every window mixes early and late groups. The code is kept; it is reported as unusable, with no post-hoc variant. The out-of-sample check is HI to LI transfer instead, labelled as transfer and not as time-leak-free (both come from one generator and cover the same dates; the datasets share 3 accounts, used by no candidate).

**Why PR-AUC.** The positives are rare (prevalence between 0.0039 and 0.0481 across our datasets), where ROC-AUC looks flattering because true negatives dominate. Precision-recall area focuses on the positives; a random scorer scores the prevalence, so we print prevalence with every number.

**Why intervals are wide, and what the effective sample size is.** The effective sample size is the number of **positive groups**, not positive candidates: 14 on HI (131 positive candidates) and 10 on LI (38 candidates); in the split report 144 of 150 fold-repeats (three tau values, HI and LI, 5 folds, 5 repeats) hold fewer than 5 positive groups, and 50 of 50 at tau 0.05. We report the intervals as they are and do not tune anything to narrow them.

**All choices are made inside training folds.** Scaling is part of the pipeline; hyperparameters are fixed in config files and never searched; no threshold is chosen on test data.

**Pre-specified primary versus exploratory.** *Pre-specified primary:* `B+D+T minus B+D`, one row per model, dataset and variant (the v3 proposal's design); *pre-specified secondary:* `B+D minus B`. Everything else, notably the T_local and T_win ablations (added after the first HI run), per-scenario rows, cap-sensitivity rows, diagnostics and the post hoc `ring_local_max` finding, is **exploratory or sensitivity**.

**Multiplicity note.** Across the project there are 495 paired-difference rows; 55 are pre-specified primary, 47 pre-specified secondary and 393 exploratory or sensitivity. No multiplicity correction is applied anywhere, the rows within a file are not independent, and the primary count includes every variant (e.g. ambiguous-positive, strict-groups), so it counts rows, not independent tests; the headline test per dataset is the primary variant. The full table is in the pack, section 6E, reproduced in section 8.12.

**Statistical reporting rule.** Synthetic results also carry mean +- std of per-seed PR-AUC over 6 seeds (the v3 requirement of at least 5 seeds); real-data results use bootstrap CIs.
