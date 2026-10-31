# Figure list (caption drafts)

Copies of existing PNGs in `paper/numbers_pack/figures/` (not regenerated). Every PR-AUC CI is a 95% percentile interval of a grouped bootstrap (resampling groups, 1000 resamples) unless stated.

- **hi_pr_auc.png**: IBM HI-Small, PR-AUC of B, B+D, B+D+T, B+D+T_local, B+D+T_win, T and the controls D, T_local, T_win by model; primary variant (ambiguous excluded), prevalence 0.0173, 14 positive groups. Error bars: 95% grouped-bootstrap CI. Pooled out-of-fold, mean of 5 repeats.
- **synthetic_pr_auc.png**: Synthetic data, PR-AUC of B, B+D, B+D+T, B+D+T_local, T by model in the four conditions (hard negatives x dense_internal_chords); primary variant, prevalence 0.0203 (hn-off_chords-default), 0.0267 (hn-off_chords-off), 0.0412 (hn-on_chords-default), 0.0481 (hn-on_chords-off). Error bars: 95% grouped-bootstrap CI over six seeds pooled.
- **synthetic_lift_per_scenario.png**: F7. Synthetic data, paired PR-AUC difference B+D+T minus B+D per scenario (isolated, dense, chorded), by model, hard negatives off (left) and on (right); filled markers = dense_internal_chords default, hollow = off. Error bars: 95% grouped-bootstrap CI of the paired difference (same resamples for both feature sets). **Prevalence differs per scenario** (positives of one scenario against all negatives): isolated 0.0108-0.0265; dense 0.0048-0.0113; chorded 0.0050-0.0118 across the four conditions; positive groups per scenario isolated 39, dense 10, chorded 8 (hn-off, chords default).
- **recall_ibm.png**: Candidate recall of complete CYCLE attempts (reachable ones, at most max_len accounts) against tau_quantile on HI and LI, days 0-5 and whole data, max_len 6 and 8. Counts, not a sample: no CI; prevalence not applicable. LI days 0-5 has one complete attempt.
- **recall_synthetic.png**: Recall of planted wash rings (isolated, dense, chorded) against tau_quantile, six seeds, hard negatives off and on. Mean over rings, no CI. With hard negatives on, the cells where the enumeration cap is hit (daggers in section 6.2, from tau 0.25 or 0.30) are limited by the cap.
- **hi_audit_scores.png**: IBM HI-Small pattern-behaviour audit: box plots of the rank-normalised s_rec, s_bal, s_reg and the combined weight (low = suspicious) by pattern group and background pairs, pooled over the audited windows. Descriptive: no CI, no class prevalence (pair counts in section 3.5).
- **li_audit_scores.png**: IBM LI-Small pattern-behaviour audit: same layout as the HI figure, for LI-Small. Descriptive: no CI, no class prevalence.

- **fig1_gd_convergence.png**: optimization-report figure (pinned report generator, deterministic); no CI and no class prevalence stated here, see the report text.
- **fig2_alpha_simplex.png**: optimization-report figure (pinned report generator, deterministic); no CI and no class prevalence stated here, see the report text.
- **fig3_birth_distributions.png**: optimization-report figure (pinned report generator, deterministic); no CI and no class prevalence stated here, see the report text.
- **fig4_precision_recall.png**: optimization-report figure (pinned report generator, deterministic); no CI and no class prevalence stated here, see the report text.
- **fig5_threshold_drift.png**: optimization-report figure (pinned report generator, deterministic); no CI and no class prevalence stated here, see the report text.

