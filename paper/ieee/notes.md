# notes.md: number-traceability file

Sources: `paper/project_explainer.md` (EX) and `paper/numbers_pack.md` (PK, section = pack section). Every number below is copied from those. PR-AUC CIs are 95% percentile intervals of a grouped bootstrap over groups (1000 resamples, seed 12345). LR = logistic regression, RF = random forest, SVM = RBF-SVM (fixed hyperparameters, `configs/classify.yaml`). Candidate-level counts are not the sample size; positive groups are.

## 1. Problem
Wash trading and laundering cycles appear as directed loops among few accounts, not as odd single transactions (EX 2). Gaps: density methods miss sparse loops; cycle enumeration is binary; fixed thresholds decay under drift (EX 2.3). Scope: closed cycles in scope; pump-and-dump only as synthetic no-topology track; spoofing out (EX 2.4).
Research question: do topological features T add predictive value over graph baseline B plus direction/behaviour D (which includes ring birth)? Pre-specified primary comparison: B+D+T minus B+D; secondary B+D minus B (EX 4.9, 7).

## 2. Method (EX 4, 6; architecture.md)
- Windows: 5 days, stride 1 day (IBM); one edge per ordered pair (volume, count, gap CV).
- Scores s_rec, s_bal, s_reg in [0,1], rank-normalised (ties lowest); weight w = 1 - mean; low = suspicious; sublevel filtration (EX 6.2).
- Proposition 1: with equal weights a multi-party ring with no reciprocal pair has weight >= 1/3 (EX 6.2). Learned alpha (0.0, 0.2067, 0.7933) raises ring-birth AP 0.5675 -> 0.9997 on pinned synthetic data (EX 8.1; PK 2).
- Candidates: sparsify at quantile tau (IBM 0.05, synthetic 0.25), SCCs, simple cycles up to max_len 6, cap 20000 cycles/SCC. Dedup removes nothing for max_len <= 9 (EX 6.3).
- Persistence: directed flag complex, dim 2, flagser and reference backends agree within tolerance (EX 6.4). Localisation is combinatorial because flagser returns no representative cycles (EX 4.7).
- Features: B 12 cols (full window graph, betweenness k=100 sampled on IBM), D 17 cols (incl. `D_birth`), T 311 cols = T_local 156 + T_win 155; 3 landscapes x 50 points on [0,1.1]; cap = window max; `T_local_ring_inf` exact Z/2 test (EX 6.5, 6.6).
- Labels: ring positive if >= 80% of its edges are backed by one structure/attempt; ambiguous reported excluded (primary) and counted positive (EX 6.7).
- Evaluation: groups = union-find on shared edges merged with attempt id; stratified group 5-fold, 5 repeats; pooled out-of-fold PR-AUC; grouped-bootstrap CIs; paired differences on the same resamples; fixed hyperparameters, scaler fitted in training folds (EX 6.8, 6.9).

## 3. Data
- IBM AMLworld HI-Small, LI-Small (synthetic AML data; Altman et al.). HI: 4,487,133 tx after dropping self-loops, 13 windows, 8074 candidates, 131 positives, 14 positive groups; LI: 6,119,572 tx, 12 windows, 10206 candidates, 38 positives, 10 positive groups (PK 1). HI primary: 7555 candidates, prevalence 0.0173; LI primary: 9720 candidates, prevalence 0.0039 (EX 8.3, 8.4; PK 3.1, 4.1).
- Self-loops dropped (591,212 of 5,078,345 = 11.6% on HI); currencies mixed; 1,968 of 5,177 laundering rows in no attempt (README; EX 5.2).
- Synthetic generator: isolated/dense/chorded wash rings, transitive decoys, shortcut-filled pseudo-rings, hard negatives (arbitrage, market makers, hot wallets); 6 seeds; four conditions (hard negatives on/off x dense_internal_chords default/off) (EX 5.1, 8.6). Prevalence 0.0203, 0.0267, 0.0412, 0.0481 (hn-off default, hn-off chords-off, hn-on default, hn-on chords-off) (figure_list).
- Dense scenario: with default chords, 7 of 24 planted dense rings keep an open loop; chords off 18 of 24 (EX 5.1; PK 5.5).
- NFT: not run (Dune free tier blocks running queries) (EX 5.3).

## 4. Results (number, source)
**R1 D vs B (secondary, pre-specified).** HI primary, B+D minus B: LR +0.399 [+0.139, +0.571]; RF +0.572 [+0.276, +0.720]; SVM +0.233 [+0.057, +0.379] (EX 8.3; PK 3.3). LI: LR +0.072 [+0.020, +0.173]; RF +0.132 [+0.045, +0.283]; SVM +0.105 [-0.060, +0.276] (EX 8.4; PK 4.3). Synthetic: CI above 0 in all four conditions, all models; LR +0.414 to +0.542 (EX 8.6, point 'i').
**R2 Primary, B+D+T minus B+D.**
- HI (14 groups, prev 0.0173): LR +0.104 [-0.002, +0.171]; RF +0.025 [-0.050, +0.102]; SVM +0.003 [-0.111, +0.077] (EX 8.3; PK 3.3).
- LI (10 groups, prev 0.0039): LR +0.033 [-0.007, +0.097]; RF -0.012 [-0.119, +0.077]; SVM -0.044 [-0.181, +0.091] (EX 8.4; PK 4.3).
- Synthetic, CI above 0 in 7 of 12 condition-model cells (EX 8.6; PK 8). hn-off default: LR +0.018 [+0.002, +0.057]; RF +0.038 [+0.010, +0.095]; SVM +0.077 [+0.019, +0.175] (PK 5.1). hn-off chords-off: +0.030 [+0.010, +0.074] / +0.043 [+0.011, +0.110] / +0.096 [+0.029, +0.210] (PK 5.2). hn-on default: +0.005 [-0.009, +0.021] / -0.003 [-0.012, +0.004] / +0.038 [-0.004, +0.080] (PK 5.3). hn-on chords-off: +0.014 [-0.006, +0.039] / +0.003 [-0.004, +0.012] / +0.052 [+0.001, +0.109] (PK 5.4).
- B+D levels on synthetic: LR 0.960-0.983, RF 0.956-0.997, SVM 0.874-0.937 (EX 1, 8.6). D alone near-separable (EX 11.1 item 5).
**R3 Exploratory ablations (added after the first HI run, uncorrected).** HI B+D+T_local minus B+D: LR +0.141 [+0.031, +0.224]; RF +0.041 [-0.037, +0.106]; SVM +0.037 [-0.081, +0.126]. B+D+T_win minus B+D: LR +0.034 [-0.005, +0.094]; RF +0.007 [-0.025, +0.059]; SVM +0.088 [+0.006, +0.176] (EX 8.3; PK 3.3).
**R4 Per scenario (exploratory), synthetic.** Isolated: CI above 0 in 8 of 12 cells; dense 2 of 12 (chords off, hn off: LR +0.130 [+0.038, +0.330], RF +0.123 [+0.029, +0.329]); chorded 6 of 12 (SVM 4 of 4, RF 2 of 4, LR 0 of 4) (EX 9.1 G3; PK 5). Not where v3 predicted. Cause untested. Smallest scenario group count 8 (EX 11.1 item 11). Positive groups per scenario: isolated 39, dense 10, chorded 8 (figure_list).
**R5 Transfer HI<->LI (labelled as transfer, one model per direction).** HI->LI B+D+T minus B+D: LR -0.017 [-0.147, +0.092]; RF -0.066 [-0.177, -0.012]; SVM -0.050 [-0.142, +0.042]. LI->HI: LR -0.276 [-0.550, -0.036]; RF -0.019 [-0.079, +0.046]; SVM -0.056 [-0.123, +0.025] (EX 8.5; PK 4.6). B+D HI->LI: 0.133 / 0.149 / 0.124.
**R6 Ring-birth signal.** `-D_birth` alone: HI 0.314 [0.097, 0.487] (prev 0.0173); LI 0.006 (prev 0.0039) (EX 8.3, 8.4; PK 7 item 3). No transfer.
**R7 T_win as time proxy (exploratory, untested directly).** T_win alone on HI: LR 0.121 [0.056, 0.198], RF 0.130 [0.061, 0.202] at prev 0.0173; LI: 0.006 / 0.004 at prev 0.0039 (EX 8.3, 11.1 item 4; PK 7).
**R8 Cap sensitivity (exploratory).** Among window_max, fixed_1.0, max_x1.1 largest absolute paired difference to window_max 0.024 (SVM, B+D+T, fixed_1.0, HI). `ring_local_max` is post hoc: SVM B+D+T +0.190 [+0.097, +0.240] on HI; not adopted (EX 8.8; PK 6C).
**R9 Bandits (synthetic drift stream, 40 windows, 6 seeds, lambda 1, lag 2).** Share of hindsight oracle: fixed 0.298 +- 0.059; retune_M5 0.861 +- 0.020; ascent 0.981 +- 0.006; ducb -0.383 +- 1.067; swts 0.091 +- 0.753. Bandit minus retune_M5 (paired by seed): ducb -1.245 +- 1.066, 0/6 seeds better; swts -0.771 +- 0.744, 2/6 (EX 8.9; PK 6A). Only positive cell: ducb at d=5 +0.258 (6/6), not a skill result. gamma 0.95 sensitivity not adopted. Seed-0 reproduces report (fixed 71.0, retune 258.0, ascent 291.0, oracle 300.0).
**R10 Optimization report (pinned generator).** Test AP of ring-birth detector 0.5675 -> 0.9997; D classifier 0.9984, D+legacy topology 0.9969; GD 322 iterations with local-curvature step; online ascent 0.97 of oracle in the report vs fixed 0.2367, retune 0.86 (EX 8.1; PK 2).
**R11 Severity (synthetic only, exploratory).** rf RMSE: B+D 0.141 [0.134, 0.149], D_length alone 0.132 [0.125, 0.139], fair persistence 0.221 [0.203, 0.237]; rf importance of `D_length` 0.796; B+D+T minus B+D rf RMSE -0.001 [-0.004, +0.002] (EX 8.10; PK 6B). Reads as recovering generator structure, not forecasting.
**R12 Clustering (exploratory).** HI k=4 D+T_local landscape: ARI vs type 0.097 (random 0.000), ARI vs group id 0.398, purity 0.489 (random 0.365, majority 0.336). LI k=3 T_local landscape: ARI 0.260, purity 0.730 (random 0.536, majority 0.459), 9 groups. Bootstrap CIs biased, not used (EX 8.11; PK 6D).
**R13 Candidate recall.** HI days 0-5, tau 0.05, max_len 6: 15 complete CYCLE attempts, 11 reachable, 8 recovered by account set, 6 also labelled positive. HI whole data tau 0.05 max_len 6: 54 complete, 34 reachable, 19 recovered (15 labelled). LI days 0-5: 1 complete attempt. Synthetic recall 1.00 at every tau with hard negatives off; with hard negatives on 0.52/0.21/0.42 (isolated/dense/chorded) at tau 0.05, 1.00 at 0.15-0.25, cap-limited above (EX 8.7; PK 6).
**R14 Audit.** HI CYCLE pairs: edge-level AUC 0.794 (weight), 0.795 (s_bal); 0.899 of CYCLE pairs have timing imputed vs 0.467 background (EX 8.2; PK 3.5).
**R15 Pump-and-dump (synthetic, no topology, exploratory).** Default: M 0.890 / 0.976 / 0.928 vs B_pd 0.388 / 0.375 / 0.414; M minus B_pd +0.502 / +0.602 / +0.514 (all CIs above 0); prevalence 0.390, 47 positive groups. `-M_exit_spread_h` alone 0.906 [0.860, 0.945]; wider exits 0.643 [0.513, 0.752] (EX 8.14; PK 6F).
**R16 Multiplicity.** 495 paired rows: 55 primary, 47 secondary, 393 exploratory/sensitivity; no correction; primary rows are not independent (EX 7, 8.12; PK 6E).

## 5. Limitations (EX 11.1, PK 7)
Few positive groups (14, 10) so wide CIs; LI has 3 CYCLE positives (cycle-only unusable); HI-vs-LI contradiction on ring birth; T_win time proxy; synthetic generator near-separable by D; 0.685-0.966 of laundering pairs have imputed timing; betweenness sampled (0.1594 non-zero on HI window 0); recall capped by max_len; hard-negative ambiguity (0.963 and 0.972 of candidates ambiguous with hard negatives on); transfer one model per direction and same generator; SVM untuned; currencies mixed; IBM is synthetic AML data not market data; single generator; networkx bounded-cycle enumeration has a reported counterexample for the underlying algorithm, untested here. Not done: NFT (Dune free tier blocks running queries), real-data pump-and-dump, undirected-complex and persistence-image ablations, per-score filtrations, window/stride/X/tau/max_len classification sweeps, GNN comparison.

## 6. Deviations from v3 (EX 11.2)
Dedup order shortest-first then birth; max_len 5 in report vs 6; window-max cap vs legacy fixed 1.0; IBM windows 5d/1d; time-ordered split unusable (HI 0 test candidates, LI 1) so dropped, HI<->LI transfer used; timing imputed for pairs with <3 payments; cycle-only subset defined by us; FPR at 80% recall replaced by rank percentiles.

## 7. Unclear [ASK]
- [ASK] Pack 1 says label rule X, synthetic generator parameters and synthetic tau 0.25 are not stored with the results; paper can only cite the current config files.
- [ASK] EX 13 says 204 tests; pytest collects 210. Paper will say 210 only if a test count is mentioned at all.
- [ASK] Per-seed CIs: synthetic cells pool six seeds; the mean +- std over seeds is in PK 5 and not used in main tables.
- [ASK] Pack 1 gives HI 8074 candidates / 13 windows; EX 8.3 primary table gives 7555 (ambiguous excluded). Paper uses 7555 for primary prevalence and states ambiguous excluded.

## 8. Additional sourced numbers used in the paper text (added at Stage 4)
- 35 windows: on stream seed 0 the discounted UCB at lag 5 never played the last arm within its 35 windows (EX 8.9).
- 150 fold-repeats: 144 of 150 hold fewer than 5 positive groups (EX 7; `results/phase3_features/ibm_split_counts.csv`).
- 159 million USD, more than 30% of tokens, structures of one or two accounts: Victor and Weintraud, PDF p.1-2 (not from the pack).
- 11.5 s for 1,113 candidates (T table) and 82 s (B step, HI window 0): README measurements (HI-Small, 5-day window).
- 705,907: LI accounts, of which 3 are shared with HI (README; CLAUDE.md).
- Ambiguous-positive variants: HI prevalence 0.0805, 142 groups, B+D+T minus B+D +0.027 [-0.015, +0.086] / +0.003 [-0.056, +0.032] / +0.176 [-0.033, +0.297]; LI prevalence 0.0513, 189 groups, -0.001 [-0.014, +0.008] / +0.003 [-0.003, +0.008] / +0.007 [-0.009, +0.028] (EX 8.3, 8.4; PK 3.4, 4.4).
- LI B+D+T_local minus B+D, LR +0.002 [-0.033, +0.042] (EX 8.4; PK 4.3).
- Synthetic condition candidate counts: 5071/103/46 groups, 3860/103/46, 2477/102/55, 1933/93/52 (PK 5.1-5.4).
