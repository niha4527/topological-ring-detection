# Product Requirements Document

**Project:** Topological Ring Detection for Market Manipulation
**Type:** Semester AIML project (RVU, SEM3)
**Design source:** `Topological_Ring_Detection_for_Market_Manipulation_v3.md`
**Related:** `architecture.md`, `README.md`, `report/Optimization_Report`, `paper/paper_content.md`
**Last updated:** 2026-10-31

## 1. Summary

Build a classical-ML pipeline that finds, scores, types and tracks coordinated manipulation rings (primarily wash trading) in transaction networks. Each time window becomes a weighted directed graph. Persistent homology of a directed flag complex measures how strongly each loop persists. Rings are localized exactly by cycle enumeration, then classified, clustered by type, and used to predict future severity. An adaptive threshold keeps detection current as tactics drift.

## 2. Problem

1. Density-based fraud detectors (community detection, centrality, GNNs) miss sparse rings in which each account has only 2 or 3 counterparties.
2. Cycle enumeration finds loops but gives a binary answer with no strength, stability or type, and flags legitimate loops (arbitrage, market making).
3. Fixed thresholds decay as manipulators change ring size, timing and decoy rate.
4. Prior TDA work on blockchain graphs flags that global shape changed. It does not isolate rings, type them, or give a continuous severity.

## 3. Goals and non-goals

### Goals
- G1. Localize candidate rings exactly (account-level) in each window.
- G2. Give each ring a continuous, interpretable suspicion score from a behaviour-weighted filtration.
- G3. Test honestly whether topological features add predictive value over baseline and direction-aware features.
- G4. Type rings by clustering and forecast next-window severity without leakage.
- G5. Keep the alarm threshold accurate under simulated drift with delayed labels.
- G6. Validate on synthetic data and IBM AML (pipeline machinery). Ethereum NFT trades (real-world transfer) are NOT done: running queries on Dune is blocked on the free tier; BigQuery access was confirmed and the schemas inspected; the planned leakage-controlled subset is filters 4 and 5 of `nft.wash_trades` only (filters 2 and 3 are cycle-type rules).

### Non-goals
- Deep learning of any kind.
- Spoofing detection (order-book phenomenon; no signal in settled data).
- Live deployment or real delayed-feedback operation (future work).
- Claiming regulator-grade ground truth.
- Pump-and-dump detection is optional stretch, not a commitment.

## 4. Users and use cases

| User | Need |
|---|---|
| Course evaluators | Correct, reproducible pipeline; honest evaluation; report and paper |
| Project team | A tested codebase to extend phase by phase |
| Hypothetical compliance analyst (framing only) | Ranked list of suspicious rings with the accounts and edges named, and a severity estimate |

## 5. Scope of detection

- **In:** closed-cycle wash trading (A ⇄ B, A → B → C → A, up to length 6 by default).
- **Optional:** pump-and-dump coordination as a temporal-motif track.
- **Out:** spoofing.

## 6. Functional requirements

| ID | Requirement | Phase | Status |
|---|---|---|---|
| FR1 | Synthetic generator with planted isolated, dense, chorded wash rings; transitive and shortcut-filled negatives; hard negatives (market makers, arbitrage bots, hot wallets); evasion and drift knobs | 1 | done |
| FR2 | Sliding windows, per-ordered-pair aggregation, scores `s_rec`, `s_bal`, `s_reg`, rank normalization, edge weight `1 − mean`, lower = more suspicious | 1 | done |
| FR3 | Candidate rings via sparsification, SCCs, bounded simple-cycle enumeration, dedup | 2 | done |
| FR4 | Weighted directed flag persistence (dim ≤ 2) with `flagser` and `reference` backends that agree | 2 | done |
| FR5 | Ring-level labels from planted ground truth by the §3.7 rule (X = 80%), ambiguous class counted and reported | 2 | done |
| FR6 | Persistence landscapes of local H1 diagrams; infinite-bar cap at window max filtration; infinite-bar count; persistence-image and scalar ablations | 2 | partial (landscapes, window-max cap, infinite-bar counts, scalar summaries done; persistence-image ablation not done) |
| FR7 | Feature sets B (graph baseline), D (direction/behaviour, includes ring birth), T (topological); one feature table per candidate | 2 | done (B, D, T tables; `features.full_feature_table`) |
| FR8 | Window-level persistence as context features; undirected clique-complex ablation | 2–3 | partial (window-level persistence done; undirected clique-complex ablation not done) |
| FR9 | Classifiers (logistic regression, random forest, RBF-SVM) on B, B+D, B+D+T, T; results overall, per scenario, with and without hard negatives | 3 | done (HI-Small, LI-Small, HI<->LI transfer, synthetic per scenario with and without hard negatives) |
| FR10 | Learn score-combination weights `alpha` on the simplex; online-ascent threshold with delay and comparators | opt. report | done |
| FR11 | IBM AML loader, patterns-file inspection, pattern-behaviour audit of each score, ring-labeling rule, optional AML-specific weight variant | 3 | done (optional AML-specific weight variant not built) |
| FR12 | Ward clustering of rings; ARI, purity, silhouette against IBM pattern types | 4 | done as an exploratory run (HI-Small and LI-Small positives, four vector variants, grouped bootstrap, group-id check; `results/phase4_cluster/summary.md`) |
| FR13 | Severity regression on next-window wash volume (ridge, random forest, gradient boosting); no input leakage | 4 | done, SYNTHETIC ONLY (IBM has no next-window wash volume; `results/phase4_severity/summary.md`) |
| FR14 | NFT loader (Dune or BigQuery), proxy labels, label-provenance audit, results on the subset not flagged by cycle rules | 5 | NOT done: cut; running Dune queries is blocked on the free Dune tier |
| FR15 | Bandit threshold policies (discounted UCB, sliding-window Thompson) in the drift benchmark; compare with fixed, re-tuned (same lag), hindsight oracle | 6 | done (discounted UCB and sliding-window Thompson run on the report's drift stream against fixed, re-tuned, online ascent and oracle; they do not beat the re-tuned comparator at the primary setting, see `results/phase6_bandit/summary.md`) |
| FR16 | Sensitivity sweeps: window length, stride, `tau_s`, `max_len`, infinite-bar cap, label threshold X | 3–7 | partial (candidate-recall sweep over `tau_quantile`, `max_len`; infinite-bar cap sensitivity on HI-Small and synthetic (`results/phase3_features/cap_sensitivity/`); classification at IBM tau 0.05 only; window length, stride and label threshold X sweeps not done) |
| FR17 | Candidate-recall vs `tau_s` curve reported separately from end-to-end precision/recall | 3 | done (`experiments/candidate_recall.py`) |
| FR18 | Pump-and-dump temporal-motif track | 7 | optional |

## 7. Non-functional requirements

- NFR1. Python 3.11 in conda env `ringtda`. Python 3.13 is unsupported (`pyflagser` has no wheels).
- NFR2. Pipeline logic in `src/ringtda/`; notebooks only import and plot.
- NFR3. Every new module has a test on a tiny synthetic case.
- NFR4. Compute stays tractable: persistence on candidate neighbourhoods, complex dimension ≤ 2, per-SCC enumeration time budget with logged fallbacks (never silently dropped).
- NFR5. Reproducible: seeds, window config and generator config stored with every result file.
- NFR6. Compare diagrams with tolerance, never exact equality.
- NFR7. Statistical reporting: mean ± std over ≥ 5 seeds (synthetic) or bootstrap CIs (real data).
- NFR8. Optimization-report outputs (`results/report/`, `results/gd_metrics.json`, `results/figures/`) are never overwritten.

## 8. Evaluation plan

| Task | Metrics |
|---|---|
| Candidate generation | Recall of ground-truth rings across the `tau_s` sweep |
| Classification | Precision, recall, F1, PR-AUC; primary comparison B+D+T vs B+D; secondary B+D vs B |
| Clustering | ARI, purity, silhouette; qualitative on NFT |
| Regression | RMSE, MAE, Spearman vs baseline-feature models |
| Adaptive threshold | Rolling precision/recall, cumulative `TP − λ·FP`, share of oracle reward |

Protocol: temporal split with a one-window gap; grouped CV by ring identity for model selection inside the training period; no performance comparison is a pass/fail gate.

## 9. Milestones and gates

Engineering gates are hard tests. Reporting checkpoints are reported whatever the value.

| Phase | Weeks | Deliverable | Checkpoint | State |
|---|---|---|---|---|
| 1 Synthetic data | 1–2 | Generator, windows, weights | Gate: planted ring edges in lowest-weight decile (easy background) | done |
| 2 Topology pipeline | 3–4 | Candidates, persistence, landscapes, feature table | Gate: planted rings are candidates; cyclic persists, transitive dies, shortcut-filled dies | gates pass; landscapes, cap, feature table open |
| Optimization report | (extra) | GD, PGD on simplex, online ascent | Report numbers reproducible | done (`report/Optimization_Report`) |
| 3 IBM AML | 5–6 | Audit, labels, classifiers, ablations | Report: B+D+T vs B+D with CIs; candidate recall | done |
| 4 Typing + severity | 7 | Clustering, regression | Report: ARI/purity, RMSE with CIs | partial: exploratory clustering, synthetic-only severity |
| 5 NFT transfer | 8–9 | Loader, provenance audit | Report with caveats | NOT done: running Dune queries is blocked on the free tier; BigQuery access confirmed and schemas inspected; planned leakage-controlled subset = filters 4/5 only |
| 6 Adaptive threshold | 10 | Bandit vs fixed vs re-tuned vs oracle | Report | done (ascent and bandits run; the bandits lose to the re-tuned comparator at the primary setting) |
| 7 Stretch + write-up | 11–12 | Pump-and-dump if time; paper | none | pump-and-dump cut; numbers pack done, write-up remaining |

Assumes a roughly 12-week semester; the project is already past the Phase 2 start date, so the remaining phases need compression or scope cuts (see §11).

## 10. Deliverables

1. Tested codebase (`src/ringtda`, `tests/`, `experiments/`).
2. Optimization report (`report/Optimization_Report.docx` and `.pdf`) [done].
3. Paper (content pack at `paper/paper_content.md`; results and inference to fill in).
4. Figures and metrics under `results/`.
5. Notebooks for figures (`notebooks/`).
6. `architecture.md` and this `prd.md`.

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Topological lift is null | Reportable finding; ring birth credited to D; scenario-stratified results (§4.8) |
| IBM AML laundering does not share wash-trading behaviour | Pattern-behaviour audit first; frame IBM as validating machinery; AML-specific weight ablation |
| NFT proxy labels leak cycle rules | Inspect label logic; report on the not-cycle-flagged subset |
| Generator circularity (designer decides hard-negative separability) | Model documented actor behaviour; report with and without; NFT as external check |
| Sparsification caps recall | Report candidate recall across `tau_s` |
| Large SCCs blow up enumeration | Time budget, local re-sparsification, logged fallbacks |
| Schedule slip (late in semester, Phases 3–7 open) | Prioritise Phase 2 finish, classifiers on synthetic, IBM AML; NFT and pump-and-dump were cut; the bandit was run (it loses to the re-tuned comparator at the primary setting) |
| Changing generator breaks report numbers | `report` config pins the old generator; diff re-runs in a scratch folder |
| Adversaries tune to the published scores | Stated in limitations; decoy-rate drift experiments |

## 12. Success criteria

- All engineering gates pass (`pytest` green; currently 204 tests).
- Every reporting checkpoint above has numbers with CIs, including null or negative results.
- The report and paper state limitations from the design (length ≤ 6, synthetic IBM data, heuristic NFT labels, offline-only adaptation, hard negatives modelled).
- Another person can reproduce the headline numbers from the documented commands.

## 13. Open questions

See the list returned in the chat summary; answers should be recorded here once decided.

1. Is the deadline fixed, and which phases are mandatory for the course? Answered: the deadline is Monday night; Phases 1-4 and 6 are done, Phase 5 (NFT) and pump-and-dump are cut.
2. Is the IBM AML data downloaded, and is the patterns file format confirmed?
3. NFT data: Dune or BigQuery access available? Answered: running queries on Dune is blocked on the free tier; BigQuery access is confirmed and the schemas were inspected, but the NFT experiment was not run (planned leakage-controlled subset = filters 4/5 of `nft.wash_trades`).
4. Adaptive threshold: report online gradient ascent, the bandit, or both? Answered: both are reported (the bandits lose to the re-tuned comparator at the primary setting).
5. Deviations (`max_len` 5 vs 6, cap 1.0 vs window max, dedup order) accepted or to be aligned with v3?
6. Paper target venue or format, and who writes which section?
