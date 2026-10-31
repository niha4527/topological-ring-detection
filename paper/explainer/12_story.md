
## 12. The story of how we got here: a decision log

Git dates are from the repository history; almost all commits are dated 2026-10-31 (the initial commit is 2026-10-25), so ordering is by commit, not by day. Each entry says what happened and why it mattered.

| # | Commit / date | Turning point | Why it mattered |
|---|---|---|---|
| 1 | `a5ce89d`, 2026-10-25 | Design (v3), generator, optimization report | **Proposition 1** (equal weights cannot push a multi-party ring below 1/3) was proved and led to *learned* alpha; the learned alpha lifted the ring-birth detector from 0.5675 to 0.9997 AP on the pinned synthetic data. |
| 2 | `a294c98` | Dense scenario found to be mostly filled; `dense_internal_chords` option added | With the default setting only 7 of 24 planted dense rings keep an open loop. Fixing the generator would have changed the report, so the default was kept and an opt-in flag added; dense rows are reported under both settings and a filled planted ring keeps its label. |
| 3 | `1bf0a2a` | De-duplication found to be effectively off | A k-cycle shares at most k-2 edges with any cycle of length at most k, so at 80% overlap nothing is removed until `max_len` 10 (nothing up to 9, verified; an earlier note said "up to 10" and was corrected). Near-duplicate rings all stay candidates, which forces grouped CV. |
| 4 | `8faae6c`, `41b248a` | IBM loader and groups; the **time-ordered split was tried and found unusable** | HI test side had 0 candidates and LI 1, because rings recur. Dropped, reported as unusable, no post-hoc variant; HI<->LI transfer became the out-of-sample check. |
| 5 | `774343e` | B and D features with a leakage guard | Kept labels, planted ids and scenario names out of every feature set by construction and by test. |
| 6 | `7ae561c` | Betweenness made feasible by sampling (k = 100) and caching | Exact betweenness is infeasible on 400k-node windows; sampling makes most nodes 0 (limitation, section 11.1). |
| 7 | `ee2304a` | Pattern-behaviour audit | Showed that the wash-trading scores describe IBM laundering only partly, so IBM validates machinery, not weights. |
| 8 | `7c34e1b` | Classification harness; **the grouped-bootstrap NaN bug** | The first run produced a paired CI [+0.032, +0.172] contradicting its own "share of resamples above 0" of 0.26. Cause: 0/0 at the head of the ranking, NaN resamples dropped, CIs biased upward. Fixed and rerun; CIs got wider, point estimates did not move. |
| 9 | `ce0935e` | T_local / T_win ablation, LI and transfer | The B+D+T lift on HI was confounded with the window-constant `T_win` (a time proxy: 0.121 at prevalence 0.0173). The ablation separated them; LI showed no such proxy. A gating mistake of mine (skipping the LI primary variant) was caught and the run redone. |
| 10 | `3f8ab5b` | Synthetic runs and candidate recall; **recall definition reconciled** | A label-based recovery count (6/6/6/7) disagreed with the README measurement (8/8/8/9); the README definition (account-set match) was adopted and the label-based count kept alongside. |
| 11 | `b5e22b4`, `dd04d9b` | Numbers pack; **two unverifiable README claims corrected** | "58 of 60 folds" became 144 of 150 fold-repeats; the "k = 50 gave 166 non-zero nodes" claim was removed because no stored file supports it. |
| 12 | `a20928c` | Bandit benchmark | The bandits lost to re-tuning. A sensitivity table showed that discount 0.95 does much better; **it was not adopted**, because choosing it after seeing results would be tuning on the test stream. |
| 13 | `abd765b`, `eb8ba3d` | Clustering and severity | Both exploratory; the clustering bootstrap CIs were found to be biased and are not used as CIs. |
| 14 | `30102c3` | Cap sensitivity | `window_max` stayed the pre-set primary; the `ring_local_max` SVM gain is post hoc and not adopted. |
| 15 | `a8fa571` | Severity diagnostic | B+D beat the persistence reference, which looked like skill. A fair persistence fit and structure-only predictors showed that ring size alone does as well: recovery of generator structure. |
| 16 | `92e058c` | Comparison counts redefined | An earlier count of 115 "primary" rows counted per-scenario and cap rows; the definition was fixed (primary = B+D+T minus B+D) and recounted to 55 primary, 47 secondary and 387 exploratory or sensitivity rows. |
| 17 | after the experiments | NFT experiment cut | Dune's free tier blocks running queries; schemas were inspected through BigQuery, and the planned leakage-controlled subset (filters 4 and 5) was never run. |
| 18 | `11f5c53` | Optional pump-and-dump track (synthetic, no topology) | Dry runs on candidate counts (not model results) showed overlapping time slots and a heavy background tail, which were fixed in the generator and the config before any model run. Timing features separated the planted groups (a single exit-timing feature scores 0.906 in the default condition), so the track is a minor result; the pack's comparison count rose to 495 rows (393 exploratory or sensitivity, primary unchanged at 55). |

**A pattern in the log.** Every entry where a number looked better than it should (the NaN bug, the persistence reference, the T_win proxy, gamma 0.95, `ring_local_max`) was resolved by checking for an artefact and not by adopting the better-looking number.
