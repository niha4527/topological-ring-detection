# Outline (about 8 pages, two-column IEEEtran conference)

**Working title.** Does Topology Help Ring Detection? Persistent Homology of Directed Transaction Graphs Against Direction and Behaviour Features

**Contributions (honest).**
1. A behaviour-weighted filtration and a directed-flag-complex pipeline for candidate rings (exact cycle localisation, two agreeing persistence backends, exact ring-survival test). We apply a directed flag complex to ring candidates; no priority is claimed.
2. A pre-specified comparison B+D+T versus B+D under group-aware CV: on IBM AML data (itself synthetic) no detectable topological lift (CIs cannot rule out a small effect either way; HI 14 and LI 10 positive groups), while direction and behaviour features (D, incl. ring birth) carry the detection signal.
3. On a near-separable synthetic generator a small lift appears in 7 of 12 cells, not concentrated where the design predicted, and it does not carry to IBM; the ring-birth signal does not transfer HI to LI.
4. Documented negative and exploratory results: bandits lose to periodic re-tuning (online gradient ascent 0.981 of oracle), severity recovers generator structure, clustering weak, cap choice immaterial.

## Section plan
| Section | Content | Sources | Pack / explainer results |
|---|---|---|---|
| Abstract, Index Terms | question, method, findings with numbers, NFT not done | notes.md | R1, R2, R6, R9 |
| I Introduction | wash trading as loop; two gaps; RQ; contributions; roadmap | victor2021wash, tosic2025beyond, sirolly2025network, altman2023realistic | EX 2 |
| II Related Work | synthesis plan below (4 paragraphs + gap) | all PDFs | EX 3, 10 |
| III Method | windows, weights, Prop. 1, candidates, directed flag complex, B/D/T, labels, groups, classifiers, threshold policies | lutgehetmann2020flagser, govc2021tournaplex, bubenik2015landscapes, johnson1975circuits, gupta2021bounded, garivier2011switching, trovo2020swts | EX 4, 6; Fig 1 (pipeline), Fig 2 (triangles) |
| IV Experimental Setup | IBM HI/LI, synthetic generator, labels X=80%, tau 0.05/0.25, max_len 6, CV, bootstrap, primary vs exploratory, multiplicity | altman2023realistic | PK 1; EX 5, 7 |
| V Results | A: D vs B; B: primary B+D+T vs B+D (HI, LI, synthetic); C: transfer + ring birth; D: exploratory (T_local/T_win, per scenario, cap, recall); E: threshold (bandits); F: severity, clustering, pump-and-dump (short) | none new | R1-R15; Tables 1-5, Figs 3-6 |
| VI Discussion and Limitations | D carries signal; why null cannot be read as "no effect"; time-proxy; generator circularity; IBM not market data; NFT not done (Dune free tier blocks running queries); not done list; deviations | arnold2024motifs, bauernoppel2025bounded | EX 11 |
| VII Conclusion and Future Work | restate; NFT subset filters 4/5, undirected and persistence-image ablations, sweeps | none | EX 11.4 |
| References | refs.bib | | |
| Appendix | optimization-report numbers (R10), full per-condition synthetic table, deviations from v3, reproduction commands | | PK 2, 5; EX 11.2, 13 |

## Related Work synthesis plan (what supports which sentence)
1. *Wash-trade measurement and rules* (what they see: self-trades, 2-cycles, repeat buys; cannot see: strength, longer rings): victor2021wash (most structures 1-2 accounts), liu2023nft (30-day repurchase rule), tosic2025beyond (intermediary transfers; up to 24% volume), cao2016wash (directed trader graph on order data), sirolly2025network (closed clusters).
2. *Graph and learning detectors for AML/fraud* (what they see: node or subgraph classes; cannot: explicit loop strength; need labels, not compared here): weber2019aml (RF beats GCN on Elliptic), ouyang2024bitchetg, pei2020sade (undirected subgraph embedding), chen2023parallel (SCC + DFS, binary).
3. *Cycle enumeration as localiser*: johnson1975circuits, gupta2021bounded, bauernoppel2025bounded (reported counterexample; untested here), chen2023parallel.
4. *Topology on graphs*: ofori2021topological (clique complex, global shape), lutgehetmann2020flagser and govc2021tournaplex (directed complexes, neuroscience), zhao2019wkpi and bubenik2015landscapes (vectorisation). Where this work sits: per-ring directed features tested against a baseline. No priority claim.
5. *Drift and thresholds*: vivek2025rosfd, garivier2011switching, trovo2020swts, fiandri2024swts. Our setting is simulated only.
6. *Data*: altman2023realistic, jensen2023synthaml (alternative, unused).
Community-detection, centrality and Fraudar sentences: Fraudar only (hooi2017fraudar [VERIFY]); the others stay [citation needed] and are phrased without citation or dropped.

## Tables (max 6) and figures (max 6)
Tables:
- T1 Datasets and candidate counts (PK 1; HI/LI primary rows, synthetic conditions). *Exists* (data in PK).
- T2 HI-Small PR-AUC by feature set and model, with prevalence and groups (EX 8.3). *Exists.*
- T3 Primary and secondary paired differences: HI, LI, synthetic four conditions, transfer (EX 8.3-8.6). *Exists.*
- T4 Exploratory: T_local/T_win, cap sensitivity, per scenario summary (EX 8.3, 8.8, 8.6). *Exists.*
- T5 Bandit benchmark: share of oracle by policy at lambda 1, lag 2, and paired difference vs retune_M5 (EX 8.9). *Exists.*
- T6 Exploratory side studies (severity, clustering, pump-and-dump) one-line each with key number. *Exists.*

Figures:
- F1 Pipeline diagram. **Missing** (ASCII only in explainer); draw in TikZ in the paper (no data).
- F2 Circular vs transitive triangle with bars. **Missing**; TikZ.
- F3 `hi_pr_auc.png` HI PR-AUC by feature set and model. Exists, check legibility at column width.
- F4 `synthetic_lift_per_scenario.png` per-scenario paired lift. Exists (exploratory; label so).
- F5 `bandit_drift.png`. Exists.
- F6 `recall_ibm.png` or `synthetic_pr_auc.png`. Exists; choose one. Space permitting, `recall_ibm.png`.
- Not used: `cluster_pca_*.png` (exploratory, appendix at most), `pump_dump_pr_auc.png`, audit figures, report figures `fig1-5` (appendix only; `fig4_precision_recall.png` is the report's own and is on the pinned generator). **No PR-curve figure exists (F6 in paper_content never made).** No LI PR-AUC figure exists (LI numbers go in a table).
- Stale flags: `hi_pr_auc.png` was regenerated after the bug fix per figure_list; none flagged stale. Report figures are from the pinned generator and must be labelled so.

## Build route
No LaTeX installation found (no `pdflatex`, `latexmk`, `xelatex`, `tectonic`, `kpsewhich`); pandoc is available. Stage 4 will write `paper/ieee/main.tex` (IEEEtran conference) that cannot be compiled here, and a pandoc-built `main.docx` from the same content as the readable draft. Page count of the final PDF therefore cannot be measured here; word budget about 6500 words plus tables.
