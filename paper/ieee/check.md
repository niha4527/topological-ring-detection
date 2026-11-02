# check.md: Stage 5 verification of main.tex

Pages are PDF page indices in `paper/Reference papers/`. URLs were opened in Stage 5 (WebFetch/WebSearch, 2 Nov 2026).

## 1. Citations: claim supported and where

| key | claim in main.tex | backing |
|---|---|---|
| victor2021wash | Order-book DEXs; lower bound 159 million USD; more than 30% of tokens; mostly one or two accounts | PDF p.1 (abstract, image-only, viewed) and p.2 contributions list |
| tosic2025beyond | NFT ownership traces joined with Ethereum network; up to 24% of volume; Meebits 93% | PDF p.1 (abstract) |
| liu2023nft | 30-day repurchase rule; seven collections; average 0.14% of transactions | PDF p.1 (abstract); rule also p.2 |
| sirolly2025network | closed clusters of colluding counterparties; prediction market; working paper | PDF p.1 (abstract; "Polymarket exchange"); not peer reviewed: stated in text |
| cao2016wash | directed trader graph; order matching; two knapsack-type steps by dynamic programming; stock data | PDF p.1 (abstract) |
| hooi2017fraudar | camouflage-resistant dense-subgraph detection on bipartite review graphs | PDF first page of the TKDD version fetched from https://alexbeutel.com/papers/tkdd_fraud_detection_with_camo.pdf |
| weber2019aml | labelled Bitcoin graph; random forest outperforms GCN | PDF p.1 (abstract) and p.6 ("Random Forest ... also outperforms GCN") |
| ouyang2024bitchetg | laundering-group detection by subgraph contrastive learning; Micro-F1 at least 5% above others (not quoted in text) | PDF p.1 (abstract) |
| pei2020sade | role-guided subgraph embedding + Isolation Forest; undirected, unweighted | PDF p.1-2 (image-only, viewed; Definition 1 and surrounding text "restrict ... to undirected, unweighted networks") |
| chen2023parallel | shrink SCCs, parallel DFS, mark suspicious accounts | PDF p.1 (abstract) |
| networkx_simple_cycles, johnson1975circuits, gupta2021bounded | Johnson for unbounded case; version of Gupta-Suzumura for bounded case | https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cycles.simple_cycles.html |
| bauernoppel2025bounded | presents examples where Gupta-Suzumura fails to enumerate valid cycles | https://arxiv.org/abs/2512.08392v1 (abstract; v1 submitted 9 Dec 2025) |
| ofori2021topological | clique persistent homology; multilayer blockchain; stacked PD; outperforms state-of-art | PDF p.1; venue and pages: https://www.doi.org/10.1007/978-3-030-86486-6_48 and https://mlanthology.org/ecmlpkdd/2021/oforiboateng2021ecmlpkdd-topological (search result: ECML PKDD 2021 research track, LNCS 12975, pp. 788-804) |
| lutgehetmann2020flagser | flagser computes persistence of directed flag complexes; brain data; built on Ripser | PDF p.1 (abstract) |
| govc2021tournaplex | flag tournaplexes distinguish graph dynamics more finely than the directed flag complex | PDF p.1 (abstract) |
| bubenik2015landscapes | persistence landscapes | https://www.jmlr.org/papers/v16/bubenik15a.html (JMLR 16(3), 77-102, 2015) |
| zhao2019wkpi | learned weighting of persistence summaries | PDF p.1 (abstract) |
| vivek2025rosfd | drift detectors decide when to retrain; fewer retrains with little AUC loss | PDF p.1 (abstract) |
| garivier2011switching | discounted UCB for switching bandits | web search result: ALT 2011, pp. 174-188, DOI 10.1007/978-3-642-24412-4_16 (https://mlanthology.org/alt/2011/garivier2011alt-upperconfidence); arXiv 0805.3415 title is "...Non-Stationary Bandit Problems" (https://www.arxiv.org/abs/0805.3415), and the search result says it analyzes discounted and sliding-window UCB. Whether the ALT paper has the same content was not confirmed |
| trovo2020swts | sliding-window Thompson sampling | https://jair.org/index.php/jair/article/view/11407 (JAIR 68, 2020, DOI 10.1613/jair.1.11407) |
| fiandri2024swts | analysis revisited and corrected | https://arxiv.org/abs/2409.05181 (abstract: "corrects and generalizes existing work"; it does not say which variant, so the text no longer names the Beta prior) |
| altman2023realistic | synthetic AML data with complete labels | PDF p.1, p.7 ("ground truth labels are complete") |
| jensen2023synthaml | SynthAML built from real bank data | PDF p.1 (abstract) |
| arnold2024motifs | whole-network motif counts can mislead; per-user and per-period views reveal more | PDF p.1 (abstract) |

## 2. Claims not backed by a PDF or URL (project-internal, from repo files)
- Persistence library interface returns no representative cycles (v3 design §3.3 and explainer 4.7): not in the flagser paper, so the text says "the pyflagser interface we use".
- IBM patterns file groups laundering transactions into attempts; 1,968 of 5,177 laundering rows in no attempt; 11.6% self-loops; 3 of 705,907 LI accounts shared: README / explainer 5.2, 8.5.
- Two of five filters of the Dune wash-trade model are cycle-type rules; BigQuery schemas inspected: explainer 5.3 (our own inspection; stated as "as inspected by us").
- 82 s and 11.5 s timings: README measurements.
- Garivier and Moulines: the ALT 2011 content equals the arXiv analysis only per a search snippet. Only used as "discounted UCB \cite{...}", which the arXiv abstract page and ALT title support.
No claim was removed. One was softened (Fiandri) and two were reworded (pyflagser).

## 3. Remaining placeholders
- Title block: four authors (RV University, Bengaluru) filled in; no author placeholder remains.
- No `[VERIFY]` or `[citation needed]` remains in main.tex. In refs.bib: `weber2019aml` has no DOI (none printed in the PDF); `trovo2020swts` has no page range (not listed on the JAIR page); `johnson1975circuits` carries no DOI or pages (only what the NetworkX page states); `fiandri2024swts` and `bauernoppel2025bounded` are preprints.
- Community-detection and centrality sentences were dropped, so the explainer's [citation needed] items do not appear.

## 4. Numeric self-audit
`python experiments/check_paper_numbers.py paper/ieee/main.tex` (tokens normalised, signs ignored, searched in notes.md, the pack, and `paper/numbers_pack/csv/*.csv`):
- numbers in the text: 710 occurrences, 346 distinct; matched 709; unmatched 1 (`6}`, the exponent of $10^{-6}$ read as a token: not a number).
- First run had 6 unmatched; the real ones (35 windows, 150 fold-repeats, 159 million, 11.5 s, 705,907) were added to notes.md section 8 with their sources. 159 million comes from the Victor PDF, 35/150 from the explainer, 11.5 and 705,907 from README.
- Caveat: token-level matching can match a number by coincidence. A stricter check on all 99 `x [a, b]` interval triples in the text found each verbatim (signs dropped) in `paper/numbers_pack.md`; none missing.
- All pack and explainer tables behind the cited cells were read in this session (explainer 8.3-8.9 and 11 with its "copied from the pack" tables, pack sections 1, 5 headers, 7, 8, 9); the pack's mid-section tables that duplicate them were verified by the token and triple check rather than line by line.

## 5. Static check output (`python experiments/check_latex_static.py paper/ieee`)
```
cite commands: 36, distinct keys cited: 26, bib entries: 26
bib entries never cited: none
labels: 19, refs: 25
labels never referenced: ['sec:conclusion', 'sec:intro', 'tab:expl']
includegraphics: 4 -> figures/f1_pipeline.png, f2_triangles.png, f3_hi_pr_auc.png, f4_lift_per_scenario.png
environments checked: 21
main-text figures: 4, tables: 5
OK: no errors
```
LaTeX was not compiled (no TeX installation here), so LaTeX warnings and the final page count are unknown. `pytest -q`: 210 passed.

## 6. Word counts (main.tex, tables, figures and math excluded; approximate counter)
Abstract 228; Introduction 564; Related Work 719; Method 840; Experimental Setup 578; Results 1395; Discussion and Limitations 925; Conclusion 165. Body total 5186. Captions 548. Page estimate from words alone: about 8 to 9 pages in two-column IEEEtran including 4 figures, 5 tables and references, plus a one-page appendix; this is an estimate, not a measurement.

## 7. Figures and tables in the main text
Figures (4; limit 5):
- Fig. 1 `f1_pipeline.png`: pipeline schematic (new, `experiments/make_paper_figures.py`; text 7 pt at 7.0 in width).
- Fig. 2 `f2_triangles.png`: H1 bars of circular, transitive and shortcut-filled triangles, computed with the reference backend on toy graphs (new; 7 pt at 3.4 in).
- Fig. 3 `f3_hi_pr_auc.png`: HI-Small PR-AUC, six feature sets, LR/RF/SVM, prevalence 0.0173, 14 groups, grouped-bootstrap CI.
- Fig. 4 `f4_lift_per_scenario.png`: synthetic per-scenario paired lift (exploratory), grouped-bootstrap CI.
Figs. 3 and 4 are replotted from the same result CSVs with `experiments/replot_paper_figures.py` because the original PNGs (`paper/numbers_pack/figures/hi_pr_auc.png`, `synthetic_lift_per_scenario.png`) were drawn at 11 in width and would be about 3 pt at column width. Values are unchanged; Fig. 3 shows 6 of the 9 feature sets. The originals, `bandit_drift.png`, `recall_ibm.png`, cluster and audit figures, and the optimization-report figures are not used. None of the existing figures is stale.
Tables (5; limit 5): T1 data (Table I), T2 PR-AUC HI and LI, T3 paired differences (primary, secondary, transfer, synthetic), T4 candidate recall, T5 bandits. Appendix: A1 exploratory side studies. Figure 1 and Table III span two columns.

## 8. Sentences I am unsure about
1. Intro: "Measurement studies find it on decentralized exchanges and in NFT markets [victor, tosic, liu]": Liu et al. flag a small share (0.14%) on average; "find it" is correct but the three results differ widely in size.
2. Related Work, Ofori-Boateng: "it describes the network as a whole, not individual rings" is my reading of the abstract ("track evolution of the network shape"), not a sentence of theirs.
3. Related Work, Fraudar "dense-subgraph detection under camouflage on bipartite review graphs" is accurate for the cited paper; it is a fraud detector, not a wash-trade one.
4. Method, Proposition 1 is stated without proof (explainer 6.2 gives the one-line argument).
5. Discussion, "the largest upper bound is +0.171": taken over the within-dataset primary rows (HI-Small and LI-Small). The text now scopes "every interval contains 0" to those rows and names the two transfer cells with intervals that exclude 0 (HI to LI RF, LI to HI LR); the largest transfer upper bound is +0.092.
6. Results, Limitation 4 wording on ambiguity shares comes from the explainer (11.1 item 10).
7. Appendix B severity sentence ("fair persistence fit") is explained only in the explainer (6.12, 8.10), not in the paper body.
8. Garivier and Moulines, see section 2.
9. The "Practical reading" paragraph is interpretation, not a result.

## 9. Revision after review (edits applied to main.tex, not committed)
Scope of "no detectable lift" is now within-dataset (HI-Small, LI-Small); the two negative transfer cells are stated in the abstract, contribution 2, V-C, VI and the conclusion. Ring-birth wording changed from "does not transfer" to "does not replicate on LI-Small" (two separate no-model scores; LI interval 0.006 [0.002, 0.025] contains the prevalence 0.0039). Audit AUCs: raw balance 0.795 and timing 0.601, rank-normalised 0.804 and 0.589, weight 0.794 (pooled windows 0 and 2). The sentence "other pattern types closer to chance" was wrong for RANDOM (weight AUC 0.785) and was corrected. "Practical reading" was cut to one sentence; the 82 s and 11.5 s figures were removed.
