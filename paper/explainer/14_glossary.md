
## 14. Glossary and FAQ

### 14.1 Glossary

* **Accumulation window.** A time band in which at least m accounts each buy a large amount; the candidate unit of the pump-and-dump track.
* **Ambiguous (candidate).** A candidate whose label cannot be decided by the 80% rule (split backing) or, on IBM, whose flagged edges rest only on laundering rows that belong to no attempt. Reported excluded and counted positive.
* **ARI (adjusted Rand index).** Agreement between two partitions, 0 for random labelling, 1 for identical.
* **Arm (bandit).** One selectable option; here a fixed flagging threshold.
* **Attempt.** One laundering episode in the IBM patterns file, with a pattern type.
* **B features.** Graph baseline features (degrees, betweenness, clustering, neighbourhood density), 12 columns.
* **Background (candidate).** A candidate none of whose edges are backed by a planted structure or attempt.
* **Bandit.** A sequential decision method that learns which option pays best from the rewards of the options it plays.
* **Bar.** A (birth, death) pair of one hole in a persistence diagram.
* **Betweenness.** How often a node lies on shortest paths between others; sampled with k = 100 sources on IBM windows.
* **Birth.** The filtration value at which a hole (or a ring) first exists; for a ring, the maximum edge weight on it.
* **Bootstrap, grouped.** Resampling whole groups with replacement to get a confidence interval.
* **Candidate ring.** A directed simple cycle of at most 6 accounts found in a strongly connected component of the sparsified graph.
* **Cap (infinite bar).** The finite value substituted for an infinite death (default: the window's maximum edge weight).
* **Circular triangle.** a to b to c to a; never filled in the directed flag complex.
* **Class weight.** Weighting of rare positives in training; used as balanced in all classifiers.
* **Clique / flag complex.** The complex that fills every fully connected set of nodes (undirected) or every ordered transitive set (directed).
* **CV (coefficient of variation).** Standard deviation divided by mean; used for the gaps between payments.
* **D features.** Direction and behaviour features, 17 columns, including ring birth.
* **Death.** The filtration value at which a hole is filled in; infinity if never.
* **Dedup.** Removing near-duplicate candidates; effectively off at `max_len` of at most 9.
* **Delay (d).** The number of windows by which labels arrive late.
* **Directed flag complex.** A complex whose 2-cells are ordered triples (a, b, c) with a to b, a to c, b to c.
* **Discounted UCB.** Upper-confidence-bound bandit that down-weights old observations by gamma per window.
* **Drift.** Gradual change in manipulators' tactics over time.
* **Edge weight.** One minus the mean of the three rank-normalised scores; low = suspicious.
* **Enumeration cap.** The limit of 20000 cycles per strongly connected component.
* **Filtration.** A nested family of graphs or complexes indexed by a rising threshold.
* **Exit synchrony.** How tightly the members of a group sell at the same time (the spread of their first-sell times); the main feature of the pump-and-dump track.
* **Fixed threshold.** A flagging cutoff tuned once and frozen.
* **flagser.** The C++ engine (via pyflagser) computing persistent homology of directed flag complexes.
* **Generator circularity.** The risk that a method looks good because it matches the data designer's assumptions.
* **GNN.** Graph neural network.
* **Group.** A set of candidates that must stay together in cross-validation (shared edge or attempt id).
* **H0, H1.** Homology in dimension 0 (components) and 1 (loops).
* **Hard negative.** A legitimate loop (arbitrage, market maker, hot wallet) that resembles a ring.
* **Homology.** A tool that counts holes of each dimension.
* **Imputation (timing).** Replacing an undefined timing score by a median.
* **Infinite bar.** A hole that is never filled in.
* **Label rule (X = 80%).** A ring takes a structure's label if at least 80% of its edges are backed by that structure.
* **Landscape (persistence).** A fixed-length vector summary of a persistence diagram.
* **Lag.** See delay.
* **Majority purity.** Purity obtained by assigning every ring to the largest class; a baseline.
* **Multiplicity.** The inflation of false positives when many comparisons are made; not corrected here.
* **Oracle (hindsight).** The best threshold per window using that window's own labels; an upper bound.
* **Out-of-fold score.** A prediction made by a model that did not see that candidate's group in training.
* **Paired difference.** A difference between two feature sets computed on the same bootstrap resamples.
* **Pump-and-dump.** A scheme in which a group accumulates an asset, pushes the price and sells together; a temporal coordination motif, not a closed cycle.
* **Persistence diagram.** The set of (birth, death) bars of a filtration.
* **Persistent homology.** Tracking holes along a filtration.
* **Pooled out-of-fold PR-AUC.** Average precision of all out-of-fold scores taken together.
* **PR-AUC.** Area under the precision-recall curve (average precision); a random scorer scores the prevalence.
* **Prevalence.** The share of positives among the evaluated candidates.
* **Primary (comparison).** The pre-specified B+D+T minus B+D.
* **Projection onto the simplex.** Mapping a vector to the nearest non-negative vector summing to 1.
* **Proposition 1.** Equal weights cannot push a multi-party ring below 1/3.
* **Purity.** The share of rings assigned to the majority type of their cluster.
* **Rank normalisation.** Replacing a score by its rank within the window, scaled to [0, 1].
* **Recall (candidate).** The share of true rings returned by the candidate generator.
* **Reference backend.** The pure-Python persistence implementation used as a cross-check.
* **Ring.** A directed cycle of colluding accounts.
* **Ring birth.** See birth.
* **Ring survival (`T_local_ring_inf`).** Exact Z/2 test that a ring's own loop is never filled.
* **SCC.** Strongly connected component.
* **Silhouette.** A clustering quality score in [-1, 1].
* **Simplex (probability).** The set of non-negative vectors summing to 1.
* **Sliding-window Thompson sampling.** A bandit that samples from beliefs built on the last W windows.
* **Sparsification.** Keeping only edges whose weight is at most tau.
* **Strict groups.** Groups that also merge candidates sharing an account.
* **Sublevel-set filtration.** Cells enter when their value is at most a rising threshold.
* **T features.** Topological features (T_local and T_win), 311 columns.
* **Tau (tau_quantile).** The sparsification cutoff, a quantile of the window's edge weights.
* **Tail share.** The share of a group's pairs in the most suspicious 10% tail of a score.
* **Transitive triangle.** a to b, b to c, a to c; filled in the directed flag complex.
* **Transfer.** Training on one dataset and testing on the other.
* **Wash trading.** Trading among colluding accounts to fabricate volume.
* **Window.** A time slice of transactions (5 days with a 1-day stride on IBM).
* **Z/2.** Arithmetic modulo 2, used for the homology computations.

### 14.2 Twenty questions an examiner would ask

1. **What is the problem?** Detecting loops of colluding accounts (wash trading, laundering cycles) that single transactions do not reveal (section 2).
2. **Why a graph and a loop?** Value circulating and netting to about zero is a statement about directed paths among a few accounts (sections 2.2, 4.1).
3. **Why does density-based detection miss it?** A sparse loop is not dense; density ignores direction (section 3.1).
4. **What does persistence add over cycle enumeration?** Enumeration gives a binary answer; persistence says how long the loop stays open as weaker edges are admitted, which differs only when the neighbourhood fills, splits or crowds the loop (section 4.6).
5. **Why directed?** Undirected cliques fill circular and transitive triangles alike; the directed flag complex keeps the circular one open (section 4.5).
6. **Why not read rings from the homology library?** It returns no representative cycles, so localisation is combinatorial (section 4.7).
7. **What are B, D and T?** Baseline graph features, direction/behaviour features including ring birth, and topological features (section 4.8).
8. **Why is ring birth in D?** It is a function of the ring's own edge weights; counting it in T would overstate topology (section 4.8).
9. **What is the primary comparison and why?** B+D+T against B+D: it isolates the topological increment (section 4.9).
10. **What was the primary result and its uncertainty?** No detectable topological lift on IBM: HI logistic regression +0.104 [-0.002, +0.171], with 14 positive groups (section 8.3).
11. **What does D do?** It adds clearly over B on HI (CI excludes 0 for 3 of 3 models) (section 8.3).
12. **Why are the intervals wide?** The effective sample size is the number of positive groups, 14 and 10 (section 7).
13. **Why grouped cross-validation?** Near-duplicate rings recur and would leak across a random split (section 7).
14. **What failed?** Time-ordered split (unusable), the ring-birth detector on LI, the bandits, any topological lift on IBM, clustering by type, severity as forecasting (sections 8, 12).
15. **Did topology help anywhere?** Slightly on the synthetic generator (7 of 12 cells), which is nearly separable by D (section 8.6).
16. **Is the severity result forecasting?** No: ring size alone does as well; synthetic, generator decides persistence (section 8.10).
17. **Why did the bandits lose?** One exploratory play of the flag-everything arm costs about as much as the whole oracle total and is hidden by reward clipping; one stream, fixed hyperparameters (section 8.9).
18. **What did you not do and why?** NFT (Dune free tier blocks running queries), real-data pump-and-dump (a synthetic, no-topology track was run), undirected and persistence-image ablations, window and label-threshold sweeps (section 11.4).
19. **Where do you differ from published work, and where can you not claim to?** Direction and exact localisation are differences; no comparison with a GNN or the published clique method was run (section 10).
20. **How do I reproduce it?** Section 13; every number is traceable through `paper/numbers_pack.md` and `paper/numbers_pack/provenance.csv`.

### 14.3 References (verified) and citation markers

Verified by web search (URL given):

1. Altman, Blanuša, von Niederhäusern, Egressy, Anghel, Atasu. *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models.* NeurIPS 2023 (Datasets and Benchmarks). https://proceedings.neurips.cc/paper_files/paper/2023/hash/5f38404edff6f3f642d6fa5892479c42-Abstract.html ; arXiv 2306.16424.
2. Lütgehetmann, Govc, Smith, Levi. *Computing persistent homology of directed flag complexes.* Algorithms 13(1), 19, 2020 (flagser). https://arxiv.org/abs/1906.10458
3. Garivier, Moulines. *On Upper-Confidence Bound Policies for Switching Bandit Problems.* ALT 2011 (discounted and sliding-window UCB). https://www.arxiv.org/abs/0805.3415
4. Ofori-Boateng, Segovia Dominguez, Kantarcioglu, Akcora, Gel. *Topological Anomaly Detection in Dynamic Multilayer Blockchain Networks.* ECML PKDD 2021. https://arxiv.org/abs/2106.01806
5. Bubenik. *Statistical Topological Data Analysis using Persistence Landscapes.* JMLR 16, 2015. https://www.jmlr.org/papers/v16/bubenik15a.html

Claims left as **[citation needed]** in the text: community and density methods (3.1), centrality scores (3.2), the specific cycle-enumeration algorithms (3.4), sliding-window Thompson sampling (6.11).
