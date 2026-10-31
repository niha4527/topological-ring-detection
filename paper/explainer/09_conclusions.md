
## 9. Conclusions

### 9.1 Goals and what the data say

The goals G1 to G6 are those of `prd.md`. One line each, no spin.

| Goal | What the data say |
|---|---|
| G1 Detect rings with a continuous, interpretable score | Rings are found exactly (candidate recall is partial and tau-dependent: section 8.7); ring birth is a continuous score, but as a single-feature detector it works on HI (0.314 at prevalence 0.0173) and does not carry to LI (0.006 at prevalence 0.0039). |
| G2 Test whether topology adds value over a density/centrality baseline | **Direction and behaviour features add clearly** (B+D minus B: CI excludes 0 in 3 of 3 models on HI, 2 of 3 on LI). **No detectable lift from topology on IBM** (B+D+T minus B+D includes 0 in 3 of 3 models on HI and LI; CIs cannot rule out a small effect either way); a small lift in 7 of 12 synthetic condition-model cells. |
| G3 Stratify by scenario | Done on synthetic data. The lift is **not** concentrated where v3 section 3.4 predicted: per scenario, B+D+T minus B+D over the four conditions and three models (12 cells each): isolated rings 8 of 12 cells with a CI above 0 (SVM 4 of 4, +0.096 to +0.191; logistic regression +0.032 to +0.071 and random forest +0.002 to +0.101, 2 of 4 each); dense rings 2 of 12 (only with the internal chords removed and hard negatives off: logistic regression +0.130 [+0.038, +0.330] and random forest +0.123 [+0.029, +0.329]); chorded rings 6 of 12 (SVM 4 of 4, +0.229 to +0.319; random forest 2 of 4, -0.011 to +0.090; logistic regression 0 of 4, -0.023 to -0.004). It is largest for isolated rings, where topology was predicted to be nearly redundant. The cause is untested (candidates: T also encodes neighbourhood size and degree structure; generator artefacts). The generator is nearly separable by D alone. |
| G4 Type rings by clustering and forecast severity without leakage | Exploratory. Clustering agrees weakly with pattern types and more with attempts; severity (synthetic only) recovers generator structure, not forecasting skill, and T adds nothing. |
| G5 Keep the threshold accurate under drift with delayed labels | Online gradient ascent reaches 0.981 of the hindsight oracle at lambda 1; re-tuning every 5 windows 0.861 at lag 2; **the bandits lose to re-tuning at the primary setting**; fixed thresholds earn about 0.298. |
| G6 Validate on synthetic data and IBM AML | Done for both. The NFT leg was **not** done (Dune free tier blocks running queries). |

### 9.2 The overall answer to "does topology add predictive value?"

**On the evidence in this project: not detectably, on IBM AML data (itself synthetic), and slightly, on a synthetic generator built so that topology should matter.** The honest summary has four parts.

1. *The strongest signal is not topological.* Direction and behaviour features (including ring birth, which the filtration produces) carry most of the predictive power in every dataset.
2. *The topological increment is small at most.* On IBM the point estimates for B+D+T minus B+D are small (HI: logistic regression +0.104 [-0.002, +0.171], random forest +0.025 [-0.050, +0.102], SVM +0.003 [-0.111, +0.077]) and their intervals include 0. With 14 and 10 positive groups, a small positive effect and a small negative effect are both compatible with the data.
3. *Where the synthetic gain appears it is not where the design predicted.* Per scenario (12 model-condition cells each) it is clearest for isolated rings (8 of 12 cells with a CI above 0), where v3 section 3.4 predicted topology to be nearly redundant with enumeration; for dense rings only 2 of 12 (internal chords removed, hard negatives off); for chorded rings mixed (SVM 4 of 4, random forest 2 of 4, logistic regression 0 of 4). The cause is untested: T also encodes neighbourhood size and degree structure, and the generator may have artefacts. It is a setting we control, so it does not transfer to a claim about real laundering.
4. *Exploratory local-topology gains* (B+D+T_local) are positive for logistic regression on HI, but they are one of the 9 exploratory paired rows of the HI primary variant (3 exploratory comparisons x 3 models; 36 such rows across the four HI variants) and are not corrected for multiplicity.

### 9.3 What the project can legitimately claim

* A working, tested pipeline (windows, weights, exact candidate rings, directed flag persistence with two agreeing backends, an exact ring-survival test, B/D/T features, leakage guards, group-aware evaluation).
* A documented, pre-specified comparison with a mostly null answer on IBM and the reasons the evidence is thin.
* A set of exploratory and negative results (bandits, severity, clustering, cap sensitivity), each labelled.
* A minor synthetic, no-topology pump-and-dump track showing that the harness runs on a second pattern type (section 8.14); it supports nothing about real pump-and-dump or about topology.

It cannot claim: that topology improves real-world wash-trading detection, that the bandits improve on re-tuning, that rings are typed by their topology, or anything about NFT data.

## 10. How this differs from other solutions

### 10.1 Comparison table

| | Density / community | Centrality | GNN | Cycle enumeration | Clique persistent homology | **This project** |
|---|---|---|---|---|---|---|
| What is detected | tightly connected groups | unusual nodes | learned patterns | every directed loop | global shape change | candidate rings, then classified |
| Strength score | density | centrality | learned score | none (binary) | global persistence summaries | ring birth plus persistence features |
| Ring typing | no | no | possible | no | no | exploratory clustering (weak) |
| Direction | usually ignored | partly | yes | yes | **no** (undirected clique complex) | **yes** (directed flag complex) |
| Adaptive threshold | no | no | no | no | no | online ascent and bandits (bandits lose) |
| Interpretability | medium | high | low | high | medium | high (explicit accounts and edges) |
| Evidence in this project | not run | B baseline only | **not run** | used as the localiser | **not run** | all results above |

*Note.* The columns for other methods are general characterisations of method families, not findings of this project, and two of those families (density/community and centrality) lack a citation in section 3. Only the bottom row, and the right-hand column, are evidence from this project.

### 10.2 Differences we can defend

* **Direction matters for the topology**, argued from construction (a circular triangle survives, a transitive one is filled; the tests check it). We did not run the undirected ablation, so the *size* of the difference is not measured.
* **Localisation is exact and account-level**, because flagser returns no representative cycles; persistence is a feature source, not the localiser.
* **The comparison is controlled**: ring birth is kept out of T, B sees at least as much of the graph as T, and the primary test is B+D+T against B+D under group-aware CV.
* **Negative results are reported**, with the group counts that limit them.

### 10.3 Differences we cannot defend

* **We did not compare against a GNN**, against community detection, or against the published clique-persistent-homology method. Any statement that we "outperform" them would be unsupported.
* **We cannot say topology is better than cycle enumeration with behaviour features**; on IBM the data cannot distinguish them.
* **The adaptive-threshold comparison is on one synthetic drift stream** and the bandit configuration (arm grid, reward clipping) was fixed in advance, not explored.
