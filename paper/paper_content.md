# Paper Content Pack: Topological Ring Detection for Market Manipulation

Working material for the paper. It covers everything except observations and inference, which are filled in after the experiments finish.

**Sources.** `Topological_Ring_Detection_for_Market_Manipulation_v3.md` (design), `report/Optimization_Report` (formulation and gradient methods), `README.md` and `CLAUDE.md` (implementation status and deviations).

**Conventions.**
- `[RESULT]` marks a slot to fill after the experiments.
- `[VERIFY]` marks a citation or claim that comes from memory or from the project docs and must be checked against the source before submission. No reference below has been checked.
- `[TODO-CODE]` marks something the paper will claim that the code does not do yet.

---

## 0. Paper at a glance

### 0.1 Candidate titles
1. Topological Ring Detection for Market Manipulation: Persistent Homology on Directed Flag Complexes
2. Seeing the Loop: Persistent Homology of Directed Transaction Graphs for Wash-Trading Detection
3. Beyond Cycles and Clusters: Persistence-Based Scoring, Typing and Adaptive Thresholding of Manipulation Rings

### 0.2 One-paragraph pitch
Wash trading hides in the shape of the transaction graph rather than in any single trade: value circulates through a small group of accounts and returns to its origin. We model each time window as a weighted directed graph whose edge weights come from behavioural suspicion scores, localize rings exactly by cycle enumeration, and ask whether persistent homology of the **directed flag complex** (circulating flow survives, one-way flow is filled in) adds predictive value over direction and behaviour features. The answer is mixed and reported with its uncertainty. The direction and behaviour features carry the signal: adding them to a graph-density baseline raises PR-AUC on IBM AML data. We found **no detectable topological lift on IBM AML** (the paired confidence intervals for B+D+T minus B+D include 0 with 14 and 10 positive groups, and cannot rule out a small effect either way). On a synthetic generator that is nearly separable by the direction features, topology gives a small lift. The adaptive-threshold study shows that discounted UCB and sliding-window Thompson bandits **lose to periodic re-tuning** at the primary setting, while online gradient ascent tracks drift well. The real-data NFT experiment was **not done** (running Dune queries is blocked on the free tier). Clustering of positive rings and severity regression are exploratory and, for severity, synthetic only.

### 0.3 Contributions (aligned with the results; see `paper/numbers_pack.md`)
1. **A behaviour-weighted directed filtration for rings.** Three rank-normalized edge scores (pair reciprocity, node flow balance, timing regularity) define a sublevel-set filtration, so ring strength becomes a continuous quantity (§4.2). Proposition 1 shows equal weights cannot push a multi-party ring below 1/3, which motivates learning the weights.
2. **Directed topology for circulation, implemented and tested.** A weighted directed flag complex keeps a circular triangle open and fills a transitive one; the exact ring-survival test (`T_local_ring_inf`) and two agreeing persistence backends are verified by tests (§4.3).
3. **Exact localization plus topological features.** Candidate rings come from SCC decomposition and bounded cycle enumeration; persistence of each ring's neighbourhood is a feature source, not the localizer (§4.4).
4. **An honest test of topological lift, with a mostly null answer on real-structure data.** Feature sets B, B+D, B+D+T and T, with ring birth in D; grouped cross-validation and grouped-bootstrap CIs. Finding: D carries the signal; no detectable topological lift on IBM HI-Small and LI-Small (CIs cannot rule out a small effect either way); a small lift on near-separable synthetic data.
5. **Weights and threshold as optimization problems.** Weights learned by projected gradient descent on the simplex; the threshold tracked by online gradient ascent with delayed labels. A bandit benchmark (discounted UCB, sliding-window Thompson) was run and the bandits lose to periodic re-tuning at the primary setting.
6. **An evaluation protocol for this setting and a documented set of negative results.** Group-aware splits (the time-ordered split was unusable and dropped), a pre-specified primary comparison versus exploratory ones with a multiplicity count, candidate-recall reporting, and a label-ambiguity rule counted both ways.
7. **Exploratory extras, not headline claims.** Ward clustering of positive IBM rings against pattern types (weak agreement, clusters track attempts more than types) and a synthetic-only severity regression (recovers generator structure, not forecasting skill).

**Done as an optional extra.** A synthetic-only pump-and-dump temporal-motif track with no topology (§5.11; `paper/numbers_pack.md` section 6F). **Not done.** NFT transfer (Phase 5), real-data pump-and-dump validation, undirected-complex and persistence-image ablations, per-score classifier filtrations, and sweeps over window length, stride and label threshold.

---

## 1. Abstract (skeleton, no results)

> Wash trading and related manipulation schemes fabricate market activity by circulating assets among colluding accounts. Individual trades look ordinary; the pattern of connections does not. Existing graph-based detectors either target dense clusters, and so miss sparse rings, or enumerate cycles and give only a binary verdict with no measure of strength, stability or type. We present a pipeline that represents each time window as a weighted directed graph and measures how strongly each loop persists using persistent homology of a directed flag complex, in which circular flows are never filled while one-way flows are. Edge weights combine reciprocity, flow balance and timing regularity. Candidate rings are localized exactly by cycle enumeration inside strongly connected components and are described by direction-aware, graph-baseline and topological features. On these we perform classification, clustering into ring types, and next-window severity regression. We further pose the score weights and the alarm threshold as optimization problems solved by projected gradient descent and online gradient ascent, so that the detector adapts to drifting tactics. We evaluate on a synthetic generator with planted rings and realistic legitimate loops, on the IBM AML benchmark, and on Ethereum NFT trades, using temporal splits, bootstrap confidence intervals and scenario-stratified ablations. [RESULT: one or two sentences with headline numbers and the topological-lift finding, whichever way it goes.]

---

## 2. Introduction

### 2.1 Motivation (paragraph-level outline)
- **P1. Wash trading is real and large.** Wash trading means the same party, or a coordinated group, is on both sides of a trade. It inflates volume, misleads other traders, pushes prices, and earns volume-based rewards. It is illegal in regulated securities markets and widespread in crypto-asset and NFT markets, where accounts are cheap and pseudonymous. `[VERIFY: cite measurement studies of wash trading on crypto exchanges and NFT markets (e.g. Cong et al.; Victor & Weintraud; von Wachter et al.)]`
- **P2. The signal is relational.** A single wash trade is unremarkable. Evidence lives in the topology of who trades with whom: A ⇄ B ping-pong, A → B → C → A rings, with near-zero net position per member and a bot-like cadence.
- **P3. Two gaps.**
  - Density-based detectors (community detection, centrality, dense-subgraph mining, GNN classifiers) look for tight clusters. A ring can be sparse: each member has only 2–3 counterparties.
  - Cycle-based methods are combinatorial and binary. They do not say how strong or stable a cycle is, they cannot separate a wash loop from a legitimate arbitrage or market-making loop with the same shape, and they do not type rings.
- **P4. Thresholds decay.** Manipulators adapt ring size, timing and decoy trades, so a threshold tuned once degrades.
- **P5. What we do.** Summarize the approach and contributions (§0.3).
- **P6. Roadmap.**

### 2.2 Research questions
- **RQ1 (localization).** Do bounded cycle enumeration in SCCs and a sparsification cutoff recover the planted rings, and how does candidate recall depend on the cutoff `τ_s` and the maximum cycle length?
- **RQ2 (filtration).** Do behaviour-based edge weights place wash rings earlier in the filtration than legitimate loops, and which component score carries the signal?
- **RQ3 (topological lift).** Does adding topological features T to baseline and direction-aware features (B+D+T vs B+D) improve classification, overall and per scenario? Is the lift concentrated where neighbourhoods fill, split or crowd the loop?
- **RQ4 (directed vs undirected).** How much does the directed flag complex add over an undirected clique complex?
- **RQ5 (typing).** Do cycle-based pattern types have distinguishable topological and behavioural fingerprints under clustering?
- **RQ6 (severity).** Can next-window wash volume be predicted without leakage, and do topological features help?
- **RQ7 (adaptation).** Does an online-learned threshold hold up under tactic drift compared with fixed, periodically re-tuned and hindsight-oracle thresholds?
- **RQ8 (transfer).** Do conclusions on synthetic data carry over to IBM AML (laundering patterns, for the machinery) and Ethereum NFT trades (wash trading, with proxy labels)?

---

## 3. Problem Statement

### 3.1 Informal
Given a stream of timestamped transactions between accounts, find sets of accounts that circulate value among themselves to fake activity (wash-trading rings). Score how suspicious each ring is, group rings by type, estimate how severe the scheme will be, and keep the flagging threshold accurate as tactics change.

### 3.2 Formal setting
- **Transactions.** A transaction is `(t_i, u_i, v_i, a_i)`: time, sender, receiver, amount.
- **Windows.** For window length `L` and stride `S`, window `k` covers `W_k = [kS, kS+L)`. Its graph `G_k = (V_k, E_k)` has one directed edge `(a,b)` for every ordered pair with at least one transaction in the window.
- **Edge aggregates.** `V_ab` is total volume, `n_ab` is the transaction count, and `g_j = t_{j+1} − t_j` are inter-transaction gaps for the pair.
- **Candidate ring.** A simple directed cycle of length 2–6 inside a strongly connected component of the sparsified window graph.
- **Ground truth.** Each candidate has a label `y_R` (wash ring or not) and, when available, a type.

### 3.3 Tasks
| Task | Input | Output |
|---|---|---|
| T0 Localization | window graph | set of candidate rings (accounts, edges, underlying transactions) |
| T1 Classification | candidate features | manipulation vs benign cycle, with a score |
| T2 Typing | candidate features | cluster assignment, compared with labelled pattern types |
| T3 Severity | candidate features at window `k` | the ring's wash volume in window `k+1` |
| T4 Adaptive thresholding | classifier or birth scores per window, delayed labels | threshold `τ_t` per window |

### 3.4 Scope
- **In scope (primary).** Closed-cycle wash trading in settled transaction data.
- **Optional.** Pump-and-dump coordination, as a temporal-motif track without topology.
- **Out of scope.** Spoofing. It is an order-book phenomenon (orders placed and cancelled without executing), and settled transactions contain no signal for it.

### 3.5 Why the problem is hard
1. **Class imbalance.** Rings are rare among all cycles. Most cycles in a dense window are incidental.
2. **Hard negatives.** Market makers (two-sided, regular), arbitrage bots (loops with net profit), and exchange hot wallets (high-degree hubs with balanced flow) share features with wash rings.
3. **Weight design is decisive.** A fixed equal combination of the three scores cannot place a multi-party ring below 1/3 (Proposition 1 below), and it ties wash rings to market makers.
4. **Localization is not given by homology.** Common libraries do not return full representative cycles.
5. **Label scarcity and leakage.** Real labels are heuristic. Some public wash-trade heuristics use cycle rules, which can make a cycle-based method predict its own label rule.
6. **Drift and delay.** Tactics change; confirmed labels arrive late.

---

## 4. Related Work and Existing Solutions

Citations below are from memory and must be checked `[VERIFY]` before use. Replace each parenthetical with a proper reference.

### 4.1 Wash trading and market-manipulation detection
- **Measurement and heuristics.** Empirical studies quantify wash trading on centralized exchanges, decentralized exchanges and NFT marketplaces, using statistical tests, round-trip and self-trade heuristics, and graph rules. `[VERIFY: Cong et al. 2021; Victor & Weintraud 2021; von Wachter et al. 2022]`
- **Typical heuristics.** Same-wallet or back-and-forth trades, closed loops between a few wallets, funding from a common source. Strength: transparent. Weakness: binary, tuned to known tactics, and some use cycle rules that overlap with graph-cycle methods (the leakage issue, §6.7).
- **Regulatory framing.** `[VERIFY: securities and commodities law definitions]`

### 4.2 Graph-based fraud and AML detection
- **Density and community methods.** Dense-subgraph mining, community detection, centrality scores.
- **Cycle methods.** SCC detection and cycle enumeration (Tarjan's SCC algorithm; Johnson's cycle enumeration) `[VERIFY]`. They find loops but give no strength or type.
- **Graph neural networks.** GNN classifiers on transaction graphs; benchmark data sets include Elliptic and the IBM AML (AMLworld) synthetic data `[VERIFY: Weber et al. 2019; Altman et al. NeurIPS 2023]`. Our project excludes deep learning by design (classical ML only), which is a stated scope choice and not a claim of superiority.
- **Pattern-based AML.** Fan-in, fan-out, scatter-gather, bipartite, stack and cycle patterns are the typology used in AMLworld.

### 4.3 Topological data analysis in finance and networks
- **Persistent homology basics.** `[VERIFY: Edelsbrunner & Harer; Carlsson]`.
- **Vectorizations.** Persistence landscapes (Bubenik), persistence images (Adams et al.), entropy and total-persistence summaries `[VERIFY]`.
- **TDA on financial time series.** Early-warning indicators of crashes from landscape norms `[VERIFY: Gidea & Katz 2018]`.
- **TDA on blockchain graphs.** Persistent homology and clique complexes on dynamic multilayer blockchain networks for structural anomaly detection `[VERIFY: Akcora et al. and related]`. These flag *that* the network's shape changed. They do not isolate individual rings, type them, or give a continuous severity estimate. That is the gap this project targets.
- **Directed topology.** Directed flag complexes and their persistent homology `[VERIFY: Lütgehetmann, Govc, Smith, Levi; flagser/pyflagser]`. Used mainly in neuroscience connectomics; applying them to transaction rings is, to our knowledge, new `[VERIFY with a literature search before claiming novelty]`.

### 4.4 Online learning under drift
- Multi-armed bandits for non-stationary environments, discounted UCB and sliding-window methods `[VERIFY: Garivier & Moulines 2011]`, sliding-window Thompson sampling.
- Online gradient methods and projected gradient descent on the simplex `[VERIFY: Duchi et al. 2008 for the simplex projection; Zinkevich 2003 for online gradient]`.
- Threshold tuning with delayed, censored feedback is a harder setting than a standard bandit, and we treat it as an offline simulation.

### 4.5 Gap analysis (summary table for the paper)

| Capability | Density / community | Cycle enumeration | GNN | Blockchain TDA (global) | **Ours** |
|---|---|---|---|---|---|
| Finds sparse rings | no | yes | partly | no (global shape) | yes |
| Names the accounts in a ring | partly | yes | partly | no | yes (exact) |
| Continuous strength of a ring | no | no | score only | no | yes (filtration birth, persistence) |
| Distinguishes circulation from one-way flow | no | not by itself | learned | no | yes (directed flag complex) |
| Uses behaviour along edges | partly | no | yes | partly | yes (edge weights) |
| Ring typing | no | no | no | no | yes |
| Severity forecast | no | no | no | no | yes |
| Adapts threshold to drift | no | no | no | no | yes |
| Classical, interpretable | yes | yes | no | yes | yes |

`[VERIFY every cell against the literature before publishing; adjust to what the cited work actually does.]`

---

## 5. Our Approach (detailed)

### 5.1 Overview
Seven independently testable stages:
1. Data ingestion (IBM AML, NFT trades, synthetic generator with planted rings and drift).
2. Graph construction (windowing, per-pair aggregation, edge weights, baseline graph features).
3. Topology (sparsification, SCCs, cycle enumeration, directed flag persistence, landscapes).
4. Temporal motif (optional pump-and-dump).
5. Models (classification, clustering, regression, baselines).
6. Adaptive threshold (offline drift simulation).
7. Evaluation (temporal splits, metrics, ablations, sensitivity).

Figure plan: F1 pipeline diagram. F2 the three motifs (cyclic triangle, transitive triangle, cyclic triangle filled via a fourth account) with their persistence bars.

### 5.2 Graph construction and aggregation
All transactions from `a` to `b` in a window form one directed edge `a → b`. Aggregation gives `V_ab`, `n_ab` and the gap sequence. A reciprocal pair gives two distinct edges.

### 5.3 Edge suspicion scores and filtration weight
Each score lies in [0, 1] (higher = more suspicious). The proposal's formulas:

| Score | Definition | What it catches |
|---|---|---|
| Pair reciprocity `s_rec(a,b)` | `1 − |V_ab − V_ba| / (V_ab + V_ba)` | two-party ping-pong |
| Node flow balance `s_bal(a,b)` | `min(β(a), β(b))`, `β(x) = 1 − |in_x − out_x| / (in_x + out_x)` | multi-hop rings, where no pair is reciprocal but members net to about zero |
| Timing regularity `s_reg(a,b)` | `1 / (1 + CV)` of the pair's inter-transaction gaps | bots on a fixed cadence; bounded as CV → 0 |

**Edge cases.**
- Pairs with fewer than 3 transactions get the window median `s_reg`, so they are biased neither way.
- A pair with no reverse volume has `s_rec = 0`. This is intended, and multi-hop rings are covered by `s_bal`.

**Rank normalization.** Each score is replaced by its empirical rank within the window, `z = (rank_min − 1) / (m − 1)`. Ties take the lowest rank, so the large block of `s_rec = 0` edges maps to 0. No score dominates by scale, and outliers do not compress the range.

**Combination.** `w_e(α) = 1 − αᵀ z_e` with `α` on the probability simplex. The baseline is `α = (1/3, 1/3, 1/3)`. Lower weight means more suspicious. The filtration is a **sublevel-set filtration**: low-weight edges enter first, and a higher simplex enters at the maximum weight of its edges.

**Proposition 1 (the 1/3 floor).** In a ring of three or more accounts with no reciprocal pair, `z_rec = 0` on every edge, so `w_e(α) = 1 − α_bal z_bal − α_reg z_reg ≥ α_rec`. With equal weights, no multi-party ring can score below 1/3, however perfect its balance and timing. This motivates learning `α` (§5.8).

**Scope of the weights.** The scores encode wash-trading behaviour. Laundering cycles in IBM AML need not be reciprocal, balanced or regular (they are often one-directional, may decay in amount, and need not be regular in time). The weights are validated for wash trading on synthetic and NFT data. On IBM AML a pattern-behaviour audit (§6.5) is done first, and AML is used to validate the pipeline machinery, not the wash-trading weights.

### 5.4 Weighted directed flag complex and persistence
**Why not undirected.** In an undirected flag complex a 3-cycle A–B–C is filled by a 2-simplex when its last edge enters, so its H1 class has zero persistence. The canonical three-party ring would be invisible. Projecting to undirected also loses direction, reciprocity and flow balance.

**Definition.** A directed `k`-simplex is an ordered tuple `(v_0,…,v_k)` with an edge `v_i → v_j` for every `i < j`. We build the complex up to dimension 2 (2-simplices are needed for H1 classes to die). H2 is not used.

**Filtration.** `f(σ) = max_{e∈σ} w_e`, vertices at 0. `K_t = {σ : f(σ) ≤ t}`. The sublevel complexes are nested, so persistence is well defined. Homology is over Z/2, `H1(K_t) = ker ∂1 / im ∂2`. Pairs `(b, d)` are read from the reduced boundary matrix.

**Propositions (each is also an automated test).**
- **P2. A cyclic triangle is never filled.** A 2-simplex needs a vertex with edges to both others. In A → B → C → A every vertex has one outgoing edge inside the triangle. The loop persists: bar `(b, ∞)`.
- **P3. A transitive triangle dies at once.** For A → B, B → C, A → C, the ordering (A, B, C) is a 2-simplex entering with its last edge. Zero persistence.
- **P4. Filling through a fourth account.** Add D → A, D → B, D → C (weight 0.6) to a cyclic triangle (weight 0.1). Then (D,A,B), (D,B,C), (D,C,A) are 2-simplices, and over Z/2 the D-edges cancel in the boundary sum, leaving `AB + BC + CA`. The ring has bar `(0.1, 0.6)`. This is a case where homology carries information that cycle enumeration does not.
- **P5. A reciprocal pair** gives two edges between the same vertices and an H1 class representing ping-pong.

**Backends.** `flagser` via `pyflagser` (default, float32) and a pure-Python boundary-matrix reduction (`reference`) that must agree within tolerance. The reference backend builds the complex, sorts simplices by `(value, dimension)`, and reduces columns by symmetric difference over Z/2.

**Undirected comparison.** An undirected clique complex (`gudhi`) is kept as an ablation (RQ4).

### 5.5 Candidate ring generation (exact localization)
Persistence diagrams describe a whole complex. An H1 bar does not name the accounts responsible, and common libraries do not return full representative cycles (`ripser.py` returns cocycles, GUDHI's flag generators return only birth/death edges, `pyflagser` returns no generators). Localization is therefore combinatorial:
1. Sparsify: drop edges with weight above `τ_s` (hyperparameter, swept).
2. Compute SCCs; every directed cycle lies in one SCC.
3. Within each SCC, enumerate simple directed cycles of length 2–6 (`networkx.simple_cycles(G, length_bound=6)`).
4. De-duplicate cycles that share ≥ 80% of their edges.
5. Each survivor is a candidate ring: an explicit set of accounts, edges and transactions.

SCCs too large for enumeration are first split by raising `τ_s` locally; any that still exceed the budget are logged and reported, never silently dropped.

**Implementation deviation to disclose.** The proposal keeps the lowest-birth cycle among near-duplicates. With that rule a planted dense-neighbourhood ring could be replaced by a detour through a neighbour that shares 4/5 of its edges and is born slightly earlier. The code de-duplicates shortest-first, then by birth. A `k`-cycle shares at most `k−2` edges with any cycle of length ≤ `k`, so at overlap 0.8 nothing is dropped unless `max_len` ≥ 10. A different overlap measure is needed if real de-duplication is wanted.

**Two-stage detection.** Sparsification removes high-weight edges, so some true rings never become candidates, and it shapes which negatives exist. Candidate-generation recall and end-to-end precision/recall are therefore reported separately, across the `τ_s` sweep.

### 5.6 Features: B, D and T
What persistence can and cannot add: for an *isolated* ring, the H1 class is born at the ring's maximum edge weight and never dies, so persistence is nearly redundant with enumeration. It carries extra information only when the neighbourhood interacts with the loop (fill by a fourth account, chords that split a long ring, many incidental loops).

| Set | Content |
|---|---|
| **B** baseline graph | member degree and betweenness centrality, local clustering coefficient, edge-weight variance, neighbourhood density |
| **D** direction-aware and behavioural | **ring birth** (maximum edge weight on the ring), ring length, in/out-degree imbalance of members, mean and minimum pair reciprocity, mean node balance, volume and timing statistics |
| **T** topological | persistence landscapes of the local H1 diagram (ring plus 1-hop neighbourhood) and of the window, infinite-bar count, ablations (persistence images, scalar summaries: entropy, total persistence, H1 count) |

**Ring birth belongs to D, not T.** It is a plain function of the ring's own edge weights and needs no homology. Crediting it to T would overstate topology's contribution.

**Infinite bars.** Classes that never die get death = the window's maximum filtration value (a documented cap) so landscapes are finite. The number of infinite H1 bars is recorded separately, and sensitivity to the cap is checked. `[TODO-CODE: persistence landscapes, infinite-bar cap and the full feature table are listed as unfinished in the README status]`

**Landscapes.** `gudhi.representations.Landscape` with a fixed number of landscapes and resolution.

### 5.7 ML tasks
**Classification (T1).** Manipulation vs benign candidate ring. Logistic regression, random forest, SVM (RBF). Feature sets: B; B+D; B+D+T; T. **Primary comparison: B+D+T vs B+D** (isolates topology). Secondary: B+D vs B (isolates behavioural and direction features). Reported overall and stratified by synthetic scenario. A null topological lift is a legitimate, reportable result.

**Clustering (T2).** Persistence diagrams exist only for cycle candidates, so typing is among cycle-based patterns, not between wash trading and pump-and-dump. Ward hierarchical clustering on standardized landscape plus behavioural vectors, Euclidean distance. Diagrams are vectorized first, so no diagram-space (Wasserstein) distance is used. On IBM AML, evaluated against the labelled pattern types. On NFT data, inspected qualitatively (ping-pong, multi-hop rings, self-funded wallet clusters).

**Regression (T3).** The target must not be computable from the inputs. A ring's own current-window volume is an input, so predicting it is trivial. Targets:
- primary: the ring's wash volume in the **next** window;
- real data alternative: post-window price impact on the traded asset, where prices exist;
- synthetic: the planted inflated volume, with ring volume excluded from the inputs.

Models: ridge, random forest regressor, gradient boosting.

### 5.8 Learning the edge-weight combination (Problem B)
Instead of fixing `α = (1/3,1/3,1/3)`, learn it so wash rings get low birth values and legitimate loops get high values.

- **Smooth ring birth.** `b_R^β(α) = (1/β) log Σ_{e∈R} exp(β w_e(α))`, with `max_e w_e ≤ b ≤ max_e w_e + log|R|/β`. The softmax weights `π_{R,e}` concentrate on the ring's highest-weight edge (its weakest link).
- **Model.** `p_R = σ(c − k·b_R^β(α))`, offset `c`, slope `k > 0`.
- **Objective.** Class-balanced cross-entropy with `ω_R = N / (2 N_class)`:
  `min_{α,c,k} J = −(1/N) Σ ω_R [y log p + (1−y) log(1−p)]` subject to `α ≥ 0`, `1ᵀα = 1`.
- **Gradients.** With `r_R = ω_R (p_R − y_R)/N`: `∂J/∂c = Σ r_R`, `∂J/∂k = −Σ r_R b_R`, `∇_α J = k Σ_R r_R Σ_{e∈R} π_{R,e} z_e` (from `∂w_e/∂α = −z_e`).
- **Update.** `α ← Π_Δ(α − η ∇_α J)` with Euclidean projection onto the simplex, computed by the sort-based KKT solution: `α_j = max(u_j − ν, 0)`, `ν = (1/ρ)(Σ_{i≤ρ} u_(i) − 1)`, `ρ = max{j : u_(j) − (Σ_{i≤j} u_(i) − 1)/j > 0}`.
- **Optimality.** Stationarity on the simplex: `∂_j J = −ν` for used weights and `∂_j J ≥ −ν` for dropped ones (a KKT condition we check numerically).
- **Non-convexity.** Sigmoid ∘ log-sum-exp is not jointly convex. Gradient methods find a stationary point; restarts from equal weights and from each corner check sensitivity.

### 5.9 Problem A: a convex classifier solved by gradient descent
`p_R = σ(θᵀx_R)`, objective `J(θ) = −(1/N) Σ ω_R [ ... ] + (λ/2)‖θ‖²` (bias not penalized), gradient `(1/N) Xᵀ Ω (p − y) + λθ`. The Hessian `(1/N) Xᵀ Ω S X + λI ⪰ 0` is positive semi-definite, so every stationary point is global. Smoothness `L = λ_max(XᵀΩX)/(4N) + λ` since `p(1−p) ≤ 1/4`. Step theory: any `η < 2/L` decreases `J` (descent lemma), `η = 1/L` gives the largest guaranteed decrease, and with strong convexity convergence is linear. A step matched to the local curvature `L_local` at the optimum is much larger than `1/L` because most rings are classified confidently (`p(1−p) ≈ 0`). `[RESULT: step-size comparison table]`

### 5.10 Adaptive threshold (Problem C)
**Online gradient ascent (primary method in the optimization report).**
- Window `t`, candidate `i`, score `s_i = b_i`; flag if `s_i ≤ τ`.
- Reward `R_t(τ) = Σ g_i 1[s_i ≤ τ]`, `g_i = y_i − λ(1−y_i)`, i.e. `TP − λ·FP`. It is a step function of `τ`, so its gradient is zero almost everywhere.
- Smoothing: `R_t(τ) = Σ g_i σ((τ − s_i)/T)`, `dR/dτ = (1/T) Σ g_i σ_i(1−σ_i)`. Only rings with scores within a few `T` of the threshold contribute.
- Optimality: as `T → 0`, stationarity becomes `n₊ f₊(τ*) = λ n₋ f₋(τ*)`, the Bayes rule written as a gradient condition.
- Update with labels `d` windows late: `τ_{t+1} = Π_[0,1]( τ_t + η (1/n_{t−d}) dR_{t−d}/dτ (τ_t) )`.

**Bandit formulation (from the proposal, §4.5).** Arms = K discrete threshold levels (e.g. 10). Policies: discounted UCB and sliding-window Thompson sampling (directly or via `river.bandit`). Reward `TP − λ·FP`. Lag `d`. `[TODO-CODE: the implemented policy is online gradient ascent; the bandit variants are in the proposal only. Choose which the paper reports, or include both.]`

**Comparators.**
1. *Fixed threshold*: best on the pre-drift period (weak baseline).
2. *Periodically re-tuned* every `M` windows, with the **same label lag `d`** (primary comparison, no information advantage).
3. *Hindsight oracle*: best threshold per window with full labels (an upper bound, not achievable).

**Drift model.** The generator shifts ring size, timing regularity, jitter, amount noise and decoy-trade rate mid-stream.

### 5.11 Pump-and-dump temporal-motif track (optional)
**Status: done as a synthetic-only, no-topology exploratory track** (own generator `data/pump_dump.py`, features `motif/pump_dump.py`, run `experiments/pump_dump_run.py`; results in `paper/numbers_pack.md` section 6F). The generator author decides separability, detection is retrospective, and no real-data result exists. The text below is the original design.

Not a closed cycle, so no topology stage. Candidates are coordinated-accumulation windows (multiple accounts raising position in the same asset in a short band) followed by coordinated-exit windows, scored by accumulation concentration, exit synchrony and price trajectory around the exit. It shares the classification harness but not the features. Attempted only after the main phases meet their gates.

### 5.12 Complexity and feasibility
- Cycle enumeration inside SCCs is the dominant cost, bounded by the length limit 6 and per-SCC time budgets.
- Persistence is computed on candidate neighbourhoods, not the whole graph, with the complex capped at dimension 2.
- Window-level persistence uses the sparsified graph only.
- `[RESULT: runtime per window and per SCC size]`

### 5.13 How this differs from existing solutions
| Aspect | Existing approach | This work |
|---|---|---|
| Target structure | dense clusters | sparse directed loops |
| Ring score | binary (cycle exists) or learned black-box score | continuous filtration birth plus persistence, interpretable through edge weights |
| Direction | often projected to undirected | encoded topologically through the directed flag complex |
| Legitimate loops | not separated by shape | separated by behaviour on edges (balance, regularity) inside the filtration |
| Localization | none (global TDA) or by cycle listing only | exact: SCC plus bounded enumeration, with persistence as features |
| Topology's contribution | assumed | tested with B+D+T vs B+D, scenario-stratified, ring birth credited to D |
| Weights | hand-set | learned on the simplex by projected gradient descent, with KKT check |
| Threshold | fixed, tuned once | tracked online under drift with delayed labels, compared with re-tuned and oracle policies |
| Learning | often deep | classical, interpretable, no deep learning |

---

## 6. Experimental Design (protocol only, no results)

### 6.1 Data sources
1. **Synthetic generator** (unit tests, ablations, drift). Background directed graphs with heavy-tailed degree, plus:
   - *Positives (planted wash rings):* isolated 2-, 3- and 5-party balanced regular rings; rings in dense neighbourhoods; chorded longer rings.
   - *Topology-sensitive negatives:* transitive triangles; shortcut-filled pseudo-rings (cyclic structure filled through extra accounts, e.g. D → A, D → B, D → C).
   - *Hard negatives:* market makers (many counterparties, two-sided, fairly regular, inventory drifting over longer horizons); arbitrage bots (loops across venues, timing triggered by price gaps, net profit per cycle); exchange hot wallets (high-degree hubs, near-balanced aggregate flow).
   - *Evasion knobs:* `wash_decoy_frac`, `mm_regularity`; `hard_negatives=False` gives the "easy background".
   - Hard-negative behaviour is modelled on documented properties of the actors, not invented to be separable. Results are reported with and without them.
2. **IBM AML (AMLworld; HI-Small / LI-Small first)** `[VERIFY]`. Synthetic financial transactions with laundering attempts labelled by pattern type. Gives exact ground truth for localization, a labelled typing task, and labels independent of the baseline features. It validates machinery, not wash-trading weights.
3. **Ethereum NFT trades.** From Dune exports or Google BigQuery `crypto_ethereum` (per-address Etherscan queries are rate-limited and impractical for full-graph construction). Proxy labels from published wash-trade datasets or the Dune Spellbook `nft.wash_trades` model, subject to the provenance audit (§6.7).

### 6.2 Ring-level labelling
**IBM AML (transaction labels → ring labels).** A candidate is
- **positive** if ≥ X% of its edges (default X = 80, swept 50–100) are backed by transactions of a *single* laundering attempt, and it inherits that attempt's type;
- **negative** if none of its edges are backed by laundering transactions;
- **ambiguous** otherwise (partial overlap, or several attempts): logged, counted and excluded from training. Results are also reported treating ambiguous candidates as positives (and, in the optimization experiments, as negatives).

If attempt grouping is unavailable, the fallback is ≥ X% of edges backed by at least one laundering-flagged transaction, with the same ambiguity handling.

**Synthetic.** The same rule with the planted structure playing the role of the attempt: the structure's kind if ≥ X% of the ring's edges are backed by its own transactions, `background` if none, else `ambiguous` (`src/ringtda/eval/labels.py`). For planted rings of size 2/3/5 with `max_len` ≤ 6 and for star-shaped structures, X = 80% and X = 100% give identical labels. The exception is structures with more edges than one cycle, such as chorded 6-rings.

### 6.3 Splits
Sliding windows overlap, so a ring appears in neighbouring windows and random splits leak. All supervised results use a **temporal split**: train on earlier windows, test on later ones, with a gap of one window length. Grouped cross-validation by ring identity is used for model selection inside the training period.

### 6.4 Baselines and ablations
- **Classifier baselines:** B only; B+D; B+D+T; T only. Models: logistic regression, random forest, RBF-SVM.
- **Filtration ablations:** each component score alone (`s_rec`, `s_bal`, `s_reg`), equal-weight combination, learned `α`.
- **Complex ablation:** directed flag complex vs undirected clique complex (`gudhi`).
- **Vectorization ablations:** landscapes vs persistence images vs scalar summaries.
- **Infinite-bar cap sensitivity.**
- **Cycle-only baseline:** enumeration with ring birth as the score, to quantify what persistence adds.
- **Sensitivity sweeps:** window length, stride, `τ_s`, maximum cycle length, infinite-bar cap, labelling threshold X. Chosen values are justified on the training period only.

### 6.5 IBM AML pattern-behaviour audit
Before using the wash-trading weights on IBM AML, measure for each labelled pattern type the distribution of `s_rec`, `s_bal`, `s_reg` and the combined weight against background transactions. Expectation to check, not assume: pass-through accounts in laundering cycles may be near-balanced (so `s_bal` may transfer), while reciprocity and timing regularity likely do not. If needed, an AML-specific variant (e.g. balance only) is added as an ablation.

### 6.6 Metrics
| Task | Metrics |
|---|---|
| Candidate generation | candidate recall across the `τ_s` sweep (ground-truth rings recovered) |
| Classification | precision, recall, F1, PR-AUC (preferred under heavy imbalance); overall, per scenario, with/without hard negatives |
| Clustering | adjusted Rand index, purity, silhouette; qualitative inspection on NFT |
| Regression | RMSE, MAE, Spearman against baseline-feature regressors |
| Optimization | optimality gap `J(θ_t) − J*`, iterations to tolerance, gradient finite-difference error, KKT residuals |
| Adaptive threshold | rolling precision/recall, cumulative reward `TP − λ·FP`, share of oracle reward, variance over time |

**Statistical reporting.** Mean ± std over ≥ 5 seeds (synthetic) or bootstrap confidence intervals (real data). No model-performance comparison is a pass/fail gate.

### 6.7 Threats to validity (to carry into the paper)
1. **Directed vs undirected.** Quantified by the complex ablation.
2. **Ground-truth quality.** IBM labels are exact but synthetic; NFT labels are real but heuristic. Each is reported separately.
3. **Label leakage in both directions.** If NFT proxy heuristics resemble the density/centrality baseline, the baseline partly predicts its own label rule. If they include cycle rules (e.g. back-and-forth trading between the same wallets), the topological method may predict its own label rule and inflate its lift. Before NFT results are reported, the label logic of each source is inspected and disclosed, and results are also reported on the subset of rings not flagged by any cycle-based rule. IBM AML results are the primary evidence.
4. **Computational feasibility.** Mitigated by sparsification, local persistence, dimension cap and enumeration budgets with logged fallbacks.
5. **Reward design for adaptive thresholds.** Real confirmed catches are delayed and censored; this work uses an offline drift simulation. Live deployment is future work.
6. **Ring-type coverage.** Cycle-based wash and laundering patterns only; spoofing excluded.
7. **Hyperparameter sensitivity.** Each hyperparameter is swept.
8. **Topological lift may be small or null.** Ring birth is counted in D; the primary test is B+D+T vs B+D; results are stratified. If T adds nothing, that is reported, and it would itself suggest combinatorial cycle features suffice.
9. **Hard-negative realism and generator circularity.** The generator's designer decides what separates hard negatives from wash rings. Mitigated by modelling documented behaviour, reporting with/without, and using NFT data as the external check.
10. **Weight transfer from wash trading to laundering.** Addressed by the audit (§6.5).

### 6.8 Reproducibility
- Conda env `ringtda`, Python 3.11 (`pyflagser` has no wheels for 3.13). Install with `pip install -e ".[tda,dev]"`; `gudhi` is used for landscapes and the undirected ablation.
- Automated tests are the engineering gates: planted rings are candidates at the default `τ_s`; planted cyclic triangles give persistent H1; transitive triangles do not; a cyclic triangle filled via a fourth account dies; `flagser` and `reference` backends agree within tolerance; finite-difference checks for every gradient; simplex projection feasibility and optimality; GD against SciPy L-BFGS on the same objective.
- Optimization report numbers come from `experiments/gd_report.py --config report`. Experiments on the full generator go to `--config full`.
- Seeds, window configuration and generator configuration are stored with each result file.

---

## 7. Planned Tables and Figures

| ID | Type | Content | Status |
|---|---|---|---|
| F1 | diagram | pipeline overview (§5.1) | to draw |
| F2 | diagram | cyclic / transitive / shortcut-filled triangles and their bars | to draw (a `viz.py` helper and the notebook already exist) |
| F3 | plot | filtration birth by loop kind under equal vs learned weights | `[RESULT]` |
| F4 | plot | PGD paths on the simplex from several starts, with loss curves | `[RESULT]` |
| F5 | plot | GD optimality gap vs iterations for several step sizes | `[RESULT]` |
| F6 | plot | PR curves per feature set (B, B+D, B+D+T, T) | `[RESULT]` |
| F7 | plot | topological lift per synthetic scenario with CIs | `[RESULT]` |
| F8 | plot | candidate recall vs `τ_s` | `[RESULT]` |
| F9 | plot | threshold trajectories and cumulative reward under drift | `[RESULT]` |
| F10 | plot | dendrogram or embedding of ring types vs IBM pattern types | `[RESULT]` |
| T1 | table | notation (§8) | below |
| T2 | table | dataset statistics (windows, candidates, positives per type) | `[RESULT]` |
| T3 | table | gap analysis vs related work (§4.5) | done, needs citation check |
| T4 | table | main classification results by feature set and model | `[RESULT]` |
| T5 | table | ablations (components, complex, vectorization, cap) | `[RESULT]` |
| T6 | table | clustering ARI / purity | `[RESULT]` |
| T7 | table | severity regression | `[RESULT]` |
| T8 | table | adaptive-threshold comparison | `[RESULT]` |
| T9 | table | hyperparameter sensitivity | `[RESULT]` |

---

## 8. Notation

| Symbol | Meaning |
|---|---|
| `(t_i,u_i,v_i,a_i)` | transaction: time, sender, receiver, amount |
| `L, S` | window length, stride |
| `W_k, G_k=(V_k,E_k)` | window `k` and its directed graph |
| `V_ab, n_ab, g_j` | pair volume, count, inter-transaction gaps |
| `s_rec, s_bal, s_reg` | reciprocity, node balance, timing regularity scores |
| `z_e ∈ [0,1]³` | rank-normalized score vector of edge `e` |
| `α ∈ Δ` | combination weights on the probability simplex |
| `w_e(α) = 1 − αᵀz_e` | edge weight (low = suspicious) |
| `f(σ) = max_{e∈σ} w_e` | filtration value of simplex `σ` |
| `K_t` | sublevel complex at level `t` |
| `H1`, `(b,d)` | first homology, persistence pair |
| `b_R`, `b_R^β` | ring birth and its log-sum-exp smoothing |
| `τ_s` | sparsification cutoff |
| `τ_t` | alarm threshold at window `t` |
| `ω_R` | class-balancing weight |
| `L, L_local` | global and local smoothness constants of the logistic loss |
| `λ` | L2 penalty in Problem A; false-alarm cost in Problem C (distinguish in text) |
| `d`, `T`, `η` | label delay, smoothing temperature, step size |
| `B, D, T` | baseline, direction-aware/behavioural, topological feature sets |

Note: the optimization report reuses `λ` for two things, and `T` for both a feature set and a temperature. Rename one of each in the paper.

---

## 9. Discussion Material (to complete after experiments)
Prompts only; no claims yet.
- Where does topology help, and does it help only in dense, chorded or shortcut-filled scenarios? `[RESULT]`
- Which component score does the work, and does the learned `α` agree with the ablations? `[RESULT]`
- Does the directed complex beat the undirected one, and by how much? `[RESULT]`
- How far do synthetic findings transfer to IBM AML and to NFT data, and what do the provenance subsets show? `[RESULT]`
- What does the drift study say about when online adaptation matters, and when robust training already suffices? `[RESULT]`
- Cost of the pipeline against the benefit. `[RESULT]`

### 9.1 Limitations (stated up front in the design; keep in the paper)
- Rings longer than 6 are missed by design (length bound is swept).
- Localization is combinatorial; persistence supplies features. Very large SCCs may need local re-sparsification.
- Weights rely on behavioural heuristics. Adversaries aware of them can inject decoy trades; this is tested partly through decoy-rate drift.
- IBM AML is synthetic. NFT labels are heuristic and may share logic with either feature family.
- The adaptive threshold is evaluated offline only.
- For isolated rings, topology is expected to be largely redundant with enumeration.
- Synthetic hard negatives are modelled, not observed.
- Sparsification bounds recall at the candidate stage.
- Pump-and-dump detection is optional and non-topological; spoofing is out of scope.

### 9.2 Ethics and responsible use
Flagged rings are suspicion scores, not findings of guilt. Proxy labels are not regulator-confirmed. Describing the scores publicly lets adversaries tune to them; the limitations section notes this and the decoy-rate drift experiments address part of it. Add a data-use statement for NFT and AML data. `[VERIFY dataset licences]`

### 9.3 Future work
Live deployment with real delayed feedback; real-data pump-and-dump validation (a synthetic-only track is done); cap-free or alternative treatments of infinite bars; larger ring lengths via local re-sparsification; richer filtrations (multi-parameter); real exchange data with regulator-confirmed labels.

---

## 10. Suggested Paper Outline (section to content map)
1. Introduction → §2
2. Problem statement → §3
3. Related work → §4
4. Method → §5 (5.2–5.5 graph and topology, 5.6–5.7 features and ML, 5.8–5.10 optimization)
5. Experimental setup → §6
6. Results → *after experiments*
7. Discussion, limitations, ethics, future work → §9
8. Conclusion → one paragraph restating the contributions and the headline finding `[RESULT]`
9. Appendices: proofs / worked examples of P1–P4 and the simplex projection (worked example from the report: `u = (0.5, 0.8, −0.1)` projects to `(0.35, 0.65, 0)`); generator parameters; full hyperparameter tables; pseudocode for cycle enumeration, persistence reduction, PGD and threshold ascent.

---

## 11. Gaps to close before writing the final paper
1. **Citations.** Every `[VERIFY]` item. The related-work table needs a real literature search, including the novelty claim for directed flag complexes in transaction graphs.
2. **Unfinished code** (README status): the NFT loader and label-provenance audit (Phase 5, not done: running Dune queries is blocked on the free tier), the undirected-complex and persistence-image ablations, and per-score classifier filtrations. Done since this pack was drafted: landscapes, the infinite-bar cap and its sensitivity run, the full feature table, the IBM AML loader and audit, the evaluation harness, clustering (exploratory), severity regression (synthetic only), the bandit policies and the synthetic-only pump-and-dump track.
3. **Choice made:** the adaptive-threshold section reports both online gradient ascent and the discounted-UCB / sliding-window Thompson bandits; at the primary setting the bandits lose to the re-tuned comparator (`paper/numbers_pack.md` section 6A).
4. **Sections of the optimization report not yet read in full:** the tail of §4.5 onward and any appendix. Check them for claims to carry over.
5. **Notation clashes** (§8) to resolve.
6. **Real-data experiments** exist for IBM AML only (HI-Small, LI-Small, transfer; `paper/numbers_pack.md`). The NFT experiment was not run, so external-validity claims beyond IBM AML are not supported.
