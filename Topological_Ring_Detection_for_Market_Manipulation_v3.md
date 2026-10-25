# Topological Ring Detection for Market Manipulation

A graph + topological-data-analysis (TDA) framework for detecting, typing, and scoring coordinated manipulation rings (primarily wash trading, optionally pump-and-dump coordination) in transaction networks — with clustering, classification, regression, and an adaptive detection threshold, built entirely on classical (non-deep-learning) mathematics.

*Revision 3 — changes from v1 → v2 are summarized in Appendix A; changes from v2 → v3 in Appendix B.*

---

## 1. Problem

Market manipulation schemes such as **wash trading** and **coordinated pump-and-dump rings** rarely look anomalous at the level of a single transaction. What gives them away is the *pattern of connections* between accounts: money or assets cycling through a small group of colluding participants (A → B → C → A) to fabricate volume, mislead other traders, or launder value with no real economic activity behind it.

Two concrete gaps motivate this project:

1. **Density-based fraud detection misses cycles.** The dominant approach to graph-based fraud/AML detection — community detection, centrality scores, dense-subgraph mining, GNN classifiers — is built to find *tightly clustered* groups of accounts. A manipulation ring can be structurally sparse (each account transacts with only 2–3 others) while still forming a persistent loop. Cycle-based methods exist (SCC detection, cycle enumeration) but are typically binary/combinatorial: a cycle either is or isn't flagged, with no notion of *how strong, stable, or severe* it is, and little ability to distinguish ring *types*.

2. **Fixed detection thresholds decay.** Manipulation tactics evolve — ring size, timing patterns, and the use of decoy legitimate trades shift as bad actors adapt. A threshold tuned once on historical data degrades as tactics drift.

Prior work has applied **persistent homology** to blockchain transaction networks for general structural anomaly detection (e.g., clique persistent homology on dynamic multilayer blockchain networks). This confirms the approach is viable, but that line of work stops at flagging *that* the network's shape changed — it does not isolate individual rings, characterize ring *types*, or produce a continuous severity estimate. That gap is the target of this project.

**Scope note.** Two related but structurally different patterns:

1. **Closed-cycle wash trading** — repeated or near-repeated directed cycles (A ⇄ B, A → B → C → A, …) with suspicious behavioral attributes. This is the **primary, load-bearing target** and the one the topological pipeline (§3.1–3.4) is designed for.
2. **Pump-and-dump coordination** — a **temporal coordination motif** (shared accumulation followed by a shared exit), not a closed cycle. It is detected with time-series features (§3.6), not persistent homology, and is an **optional stretch track**, deferred to future work if the timeline is constrained.

**Spoofing is excluded.** It is an order-book phenomenon (orders placed and cancelled without executing); settled transaction data contains no signal for it.

---

## 2. Proposed Solution

One pipeline, around one windowed directed graph, performing three coupled ML tasks with topological structure as a core feature source — plus an adaptive layer that keeps the detection threshold current.

**Core idea.** A coordinated manipulation ring is a **directed loop** in the transaction graph. Persistent homology's H1 features measure how strong and stable a loop is across a filtration. Crucially, the complex is a **directed flag complex**, in which a *circular* triangle (A → B → C → A) is never filled and therefore persists as an H1 class, while a *transitive* triangle (A → B, B → C, A → C — one-way flow, not a loop) is filled and dies. This directly encodes the distinction between circulating value and ordinary flow (§3.2).

### Pipeline sketch

1. **Graph construction** — directed transaction graph per sliding time window (nodes = accounts/wallets; one edge per ordered account pair, aggregating all trades in the window).
2. **Edge weighting** — a suspiciousness-based filtration weight per edge (§3.1).
3. **Candidate ring generation** — enumerate bounded-length directed simple cycles inside strongly connected components; each cycle is an exact, account-level candidate ring (§3.3).
4. **Topological feature extraction** — weighted directed-flag persistent homology on each candidate's local neighborhood and on the whole window; vectorize into persistence landscapes (§3.2, §3.4).
5. **Clustering** — group candidate rings by topological + behavioral features to test whether distinct ring *types* have distinguishable fingerprints (§3.5).
6. **Classification** — manipulation vs. benign cycle, compared against a density/centrality baseline.
7. **Regression** — a continuous, non-leaky severity target (§3.5).
8. **Adaptive thresholding** — a discounted bandit tunes the flagging threshold under simulated tactic drift, evaluated offline (§4.5).

---

## 3. Methodology

### 3.1 Edge weighting and filtration (the central design decision)

The filtration determines what "persistence" means, and it is what separates a wash-trading loop from a structurally identical legitimate arbitrage loop. The two can share graph shape; what differs is the *behavior along the edges*.

**Aggregation.** Edge weights are computed **per ordered account pair per window**: all transactions from *a* to *b* in a window form one directed edge *a → b*.

**Component scores** (each in [0, 1], higher = more suspicious):

| Score | Definition | Rationale |
|---|---|---|
| Pair reciprocity `s_rec(a,b)` | `1 − |V_ab − V_ba| / (V_ab + V_ba)` | Catches two-party ping-pong (A ⇄ B) |
| Node flow balance `s_bal(a,b)` | `min(β(a), β(b))`, where `β(x) = 1 − |in_x − out_x| / (in_x + out_x)` | Catches multi-hop rings: participants in A → B → C → A end each window near net-zero even though no *pair* is reciprocal |
| Timing regularity `s_reg(a,b)` | `1 / (1 + CV)` of inter-transaction gaps for the pair | Bounded even for perfectly regular bots (CV → 0) |

Pair reciprocity alone would score every edge of a multi-hop ring as benign (no pair trades in both directions), so the node-balance term is required for the primary target, not optional.

**Normalization.** Each score is rank-normalized (empirical CDF) within the window before combining, so no score dominates by scale and outliers do not compress the rest of the range.

**Combination.** `w(a,b) = 1 − mean(s_rec, s_bal, s_reg)`. **Lower weight = more suspicious.** The complex uses a **sublevel-set filtration**: low-weight edges enter first; a higher simplex enters at the maximum weight of its edges.

**Edge cases.**
- A pair with a single transaction has undefined timing regularity; it receives the window's median `s_reg`, biasing it neither way.
- A pair with no reverse volume has `s_rec = 0` by definition; this is intended, and multi-hop rings are captured through `s_bal`.

**Hypothesis (to validate, not assume).** Wash-trading loops enter the filtration early and persist, because their edges score high on balance and regularity, and because circular triangles are never filled in the directed flag complex. This is tested on planted synthetic rings (Phase 2 gate) before any real data is trusted.

**Scope of the weights.** These scores encode *wash-trading* behavior (reciprocity, net-zero balance, regular cadence). Laundering cycles in IBM AML need not share these properties — they are typically one-directional, may decay in amount along the path (fees/skimming), and need not be regular in time. The weights are therefore validated for wash trading on the synthetic generator and NFT data; on IBM AML, the actual behavior of planted patterns under each score is measured first (Phase 3), and IBM AML is used to validate the *pipeline machinery* (localization, typing, topological lift), not the wash-trading weights themselves (§4.10).

**Ablations.** Filtrations using each component score alone, and the combined weight, are compared in Phase 3 to isolate which signal does the work.

### 3.2 Complex construction: weighted directed flag complex

**Why not an undirected clique complex.** In an undirected flag complex, a 3-cycle A–B–C is filled by a 2-simplex at exactly the filtration value where its last edge enters, so its H1 class has **zero persistence**. The canonical three-party wash ring would be invisible; only chordless cycles of length ≥ 4 would register. Projecting to undirected also discards direction, reciprocity, and flow balance.

**Directed flag complex.** A directed *k*-simplex is an ordered tuple (v₀, …, v_k) with an edge vᵢ → vⱼ for every i < j. Consequences:
- A **transitive** triangle (A → B, B → C, A → C) is a 2-simplex → its loop is filled → no lasting H1.
- A **cyclic** triangle (A → B → C → A) admits no such ordering → never filled → lasting H1 class.
- A **reciprocal pair** (A → B and B → A) forms two distinct 1-simplices between the same vertices → an H1 class representing ping-pong trading.

**Library.** `pyflagser` (`flagser_weighted`, max filtration), which computes persistent homology of weighted directed flag complexes. The complex is built up to **dimension 2** (2-simplices are needed for H1 classes to die); H2 is not used.

**Undirected comparison.** An undirected clique complex (`gudhi`) is retained as an **ablation** (§4.1) to measure how much the directed construction contributes.

### 3.3 Candidate ring generation (localization)

A persistence diagram describes the whole complex; an H1 bar does not by itself name the accounts responsible, and common libraries do not return full representative cycles (`ripser.py` returns cocycles; GUDHI's flag generators return only birth/death edges; `pyflagser` returns no generators). Localization is therefore done combinatorially, which is exact and cheap at this scale:

1. Sparsify the window graph: drop edges with weight above a sparsification cutoff `τ_s` (a hyperparameter; reported in a sensitivity sweep).
2. Compute strongly connected components (SCCs); any directed cycle lies entirely within one SCC.
3. Within each SCC, enumerate simple directed cycles of length 2–6 (`networkx.simple_cycles(G, length_bound=6)`).
4. Deduplicate cycles that share ≥ 80% of their edges (keep the lowest-birth one) to avoid near-duplicate candidates from dense SCCs.
5. Each surviving cycle is a **candidate ring**: an explicit set of accounts, edges, and underlying transactions.

If an SCC is too large for enumeration within a time budget, it is first split by raising `τ_s` locally; any SCC still exceeding the budget is logged and reported rather than silently dropped.

**Two-stage detection.** Sparsification removes high-weight edges, so some true rings never become candidates; it also shapes which negatives exist, and therefore the downstream class balance and difficulty. Detection is consequently a two-stage process, and the two stages are evaluated separately (§3.8): **candidate-generation recall** (fraction of ground-truth rings recovered as candidates) and **end-to-end precision/recall** (after classification), both reported across the `τ_s` sweep.

### 3.4 Topological features and vectorization

**What persistence can and cannot add.** For an *isolated* cyclic ring, persistence is nearly redundant with cycle enumeration: the H1 class is born at the ring's maximum edge weight and, with nothing to fill it, never dies. Persistence carries information beyond enumeration only when the **neighborhood interacts with the loop** — e.g., a fourth account D with D → A, D → B, D → C forms transitive triangles whose boundaries fill the cyclic triangle A → B → C → A and kill its class; chords split a long ring into shorter classes; many overlapping incidental loops in a dense neighborhood change the diagram's shape. The topological feature set is designed around these cases, and the synthetic generator plants them explicitly (§3.7, Phase 1) so the method is tested where topology should matter, not only where it is redundant.

For each candidate ring:

- **Local persistence.** Take the ring plus its 1-hop neighborhood; compute weighted directed-flag persistence (H0, H1). This captures whether the ring is an isolated persistent loop, a loop filled or split by neighboring structure, or one of many incidental loops in a dense neighborhood.
- **Infinite bars.** Classes that never die are assigned death = the maximum filtration value of the window (a fixed, documented cap) so that landscapes are finite. Because this cap determines how much signal infinite bars carry, the number of infinite H1 bars is also recorded separately, and sensitivity to the cap is checked.
- **Persistence landscapes** (`gudhi.representations.Landscape`, fixed number of landscapes and resolution) of the local H1 diagram — the primary vectorization.
- **Ablations:** persistence images, and scalar summaries (persistence entropy, total persistence, H1 count).

Window-level persistence (whole window graph) is also computed and used as context features and for the drift analysis in Phase 6.

**Direction-aware and behavioral features (D)** (computed for every candidate): **ring birth** (the maximum edge weight on the ring — the filtration value at which the loop first exists), ring length, in/out-degree imbalance of members, mean/min pair reciprocity, mean node balance, volume and timing statistics. Ring birth is deliberately placed in D, not T: it is a plain function of the ring's own edge weights, obtainable without any homology computation, so crediting it to the topological set would overstate T's contribution.

**Baseline graph features (B):** member degree and betweenness centrality, local clustering coefficient, edge-weight variance, neighborhood density.

**Topological features (T):** local and window-level persistence landscapes, infinite-bar count, and the scalar/image ablations above.

### 3.5 ML tasks

**Classification.** Manipulation vs. benign candidate ring. Classical models: logistic regression, random forest, SVM (RBF). Feature sets compared:
- (B) baseline density/centrality features only
- (B+D) baseline + direction-aware/behavioral features
- (B+D+T) baseline + direction-aware/behavioral + topological features
- (T) topological features only

The **primary comparison is B+D+T vs. B+D**, which isolates the contribution of topology. B+D vs. B separately measures the contribution of the behavioral/direction features. This comparison is reported overall and **stratified by synthetic scenario** (isolated rings vs. dense-neighborhood, chorded, and shortcut-filled rings; §3.7), since topology is expected to help mainly in the latter. A null topological lift is a legitimate, reportable result, not a failure to be tuned away.

**Clustering (ring typing).** Persistence diagrams exist only for cycle candidates, so typing is done *among cycle-based patterns*, not between wash trading and pump-and-dump. On IBM AML, clustering is evaluated against the dataset's labelled laundering pattern types; on NFT data, clusters are inspected qualitatively (e.g., two-party ping-pong vs. multi-hop rings vs. self-funded wallet clusters). Method: hierarchical clustering (Ward) on standardized landscape + behavioral vectors, Euclidean distance. Diagrams are vectorized before clustering, so no diagram-space (Wasserstein) distance is used.

**Regression (severity).** The target must not be computable directly from the input features — a ring's own volume in the current window is an input, so predicting it is trivial. Targets:
- *Primary:* the ring's wash volume in the **next** window (forward-looking persistence of the scheme).
- *Real-data alternative:* post-window price impact on the traded asset/collection, where price data exist.
- *Synthetic:* the planted inflated volume, with ring volume excluded from the inputs.

Models: ridge regression, random forest regressor, gradient boosting (scikit-learn).

### 3.6 Pump-and-dump temporal-motif track (optional)

Pump-and-dump coordination is not assumed to form a closed cycle and does not use the topology stage. Candidates are coordinated-accumulation windows (multiple accounts increasing position in the same asset within a short band) followed by coordinated-exit windows, scored with accumulation concentration, exit synchrony, and price trajectory around the exit. It shares the classification harness but not the features. **This track is attempted only after Phases 1–6 meet their gates.**

### 3.7 Phased plan

Each phase ends in one of two kinds of checkpoint:

- **Engineering gates (hard pass/fail).** Correctness tests of the pipeline — e.g., planted rings are recovered, transitive triangles are filled. These must pass before later phases rely on them; failing means a bug, not a finding.
- **Reporting checkpoints (no pass/fail).** Scientific results — effect sizes with confidence intervals. These are *reported whatever their value*. Making a performance comparison a gate that "must pass" invites tuning until it does, so no model-performance result is a gate.

| Phase | Weeks* | Work | Checkpoint |
|---|---|---|---|
| 1. Synthetic data | 1–2 | Generator (scenarios below); window + weight code | **Gate:** planted wash-ring edges rank in the lowest-weight decile against easy background |
| 2. Topology pipeline | 3–4 | SCC + cycle enumeration; weighted directed-flag persistence; landscapes | **Gate:** every planted ring is a candidate at default `τ_s`; planted cyclic triangles yield persistent H1, transitive triangles do not; a cyclic triangle filled via a fourth account (shortcut case) dies (automated tests) |
| 3. IBM AML development | 5–6 | Pattern-behavior audit and ring-labeling rule (below); full features; classifiers B / B+D / B+D+T / T; all ablations | **Report:** B+D+T vs. B+D (primary) and B+D vs. B, PR-AUC with bootstrap CIs, temporal split; candidate-generation recall |
| 4. Typing + severity | 7 | Clustering vs. IBM pattern types; next-window severity regression | **Report:** ARI/purity and RMSE with CIs vs. baseline-feature equivalents |
| 5. NFT transfer | 8–9 | Ethereum NFT trades, proxy labels, label-provenance audit (§4.3) | **Report:** results with provenance caveats |
| 6. Adaptive threshold | 10 | Offline drift simulation (§4.5) | **Report:** bandit vs. fixed, periodically re-tuned, and hindsight-oracle thresholds |
| 7. Stretch + write-up | 11–12 | Pump-and-dump track if time permits; report | — |

\*Assumes a ~12-week semester; adjust proportionally.

**Synthetic generator scenarios.** Background graphs (random directed graphs with heavy-tailed degree) plus:

*Positives (planted wash rings):*
- **Isolated rings** — 2-, 3-, and 5-party balanced, regular rings in sparse neighborhoods (the case where topology is expected to be nearly redundant with enumeration).
- **Rings in dense neighborhoods** — the same rings embedded among many incidental loops.
- **Chorded rings** — longer rings with internal chords that split the loop into shorter cycles.

*Topology-sensitive negatives:*
- **Transitive triangles** — one-way flow, filled in the directed flag complex.
- **Shortcut-filled pseudo-rings** — cyclic structures filled by transitive triangles through additional accounts (e.g., D → A, D → B, D → C), ring-like under enumeration but bounded in homology.

*Hard negatives (legitimate loops that are balanced and/or regular):*
- **Market makers** — trade with many counterparties, two-sided and fairly regular, but with inventory that drifts over longer horizons rather than netting to zero within each window.
- **Arbitrage bots** — loops across venues/assets, timing triggered by simulated price gaps rather than a fixed clock, with net profit (not net zero) per cycle.
- **Exchange hot wallets** — very high-degree hubs with near-balanced aggregate flow.

Because the generator's designer also decides what separates hard negatives from wash rings, hard-negative behavior is modeled on documented properties of these actors rather than invented to be separable, and results are reported **with and without hard negatives** (§4.9). NFT data, which contains real legitimate loops, is the external check on this.

**IBM AML pattern-behavior audit (start of Phase 3).** Before using the wash-trading weights on IBM AML, measure for each labelled pattern type the distribution of each component score (`s_rec`, `s_bal`, `s_reg`) and of combined weight, against background transactions. Expectation to check, not assume: pass-through accounts in laundering cycles may be near-balanced (so `s_bal` may transfer), while reciprocity and timing regularity likely do not. Results are reported and, if needed, an AML-specific weight variant (e.g., balance-only) is used as an additional ablation.

**IBM AML ring-labeling rule.** IBM AML labels are per transaction, while candidates are rings. Where the dataset's patterns file groups laundering transactions into attempts with a pattern type (to be confirmed on inspection of the file format), a candidate ring is:
- **positive** if ≥ X% of its edges (default X = 80, swept over 50–100) are backed by transactions belonging to a **single** laundering attempt, and it inherits that attempt's pattern type;
- **negative** if none of its edges are backed by laundering transactions;
- **ambiguous** otherwise (partial overlap, or edges from multiple attempts) — logged, counted, and excluded from training, with results also reported treating them as positive.

If attempt grouping is not available, the fallback rule is ≥ X% of the ring's edges backed by at least one laundering-flagged transaction, with the same ambiguity handling.

**Data sources.**
- **IBM AML (AMLworld, Altman et al., NeurIPS 2023; Kaggle) — primary development dataset for the pipeline machinery.** Synthetic financial transactions with laundering attempts labelled by pattern type (e.g., cycle, fan-in, fan-out, scatter-gather, bipartite, stack). Provides exact ground truth for localization, a labelled typing task for clustering, and labels independent of the baseline features. Start with the HI-Small / LI-Small variants. Laundering is not wash trading, so IBM AML validates localization, typing, and topological lift — not the wash-trading weights (see the pattern-behavior audit above).
- **Ethereum NFT trades — real-world transfer test.** Pulled from Dune exports or the Google BigQuery `crypto_ethereum` public dataset (per-address Etherscan queries are rate-limited and impractical for full-graph construction). Proxy labels from published wash-trade datasets / the Dune Spellbook `nft.wash_trades` model, subject to §4.3.
- **Own synthetic generator** — unit tests and drift simulation.

### 3.8 Evaluation protocol

- **Splits.** Sliding windows overlap, so the same ring appears in neighboring windows; random splits leak. All supervised results use a **temporal split** (train on earlier windows, test on later ones, with a gap of one window length between them). Grouped cross-validation by ring identity is used for model selection within the training period.
- **Two-stage detection:** candidate-generation recall (ground-truth rings recovered as candidates) reported separately from end-to-end precision/recall, both across the `τ_s` sweep (§3.3). Measured on synthetic data and IBM AML, where ground-truth rings are known.
- **Classification:** precision, recall, F1, PR-AUC (preferred over ROC-AUC under heavy class imbalance). Primary comparison B+D+T vs. B+D; reported overall, per synthetic scenario, and with/without hard negatives.
- **Clustering:** adjusted Rand index and purity vs. IBM pattern types; silhouette score; qualitative inspection on NFT data.
- **Regression:** RMSE, MAE, Spearman correlation vs. target; compared to baseline-feature regressors.
- **Adaptive threshold:** rolling precision/recall, cumulative reward, and their variance over time vs. three comparators (§4.5): fixed pre-drift threshold, periodically re-tuned threshold, and per-window hindsight oracle.
- **Statistical reporting:** mean ± std over ≥ 5 seeds (synthetic) or bootstrap confidence intervals (real data).

---

## 4. Threats to Validity and Scope Decisions

**4.1 Undirected vs. directed complex.** The main pipeline uses a directed flag complex, so direction is encoded topologically rather than recovered only through side features. The undirected clique complex is kept as an ablation; the difference in classifier performance between the two quantifies what the directed construction contributes.

**4.2 Ground-truth quality.** Real manipulation labels are proxy flags, not regulator-confirmed. IBM AML labels are exact but synthetic; NFT labels are real-world but heuristic. Neither alone is sufficient, which is why both are used and reported separately.

**4.3 Label leakage — in both directions.** Proxy labels for NFT wash trading are themselves produced by heuristics. If those heuristics resemble the **density/centrality baseline**, the baseline partly predicts its own label rule. If they include **cycle rules** (e.g., back-and-forth trading between the same wallets, which some public wash-trade filters use), the *topological* method may be predicting its own label rule instead, inflating its apparent lift. Before Phase 5 results are reported, the label-generation logic of each proxy source is inspected and disclosed, and results are additionally reported on the subset of rings not flagged by any cycle-based rule. IBM AML results, whose labels are independent of both feature sets, are the primary evidence for the method.

**4.4 Computational feasibility.** Mitigations: sparsify each window graph before complex construction (§3.3); compute persistence on candidate neighborhoods rather than the full graph where possible; cap the complex at dimension 2; time-budget cycle enumeration per SCC with logged fallbacks. Window-level persistence is computed on the sparsified graph only.

**4.5 Adaptive threshold: reward design and scope.** Real "confirmed catches" arrive after investigation, so rewards are delayed and censored — a substantially harder problem than a standard bandit and not this project's core contribution. Phase 6 is therefore an **offline drift simulation**:
- *Arms:* K discrete threshold levels on the classifier score (e.g., K = 10).
- *Policy:* discounted UCB (and sliding-window Thompson sampling as a comparison), implemented directly or via `river.bandit`.
- *Reward:* `TP − λ·FP` per window, with λ the relative investigation cost (swept).
- *Delay:* rewards revealed after a fixed lag of d windows (swept).
- *Drift:* the synthetic generator shifts ring size, timing regularity, and decoy-trade rate mid-stream.
- *Comparators:*
  - **Fixed threshold** — best threshold chosen on the pre-drift period (weak baseline; easy to beat under drift, so not sufficient alone).
  - **Periodically re-tuned threshold** — every M windows (swept), re-select the threshold that maximized reward on the most recent labelled windows. It receives labels with the **same lag d** as the bandit, so neither has an information advantage. This is the primary comparison.
  - **Hindsight oracle** — the best threshold per window with full knowledge of labels; an upper bound, not an achievable policy.

Results show where the bandit sits between the re-tuned baseline and the oracle. A live deployment with real delayed feedback is future work.

**4.6 Ring-type coverage.** Detection and typing claims cover cycle-based wash/laundering patterns (primary) and, if attempted, pump-and-dump coordination. Spoofing is out of scope by construction.

**4.7 Sensitivity to hyperparameters.** Window length, window stride, sparsification cutoff `τ_s`, maximum cycle length, the infinite-bar cap, and the AML labeling threshold X all change which candidates exist or how they are labelled. Each is swept in a sensitivity study and the chosen values are justified on the training period only.

**4.8 Topological lift may be small or null.** For isolated rings, persistent homology largely repeats what cycle enumeration and ring birth already provide (§3.4). Mitigations: ring birth is counted in D, not T; the primary test is B+D+T vs. B+D; results are stratified by scenario so any lift concentrated in dense, chorded, or shortcut-filled settings is visible rather than averaged away. If T adds nothing, that is reported as a finding — it would itself be evidence that combinatorial cycle features suffice for this problem.

**4.9 Hard-negative realism and generator circularity.** Without balanced, regular legitimate loops in the negatives, the classifier would appear stronger than it is. Adding them introduces a different risk: the generator's designer decides what separates them from wash rings. Hard negatives are therefore modeled on documented behavior of market makers, arbitrage bots, and hot wallets; results are reported with and without them; and NFT data with real legitimate activity serves as the external check.

**4.10 Weight transfer from wash trading to laundering.** The edge weights encode wash-trading behavior, while IBM AML contains laundering patterns that may be one-directional, decaying, and irregular. The pattern-behavior audit (§3.7) measures how each score behaves on each IBM pattern type before results are interpreted, and IBM AML results are framed as evidence about the pipeline machinery, not about the wash-trading weights.

---

## 5. System Architecture

The pipeline has seven independently testable stages:

1. **Data ingestion** — IBM AML loader, NFT trade loader (Dune/BigQuery export), synthetic generator with planted rings and drift.
2. **Graph construction** — windowing, per-pair aggregation, edge weights (§3.1), baseline graph features.
3. **Topology** — sparsification, SCCs, cycle enumeration (§3.3), weighted directed-flag persistence, landscapes (§3.4).
4. **Temporal motif** (optional) — pump-and-dump features (§3.6).
5. **Models** — classification, clustering, regression, plus baseline models.
6. **Adaptive threshold** — offline drift simulation (§4.5).
7. **Evaluation** — temporal splits, metrics, ablation and sensitivity reports.

**Repository layout**

```
src/
  data/        ibm_aml.py  nft_trades.py  synthetic.py
  graph/       windows.py  weights.py  baseline_features.py
  topology/    candidates.py  flagser_ph.py  vectorize.py
  motif/       pump_dump.py              # optional
  models/      classify.py  cluster.py  regress.py
  threshold/   bandit_sim.py
  eval/        splits.py  metrics.py  ablations.py
tests/         test_weights.py  test_planted_rings.py  test_directed_triangles.py
notebooks/     one per phase, for figures only
configs/       window, weight, and model hyperparameters (YAML)
```

**Core libraries (classical, non-deep-learning).** Python 3.10/3.11 (pinned for TDA library compatibility); `networkx` (graphs, SCCs, cycle enumeration); `pyflagser` (directed flag persistence); `gudhi` (landscapes, undirected ablation); `scikit-learn` (all models); `river` (bandit policies, optional); `pandas`/`polars` (data); `matplotlib` (figures).

---

## 6. Known Limitations

- Candidate rings are restricted to directed cycles of length ≤ 6; longer rings are missed by design (the length bound is a swept hyperparameter).
- Localization is combinatorial (cycle enumeration), with persistent homology supplying features rather than localizing rings itself; very large SCCs may need local re-sparsification.
- Edge weights rely on behavioral heuristics (reciprocity, balance, regularity); adversaries aware of them can inject decoy trades to break balance or regularity — tested partially via the decoy-rate drift in Phase 6.
- IBM AML is synthetic; NFT labels are heuristic and may share logic with either feature family (§4.3).
- The adaptive threshold is evaluated only in offline simulation (§4.5).
- For isolated rings, topology is expected to be largely redundant with cycle enumeration; its value is expected mainly where neighborhoods interact with loops (§3.4, §4.8).
- Synthetic hard negatives are modeled, not observed; their realism bounds how far synthetic results generalize (§4.9).
- Sparsification bounds recall at the candidate stage; rings lost there cannot be recovered by the classifier (§3.3).
- Pump-and-dump detection is optional and does not use topology (§3.6); spoofing is out of scope (§4.6).

---

## Appendix A — Changes from v1

| # | v1 | v2 | Reason |
|---|---|---|---|
| 1 | Undirected clique/flag complex | **Weighted directed flag complex** (`pyflagser`); undirected kept as ablation | Undirected flag complexes fill every 3-cycle immediately (zero H1 persistence), hiding three-party rings; directed flag complexes leave cyclic triangles unfilled. v1's claim that no directed-PH library exists was incorrect. |
| 2 | Volume symmetry + timing regularity | Added **node flow balance**; timing uses `1/(1+CV)`; rank normalization; explicit combination rule | Pairwise symmetry scores every edge of a multi-hop ring as benign; `1/CV` is unbounded for regular bots. |
| 3 | Localization via library representative cycles | **SCC + bounded directed cycle enumeration** | `ripser.py` returns cocycles, GUDHI returns only birth/death edges, `pyflagser` returns no generators. |
| 4 | Clustering wash trading vs. pump-and-dump by diagrams | Clustering **among cycle pattern types** (IBM AML labels) | Pump-and-dump has no persistence diagrams in this design. |
| 5 | Severity = inflated volume | **Next-window wash volume** / post-window price impact | Current-window volume is directly computable from inputs (leaky). |
| 6 | Leakage concern only toward the baseline | Leakage checked **in both directions**, plus a no-cycle-rule subset | Some public wash-trade labels use cycle rules. |
| 7 | NFT data primary; Etherscan | **IBM AML primary**, NFT as transfer test; Dune/BigQuery | Exact, typed, baseline-independent labels; Etherscan doesn't scale for full graphs. |
| 8 | Unspecified split | **Temporal split** with gap; grouped CV | Overlapping windows leak rings across random splits. |
| 9 | "H0/H1 only, no higher simplices" | Complex built to **dimension 2** | 2-simplices are required for H1 classes to die. |
| 10 | Bandit loosely specified | Arms, policy, reward, delay, drift, baseline specified | Makes Phase 6 reproducible. |
| 11 | Phases without exit criteria | **Gated phases** with week estimates | Prevents building on an unvalidated topology stage. |
| 12 | Duplicate §3.3; "five stages" listing seven | Renumbered; seven stages | Consistency. |

---

## Appendix B — Changes from v2

| # | v2 | v3 | Reason |
|---|---|---|---|
| 1 | Ring birth listed as a topological feature | **Ring birth moved to D**; explicit B / D / T feature sets | Ring birth is a plain function of the ring's edge weights; counting it in T overstates topology's contribution. |
| 2 | No account of when persistence adds information | §3.4 states that topology is near-redundant for isolated rings and informative when neighborhoods fill, split, or crowd the loop; generator plants **dense-neighborhood, chorded, and shortcut-filled** cases; results stratified by scenario; null lift is reportable (§4.8) | Prevents an unexplained null result and tests topology where it should matter. |
| 3 | Infinite bars unhandled | Capped at the window's max filtration value; infinite-bar count kept as a feature; cap sensitivity checked | Landscapes require finite deaths; the cap affects signal. |
| 4 | Phase 3 gate "B+D+T beats B" | Primary comparison **B+D+T vs. B+D**; all performance results are **reporting checkpoints with CIs**; only correctness tests are hard gates | The old gate mixed D and T effects and invited tuning until it passed. |
| 5 | Only transitive decoys | **Hard negatives**: market makers, arbitrage bots, exchange hot wallets; results with and without them (§4.9) | Balanced, regular legitimate loops are the realistic false-positive source. |
| 6 | Single detection metric | **Candidate-generation recall** reported separately from end-to-end precision/recall, across the `τ_s` sweep | Sparsification silently caps recall and shapes the negative set. |
| 7 | Wash-trading weights applied to IBM AML without checking | **Pattern-behavior audit** of each score per IBM pattern type; IBM framed as validating machinery, not weights (§4.10) | Laundering cycles need not be reciprocal, balanced, or regular. |
| 8 | No rule for ring-level labels on IBM AML | **Ring-labeling rule** (≥ X% of edges from a single attempt; ambiguous cases logged and excluded; X swept) | IBM labels are per transaction; candidates are rings. |
| 9 | Phase 6 baseline: fixed pre-drift threshold | Added **periodically re-tuned threshold** (same label lag) as primary comparator and a **hindsight oracle** upper bound | A fixed pre-drift threshold is a strawman under drift. |
