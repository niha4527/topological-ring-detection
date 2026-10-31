# Topological Ring Detection for Market Manipulation: a self-contained explainer

*Everything in this document was produced by the code in this repository. Every number comes from a result file or from `paper/numbers_pack.md`, which cites its source file, config and seeds. Where the code and an older document disagree, the code is followed and the disagreement is stated (section 11.3). Exploratory results are marked exploratory; nothing in this project is a pass/fail gate.*

## 1. One-page summary

**The question.** Wash trading (accounts trading assets round and round among themselves to fake volume) and money-laundering rings both show up as a *loop* in the graph of who paid whom, not as one odd transaction. Does a topological description of that loop, computed with persistent homology on a directed graph, help a classifier tell manipulation rings from benign loops, beyond what a graph-density-and-centrality baseline plus the ring's own behaviour already gives?

**The method in five sentences.**
1. Cut the transactions into time windows and build one directed graph per window, one edge per ordered account pair, each edge carrying a suspiciousness weight built from three behaviour scores (lower weight = more suspicious).
2. Find candidate rings by exact enumeration within a length bound: keep the most suspicious edges, take strongly connected components, enumerate directed simple cycles up to length 6; each cycle is an explicit set of accounts (recall against known rings is partial, section 8.7).
3. For each candidate, compute persistent homology of a *directed flag complex* of the ring's neighbourhood (a circular triangle stays "open", a one-way triangle is "filled"), turn the result into fixed-length topological features (T), and put them next to graph baseline features (B) and direction/behaviour features (D, which include the ring's birth value).
4. Train classical classifiers (logistic regression, random forest, RBF-SVM; fixed hyperparameters) with grouped cross-validation so that overlapping rings never sit in both train and test, and compare **B+D+T against B+D** with grouped-bootstrap confidence intervals.
5. Around that, test an adaptive flagging threshold (online gradient ascent and two bandit policies) under simulated drift, and run exploratory clustering, a synthetic-only severity regression, a cap sensitivity study and a small synthetic pump-and-dump track that contains no topology.

**Six headline findings (positive and negative).**
1. *Adding D to B helps on IBM AML data (itself synthetic).* On IBM HI-Small the paired gain of B+D over B has a CI excluding 0 for all three models; on LI-Small for two of three (section 8.3, 8.4).
2. *No detectable topological lift on IBM; the CIs cannot rule out a small effect either way.* The pre-specified primary comparison B+D+T minus B+D has a CI that includes 0 for all three models on HI-Small (14 positive groups; logistic regression +0.104 [-0.002, +0.171]) and on LI-Small (10 positive groups) (sections 8.3, 8.4).
3. *On synthetic data topology gives a small lift where B+D is already high.* B+D+T beats B+D by a paired CI above 0 in 7 of 12 condition-model cells. B+D itself scores 0.960-0.983 (logistic regression), 0.956-0.997 (random forest) and 0.874-0.937 (SVM) over the four conditions, and the generator is nearly separable by the D features alone (section 8.6).
4. *The ring-birth signal does not carry from HI to LI.* `-D_birth` alone scores 0.314 on HI at prevalence 0.0173 and 0.006 on LI at prevalence 0.0039 (section 8.5).
5. *The adaptive-threshold bandits lose to periodic re-tuning* at the primary setting (lambda 1, lag 2), while the existing online gradient ascent reaches 0.981 of the hindsight oracle (section 8.9).
6. *Severity regression (synthetic only; the generator decides that rings persist) recovers generator structure, not forecasting skill:* ring size (`D_length`) alone matches or beats B+D for random forest and gradient boosting (rf importance of `D_length` 0.796), and T adds nothing (section 8.10). It is a descriptive finding, not a supported claim.

**What was not done.** The NFT real-data experiment (Phase 5; Dune's free tier blocks running queries), real-data pump-and-dump validation (the optional pump-and-dump track exists only as a synthetic, no-topology run), the undirected-complex and persistence-image ablations, classifiers on single-score filtrations, and sweeps over window length, stride, label threshold and (for classification) tau (section 11.4).

**How to read this document.** Sections 2-4 explain the problem and the idea from scratch. Sections 5-7 explain the data, the implementation and the evaluation protocol. Section 8 gives each result with its setup, a table and a one-line honest conclusion. Sections 9-12 draw conclusions, compare with other work, list limitations and tell the story of the decisions. Sections 13-14 give reproduction commands, a glossary and a question-and-answer list for an examiner.
