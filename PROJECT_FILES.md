# What each file in the project does

A guide to the repository, one to three lines per file. Paths are from the project root (`AIML-PROJECT/`).
Descriptions come from each file's own header or docstring, so they match the code. Generated or ignored folders
(`results/`, `data/`, `medium/data`, `medium/out`) are described as folders only.

**How the project fits together.** Transactions go in, `src/ringtda/` turns them into a weighted directed graph,
candidate rings, features (B, D, T) and classifier scores, and `experiments/` runs the studies that produce
`results/`. `tests/` checks correctness. `paper/` holds the write-up. `demo/` is the web demo.

---

## 1. Top-level files

| File | What it does |
|---|---|
| `README.md` | Setup, folder layout, status of each phase and the design decisions made along the way. |
| `CLAUDE.md` | Rules for the AI assistant: environment, commands, and which result folders must never be overwritten. |
| `Topological_Ring_Detection_for_Market_Manipulation_v3.md` | The design proposal (v3). Section numbers like §3.1 in code comments refer to it. |
| `prd.md` | Product requirements: what the project must deliver and its status. |
| `architecture.md` | Technical architecture: the seven stages, data flow, and a module-by-module reference. |
| `pyproject.toml` | Python package definition and dependencies (extras: `tda`, `dev`, `demo`). |
| `sanity_check.py` | Quick check that `pyflagser` works: a cyclic triangle gives one H1 bar, a transitive one gives none. |
| `PROJECT_FILES.md` | This file. |
| `Topological_Ring_Detection_Presentation.pptx` | The slide deck for the presentation. |
| `.gitignore` | Keeps data, results, caches and reference PDFs out of git. |

---

## 2. Library code: `src/ringtda/`

### Data (`src/ringtda/data/`)
| File | What it does |
|---|---|
| `src/ringtda/data/synthetic.py` | Synthetic transaction generator. Plants wash rings, decoys and hard negatives, and returns the ground truth. |
| `src/ringtda/data/ibm_aml.py` | Loads the IBM AMLworld data. Joins laundering attempts from the patterns file to transactions and drops self-loops. |
| `src/ringtda/data/toy.py` | Tiny hand-built graphs with known topology (cyclic, transitive, shortcut-filled), used by tests and notebooks. |
| `src/ringtda/data/pump_dump.py` | Generator for the optional synthetic pump-and-dump track. |

### Graph (`src/ringtda/graph/`)
| File | What it does |
|---|---|
| `src/ringtda/graph/windows.py` | Cuts transactions into sliding time windows and aggregates them to one row per ordered account pair. |
| `src/ringtda/graph/weights.py` | Computes the three edge scores (reciprocity, flow balance, timing regularity), rank-normalises them and combines them into the edge weight. Lower weight means more suspicious. |
| `src/ringtda/graph/baseline_features.py` | The B features: degree, betweenness, clustering and neighbourhood density of a candidate ring. |
| `src/ringtda/graph/direction_features.py` | The D features: ring birth, length, edge weights, degree imbalance, volume and timing statistics. |

### Topology (`src/ringtda/topology/`)
| File | What it does |
|---|---|
| `src/ringtda/topology/candidates.py` | Finds candidate rings: sparsify, strongly connected components, enumerate directed cycles up to length 6. |
| `src/ringtda/topology/persistence.py` | Persistent homology of the directed flag complex, with a fast backend (flagser) and a pure-Python reference that must agree. |
| `src/ringtda/topology/vectorize.py` | Turns persistence diagrams into the T features: landscapes, infinite-bar cap, scalar summaries and the exact "is this ring's loop ever filled" test. |

### Features and tables (`src/ringtda/`)
| File | What it does |
|---|---|
| `src/ringtda/features.py` | Builds the B, D and T feature tables for a window and lists the columns of each feature set. |
| `src/ringtda/synthetic_table.py` | A labelled candidate table (features plus labels and groups) for one synthetic window. |
| `src/ringtda/ibm_table.py` | The same for IBM windows: candidates, ring labels and first-seen times. |

### Models (`src/ringtda/models/`)
| File | What it does |
|---|---|
| `src/ringtda/models/classify.py` | Logistic regression, random forest and RBF-SVM with fixed hyperparameters; scaling is fitted inside each training fold. |
| `src/ringtda/models/cluster.py` | Ward clustering of positive rings against pattern types (exploratory). |
| `src/ringtda/models/regress.py` | Next-window severity regression on synthetic rings (exploratory). |
| `src/ringtda/models/gd.py` | Gradient descent: logistic regression, and learning the edge-score combination weights on the probability simplex. |

### Threshold (`src/ringtda/threshold/`)
| File | What it does |
|---|---|
| `src/ringtda/threshold/ascent.py` | The adaptive flagging threshold: reward `TP − λ·FP` and online gradient ascent with delayed labels, plus the fixed, re-tuned and oracle comparators. |
| `src/ringtda/threshold/bandit_sim.py` | Discounted UCB and sliding-window Thompson sampling policies for the drift benchmark. |

### Evaluation (`src/ringtda/eval/`)
| File | What it does |
|---|---|
| `src/ringtda/eval/labels.py` | The ring-labelling rule: positive if at least 80% of a ring's edges come from one planted structure or laundering attempt. |
| `src/ringtda/eval/splits.py` | Groups rings that share edges (union-find) and builds the grouped cross-validation folds. |
| `src/ringtda/eval/metrics.py` | Average precision (PR-AUC) and the grouped bootstrap used for confidence intervals. |
| `src/ringtda/eval/recall.py` | Candidate-generation recall: how many true rings the candidate step recovers. |
| `src/ringtda/eval/audit.py` | Measures how the three edge scores behave on each IBM laundering pattern type. |

### Other (`src/ringtda/`)
| File | What it does |
|---|---|
| `src/ringtda/motif/pump_dump.py` | Candidates and features for the pump-and-dump track (trade timing only, no topology). |
| `src/ringtda/simulate.py` | The demo's simulator: merges user transactions into a background week, runs the pipeline and scores the rings, returning every step. |
| `src/ringtda/viz.py` | Plotting helpers for notebooks. |
| `__init__.py` files | Mark the folders as Python packages. |

---

## 3. Settings: `configs/`

| File | What it does |
|---|---|
| `configs/topology.yaml` | Candidate cutoff, max ring length, landscape settings and the infinite-bar cap rule. |
| `configs/features.yaml` | Betweenness settings for the B features (exact up to 3,000 nodes, else 100 sampled sources). |
| `configs/classify.yaml` | Fixed model hyperparameters, CV folds and repeats, bootstrap size and the feature sets compared. |
| `configs/ibm.yaml` | IBM data location, 5-day windows with a 1-day stride, tau values and the labelling threshold. |
| `configs/bandit.yaml` | The threshold benchmark: drift stream, number of arms, rewards, label lags and policy settings. |
| `configs/cap_sensitivity.yaml` | The alternative infinite-bar caps tested in the sensitivity study. |
| `configs/cluster.yaml` | Settings for the clustering study. |
| `configs/regress.yaml` | Settings for the severity regression. |
| `configs/pump_dump.yaml` | Settings for the pump-and-dump track. |

---

## 4. Experiments: `experiments/`

Scripts you run to produce results. They call `src/ringtda/`, write into `results/`, and are never pass/fail.

### Main studies
| File | What it does |
|---|---|
| `experiments/ibm_feature_build.py` | Builds the B, D and T feature tables for IBM windows. Slow, and resumable per window. |
| `experiments/ibm_classify.py` | The main IBM classification: grouped CV and bootstrap CIs for B, B+D, B+D+T and T. |
| `experiments/synth_classify.py` | The same on synthetic data, across four conditions and per scenario. |
| `experiments/ibm_transfer.py` | Trains on HI-Small and tests on LI-Small, and the reverse. |
| `experiments/candidate_recall.py` | How many true rings the candidate step recovers across the tau sweep. |
| `experiments/ibm_pattern_audit.py` | Measures how the edge scores behave on each IBM pattern type. |
| `experiments/ibm_split_report.py` | Counts positive groups and folds for each evaluation scheme. |
| `experiments/ibm_type_diagnostic.py` | Per-pattern-type check of ring birth and balance, positives against background. |
| `experiments/gd_report.py` | Reproduces the optimization report. Never run with `--force`. |

### Side studies (exploratory)
| File | What it does |
|---|---|
| `experiments/ibm_cluster.py` | Ward clustering of IBM positive rings against pattern types. |
| `experiments/synth_severity.py` | Next-window severity regression on synthetic rings. |
| `experiments/severity_diagnostic.py` | Checks whether the severity result is forecasting or just recovering the generator's structure. |
| `experiments/cap_sensitivity.py` | How much the infinite-bar cap changes the T features' results. |
| `experiments/bandit_report.py` | The adaptive-threshold benchmark: bandits against re-tuning and gradient ascent. |
| `experiments/pump_dump_run.py` | The synthetic pump-and-dump classification. |
| `experiments/dense_fill_breakdown.py` | Why most planted dense rings are filled (not an open loop) by neighbours. |

### Plots (drawn from saved results only, no model runs)
| File | What it does |
|---|---|
| `experiments/ibm_classify_plot.py` | PR-AUC with CIs per feature set and model on IBM. |
| `experiments/synth_classify_plot.py` | The same for synthetic data. |
| `experiments/plot_lift_per_scenario.py` | Topology lift (B+D+T minus B+D) per synthetic scenario. |
| `experiments/candidate_recall_plot.py` | Recall against tau for each data source. |
| `experiments/ibm_pattern_audit_plot.py` | The pattern-behaviour audit boxes. |
| `experiments/ibm_cluster_plot.py` | Clustering scatter plots by pattern type and by cluster. |
| `experiments/bandit_report_plot.py` | The bandit benchmark figures. |
| `experiments/pump_dump_plot.py` | The pump-and-dump results. |

### Smoke checks
| File | What it does |
|---|---|
| `experiments/phase2_features_smoke.py` | A quick check that the T feature table builds on synthetic data. |
| `experiments/phase3_features_sanity.py` | A quick check that the B, D and T tables build. Not a test and not a result. |

### Paper tooling
| File | What it does |
|---|---|
| `experiments/make_numbers_pack.py` | Assembles `paper/numbers_pack.md` from existing result files. |
| `experiments/build_explainer.py` | Assembles `paper/project_explainer.md` from the explainer sections and checks every number. |
| `experiments/make_gantt.py` | Draws the project plan and status Gantt chart to `paper/gantt_chart.png` and `.svg`. |
| `experiments/make_paper_figures.py` | Draws the pipeline schematic and the triangle persistence-bars figure for the paper. |
| `experiments/replot_paper_figures.py` | Redraws two result figures at paper size. |
| `experiments/check_paper_numbers.py` | Checks that every number in `main.tex` appears in the notes, the pack or the result CSVs. |
| `experiments/check_latex_static.py` | Checks citations, labels, figures and braces in `main.tex` without needing LaTeX. |

---

## 5. Tests: `tests/`

Run with `pytest` (220 tests). These check that the code is correct. Model performance is never a test.

| File | What it checks |
|---|---|
| `tests/test_directed_triangles.py` | A cyclic triangle stays open, a transitive one is filled, a shortcut-filled one dies. The core gate. |
| `tests/test_planted_rings.py` | Planted wash rings are recovered and rank as suspicious on synthetic data. |
| `tests/test_synthetic_scenarios.py` | The generator plants what the design says it plants. |
| `tests/test_synthetic_table.py` | The labelled synthetic candidate table is well formed. |
| `tests/test_weights.py` | Edge scores, rank normalisation and the combined weight. |
| `tests/test_candidates.py` | Candidate enumeration, sparsifying and de-duplication. |
| `tests/test_vectorize.py` | The per-candidate T features are well formed. |
| `tests/test_t_table.py` | The T feature table has the right shape, keys and window-constant columns. |
| `tests/test_baseline_features.py` | The B features, checked by hand on a 4-node graph. |
| `tests/test_direction_features.py` | The D features, checked by hand. |
| `tests/test_feature_leakage.py` | Labels and planted structures never reach the B, D or T features. |
| `tests/test_labels.py` | The 80% ring-labelling rule. |
| `tests/test_ibm_aml.py` | The IBM loader, windows and labels on tiny fixtures. |
| `tests/test_splits.py` | Groups and fold assignments. |
| `tests/test_metrics.py` | PR-AUC and the grouped bootstrap. |
| `tests/test_classify.py` | The classifier pipelines. |
| `tests/test_cluster.py` | The clustering code. |
| `tests/test_regress.py` | The severity regression code. |
| `tests/test_gd.py` | The gradient-descent optimisation code. |
| `tests/test_bandit_sim.py` | The bandit policies and the threshold benchmark. |
| `tests/test_pump_dump.py` | The pump-and-dump generator and features. |
| `tests/test_audit.py` | The pattern-behaviour audit helpers on tiny tables. |
| `tests/test_simulate.py` | The demo simulator: input parsing, merging, and a wash ring versus a one-way chain. |

---

## 6. Demo: `demo/`

| File | What it does |
|---|---|
| `demo/server.py` | The FastAPI backend. A thin layer over `ringtda` for the four demo tabs; it holds no pipeline logic. |
| `demo/static/index.html` | The single-page frontend: Loop vs flow, Simulator (with a walkthrough that draws your graph, its weights and when your rings appear as the threshold rises), Pipeline and Results tabs. |

Start it with `python -m uvicorn demo.server:app --port 8000` and open http://localhost:8000.

---

## 7. Notebooks and report

| File | What it does |
|---|---|
| `notebooks/01_sanity_visuals.ipynb` | Visual sanity checks of the graphs, candidate rings and persistence. It only imports and plots. |
| `report/Optimization_Report.pdf` / `.docx` | The optimization report: gradient descent, learned edge weights and the adaptive threshold. |

---

## 8. Paper: `paper/`

### The IEEE paper (`paper/ieee/`)
| File | What it does |
|---|---|
| `paper/ieee/main.tex` | The paper source. |
| `paper/ieee/refs.bib` | The bibliography (26 verified references). |
| `paper/ieee/figures/` | The four figures used in the paper. |
| `paper/ieee/notes.md` | Traces every number in the paper back to its source in the explainer or the numbers pack. |
| `paper/ieee/check.md` | The verification record: citations, numbers, static checks and word counts. |
| `paper/ieee/outline.md` | The paper's outline. |
| `paper/ieee/sources.md` | Summaries of the reference papers. |
| `paper/ieee/overleaf.zip` | `main.tex`, `refs.bib` and figures, ready to upload to Overleaf. |
| `paper/ieee/main_compiled.pdf` | A compiled copy of the current paper. |
| `paper/ieee/main.docx` | A Word copy of the paper. |

### The explainer (`paper/explainer/`)
A long, plain-language explanation of the whole project, in numbered sections.

| File | What it covers |
|---|---|
| `paper/explainer/00_outline.md` | The outline and running checklist. |
| `paper/explainer/01_summary.md` | One-page summary and the six headline findings. |
| `paper/explainer/02_problem.md` | The problem statement and scope. |
| `paper/explainer/03_existing.md` | Existing methods and their limits. |
| `paper/explainer/04_concept_a.md` | The concept from scratch: weights, filtration, persistence and directed flag complexes. |
| `paper/explainer/05_data.md` | The synthetic generator and the IBM datasets. |
| `paper/explainer/06_impl_a.md` | Implementation: windows, weights, candidates, persistence and the T, D and B features. |
| `paper/explainer/06_impl_b.md` | Implementation: labels, groups, classification, optimization, bandits and side studies. |
| `paper/explainer/07_eval.md` | The evaluation protocol and why it looks the way it does. |
| `paper/explainer/08a_results.md` | Results: optimization report, pattern audit, IBM classification and transfer. |
| `paper/explainer/08b_results.md` | Results: synthetic conditions, recall, cap sensitivity, bandits, severity, clustering and pump-and-dump. |
| `paper/explainer/09_conclusions.md` | Conclusions, and how this differs from other solutions. |
| `paper/explainer/11_limits.md` | Limitations, deviations from the design, and what was not done. |
| `paper/explainer/12_story.md` | A decision log of how the project got here. |
| `paper/explainer/13_repro.md` | How to reproduce the results. |
| `paper/explainer/14_glossary.md` | Glossary, twenty examiner questions and verified references. |
| `paper/explainer/audit_report.md` | An audit of the explainer's numbers. |

### Other paper files
| File | What it does |
|---|---|
| `paper/project_explainer.md` / `.docx` | The explainer sections assembled into one document. |
| `paper/numbers_pack.md` | Every result number, each with its source file, config and seeds. |
| `paper/numbers_pack/` | The `csv/` tables, `figures/`, `gd_metrics.json` and `provenance.csv` behind the pack. |
| `paper/figure_list.md` | Caption drafts for the figures. |
| `paper/paper_content.md` | Early working material for the paper's text. |

---

## 9. Extension work

| File | What it does |
|---|---|
| `extension/medium_large_estimate.md` | Measured costs and projected results and time for IBM Medium and Large, with the hardware limits. |
| `extension/probes/` | The two timing scripts and logs behind those measurements. |
| `medium/build_features.py` | The feature builder for IBM Medium. It writes only inside `medium/out/`. |
| `medium/README.md` | How to run, check, stop and resume the Medium build. |
| `medium/data/`, `medium/out/`, `medium/logs/` | The extracted Medium data (git-ignored), the built features (git-ignored) and the progress log. |

---

## 10. Folders that hold data and results

| Folder | What it holds |
|---|---|
| `data/` | The datasets, git-ignored. `data/ibm_aml/` has the IBM AML files and `ibm.zip`. `data/sample/` is a placeholder. |
| `results/` | All generated figures and metrics, git-ignored. `results/report/` and `results/gd_metrics.json` are the optimization report's numbers and must never be overwritten. |
| `results/full/` | The same experiments on the full generator (the new scenarios). |
| `results/phase2_features/`, `phase3_features/`, `phase4_cluster/`, `phase4_severity/`, `phase6_bandit/`, `phase7_pump_dump/` | The outputs of each project phase. |
