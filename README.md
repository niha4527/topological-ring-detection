# Topological Ring Detection for Market Manipulation

Code for the project described in `Topological_Ring_Detection_for_Market_Manipulation_v3.md`.

## One-time setup (macOS Terminal)

Python 3.11 via conda (pyflagser has no wheels for Python 3.13):

```bash
cd ~/Documents/RVU/SEM3/AIML/AIML-PROJECT
conda create -n ringtda python=3.11 -y
conda activate ringtda
pip install -e ".[dev]"
```

Later, for Phase 2 (persistence landscapes): `pip install -e ".[tda,dev]"` (adds gudhi).

## Every session

```bash
cd ~/Documents/RVU/SEM3/AIML/AIML-PROJECT
conda activate ringtda
pytest
```

In VS Code: open this folder, install the **Python** and **Jupyter** extensions, and pick `ringtda` as
the interpreter / notebook kernel. Then open `notebooks/01_sanity_visuals.ipynb` and Run All.

## Quick check that the topology library works

```bash
python sanity_check.py
```

Expected: the cyclic triangle has one H1 bar `[0.1, inf]`; the transitive triangle has none.

## Layout

```
src/ringtda/
  data/        synthetic.py (planted rings, topology-sensitive and hard negatives), toy.py (hand-built test graphs)
  graph/       windows.py (sliding windows, per-pair aggregation), weights.py (s_rec, s_bal, s_reg -> weight)
  topology/    candidates.py (SCC + cycle enumeration), persistence.py (directed flag persistence),
               vectorize.py (infinite-bar cap, landscapes, scalar summaries; local and window T features)
  viz.py       plotting helpers for notebooks
  features.py  candidate table used by the optimization report (legacy topo_* columns) and t_feature_table
configs/       topology.yaml (landscape settings, cap rule, local graph)
  models/      gd.py (gradient descent, alpha learner)    threshold/  ascent.py (threshold policies)
  eval/        labels.py (§3.7 ring-labeling rule)       motif/      later phases
experiments/   gd_report.py — optimization report; `--config report` reproduces the report's numbers
tests/         engineering gates: directed-triangle behaviour, weights, candidates, planted rings,
               generator scenarios (test_synthetic_scenarios.py)
notebooks/     exploration and figures only — import from src/, never define pipeline logic here
data/          datasets (git-ignored, except data/sample/)
results/       figures and metrics (git-ignored). results/report/ (+ the originals at results/gd_metrics.json,
               results/figures/) = the optimization report's numbers: do not overwrite.
               results/full/<ambiguous>/ = the same experiments on the full generator
```

## Rules of thumb

- Pipeline logic goes in `src/ringtda/*.py`; notebooks only call it and plot.
- Every new module gets a test on a tiny synthetic case first.
- Lower edge weight = more suspicious (sublevel-set filtration).
- Persistence backends: `flagser` (fast, default) and `reference` (pure Python, slow; used to
  cross-check flagser in `tests/test_directed_triangles.py` and as a fallback).

## Status

- [x] Phase 1: synthetic generator — isolated / dense-neighbourhood / chorded wash rings; transitive
      decoys and shortcut-filled pseudo-rings; hard negatives (arbitrage loops, market makers, hot
      wallets; `SyntheticConfig(hard_negatives=False)` for the "without" runs); evasion knobs
      (`wash_decoy_frac`, `mm_regularity`); windows; weights
- [x] Phase 2 (part): candidate rings; directed flag persistence; gates as tests
- [x] Candidate labels from planted ground truth (§3.7 rule, `eval/labels.py`)
- [x] Phase 2 topology features: local + window-level persistence, infinite-bar cap, landscapes and scalar
      summaries (`topology/vectorize.py`, `configs/topology.yaml`), T feature table (`features.t_feature_table`)
- [ ] Phase 3 start: D and B feature code, IBM AML pattern-behavior audit; then models, evaluation

## Decisions and deviations from the proposal

- **Candidate de-duplication (§3.3).** The proposal keeps "the lowest-birth one"; with that rule a
  planted dense-neighbourhood ring could be replaced by a detour through a neighbour that shares 4/5
  of its edges and is born slightly earlier. Candidates are now de-duplicated shortest-first, then by
  birth. Consequence: a k-cycle shares at most k−2 edges with any cycle of length ≤ k, so at the
  default overlap 0.8 nothing is dropped unless `max_len` ≥ 10. A different overlap measure is
  needed if real de-duplication is wanted.
- **Candidate labels (§3.7).** Synthetic candidates use the IBM AML ring-labeling rule with a planted
  structure as the "attempt": the structure's kind if ≥ X% (default 80%) of the ring's edges are
  backed by its own transactions, `background` if none are backed, else `ambiguous`. Ambiguous
  candidates are counted per window and results are reported both excluding them and treating them
  as negatives. For planted wash rings of size 2/3/5 with `max_len` ≤ 6, and for star-shaped
  structures (market makers, hot wallets), X = 80% and X = 100% give identical labels, so only the
  ambiguous class changes things. The exception is structures with more edges than one cycle, such
  as chorded 6-rings: e.g. 1→2→3→4→5 plus a background edge 5→1 is 4/5 backed (labelled at 80%,
  ambiguous at 100%).
- **Dedup is effectively off (§3.3).** At the default 80% overlap nothing is removed for
  `max_len` <= 9 and the first removals happen at 10 (verified on dense windows;
  `tests/test_candidates.py::test_dedup_removes_nothing_at_default_overlap`), so
  near-duplicate rings all stay as candidates. Phase 3 cross-validation must group by ring identity,
  including near-duplicate rings, not just by window.
- **Two T pipelines.** The optimization report uses the legacy scalar `topo_*` columns of
  `window_candidates` (fixed infinite-bar cap 1.0, frozen, report-only). The Phase 3 T table
  (`features.t_feature_table`) uses the window-max cap, landscapes and window-level features from
  `topology/vectorize.py` and `configs/topology.yaml`. New code uses the T table.
- **Dense scenario fills most planted rings (default).** With `dense_internal_chords=True` (the
  default, unchanged so the report reproduces), only 7 of 24 planted dense rings (seeds 0-5, `max_len`
  6) have `T_local_ring_inf = 1`. Cause split of the 17 filled rings: 6 inside only (incidental edges
  among the ring's own accounts), 4 outside accounts only, 1 either alone, 6 only in combination
  (`results/phase2_features/dense_fill_breakdown.csv`). `SyntheticConfig(dense_internal_chords=False)`
  removes the inside edges (same random stream otherwise) and raises the fraction to 18 of 24 (75%);
  the remaining filling is by outside accounts, which is kept as realistic (§3.4). Phase 3 stratified
  results report dense rows under BOTH settings. A planted positive with `T_local_ring_inf = 0` keeps
  its positive label: the column is a feature, not a label.
