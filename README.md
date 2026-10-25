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
  topology/    candidates.py (SCC + cycle enumeration), persistence.py (directed flag persistence)
  viz.py       plotting helpers for notebooks
  models/ threshold/ eval/ motif/   — later phases
tests/         engineering gates: directed-triangle behaviour, weights, candidates, planted rings,
               generator scenarios (test_synthetic_scenarios.py)
notebooks/     exploration and figures only — import from src/, never define pipeline logic here
data/          datasets (git-ignored, except data/sample/)
results/       figures and metrics (git-ignored)
```

## Rules of thumb

- Pipeline logic goes in `src/ringtda/*.py`; notebooks only call it and plot.
- Every new module gets a test on a tiny synthetic case first.
- Lower edge weight = more suspicious (sublevel-set filtration).
- Persistence backends: `flagser` (fast, default) and `reference` (pure Python, slow; used to
  cross-check flagser in `tests/test_directed_triangles.py` and as a fallback).

## Status

- [x] Phase 1: synthetic generator — isolated / dense-neighbourhood / chorded wash rings; transitive
      decoys and shortcut-filled pseudo-rings; hard negatives (market makers, arbitrage bots, hot
      wallets; `SyntheticConfig(hard_negatives=False)` for the "without" runs); windows; weights
- [x] Phase 2 (part): candidate rings; directed flag persistence; gates as tests
- [ ] Phase 2 (rest): persistence landscapes, infinite-bar cap, feature table
- [ ] Open question: §3.3 dedupe ("keep the lowest-birth one") can drop a planted dense-neighbourhood
      ring in favour of a near-duplicate detour through a neighbour (seen on 1 of 5 seeds); the
      candidate gate checks dense rings with dedupe off until this is decided
- [ ] Phase 3+: IBM AML loader and pattern audit, models, evaluation
