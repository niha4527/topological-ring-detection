# Project context for Claude Code

Semester AIML project: topological ring detection for market manipulation (wash trading).
The design is in `Topological_Ring_Detection_for_Market_Manipulation_v3.md` — read it before
making design decisions; section numbers (§3.1 etc.) in code comments refer to it.

## Environment
- Conda env `ringtda` (Python 3.11). Activate with `conda activate ringtda` before anything else.
- Do not use Python 3.13: pyflagser has no wheels for it.
- Install: `pip install -e ".[dev]"` (add `tda` extra for gudhi when Phase 2 landscapes start).

## Commands
- Tests: `pytest` (all must pass before a change is done)
- Notebooks: `notebooks/*.ipynb`, kernel `ringtda`

## Rules
- Pipeline logic lives in `src/ringtda/`; notebooks only import and plot.
- Every new module gets a test on a tiny synthetic case.
- Lower edge weight = more suspicious (sublevel-set filtration).
- Persistence backends: `flagser` (default, float32) and `reference` (pure Python); they must agree
  (`tests/test_directed_triangles.py`). Compare diagrams with tolerance, not exact equality.
- Engineering gates (correctness) are hard tests. Model-performance results are reported with CIs,
  never turned into tests that must pass (proposal v3 §3.7).
- No deep learning; classical ML only (scikit-learn).

## Status / next
See README "Status". Phase 1 generator is complete. Next: Phase 2 rest (persistence landscapes,
infinite-bar cap, feature table). Engineering gates run with `hard_negatives=False` ("easy
background"); recall with hard negatives on is a reported metric, not a gate.
