# Project context for Claude Code

Semester AIML project: topological ring detection for market manipulation (wash trading).
The design is in `Topological_Ring_Detection_for_Market_Manipulation_v3.md` — read it before
making design decisions; section numbers (§3.1 etc.) in code comments refer to it.

## Environment
- Conda env `ringtda` (Python 3.11). Activate with `conda activate ringtda` before anything else.
- Do not use Python 3.13: pyflagser has no wheels for it.
- Install: `pip install -e ".[tda,dev]"` (gudhi 3.13 is installed in `ringtda`).

## Commands
- Tests: `pytest` (all must pass before a change is done)
- Notebooks: `notebooks/*.ipynb`, kernel `ringtda`
- Optimization report: `python experiments/gd_report.py --config report` (see below)

## Rules
- Pipeline logic lives in `src/ringtda/`; notebooks only import and plot.
- Every new module gets a test on a tiny synthetic case.
- Lower edge weight = more suspicious (sublevel-set filtration).
- Persistence backends: `flagser` (default, float32) and `reference` (pure Python); they must agree
  (`tests/test_directed_triangles.py`). Compare diagrams with tolerance, not exact equality.
- Engineering gates (correctness) are hard tests. Model-performance results are reported with CIs,
  never turned into tests that must pass (proposal v3 §3.7).
- No deep learning; classical ML only (scikit-learn).

## Optimization report — do not overwrite
- The report's §4 numbers and figures come from `experiments/gd_report.py --config report`, which
  pins the generator the report was written for (dense / chorded / shortcut-filled / hot-wallet
  counts = 0) and counts ambiguous candidates as negatives (report §2.4). Its outputs are
  `results/report/` and the identical originals at `results/gd_metrics.json` + `results/figures/`.
- Never overwrite either location. The script refuses to overwrite `results/report/` without
  `--force`; only use `--force` when the user asks to regenerate the report, and only after
  checking the report's quoted values still match.
- Changing the generator defaults, labels, candidates or weights can change the report; keep the
  `report` config reproducing the same numbers (re-run into a scratch copy and diff).
- Experiments with the new scenarios go to `--config full` (`results/full/<ambiguous>/`) or
  another new folder, never to the report's.

## Status / next
See README "Status". Phase 1 and Phase 2 are done: generator, candidates, persistence, and the T
feature table (`features.t_feature_table`, `src/ringtda/topology/vectorize.py`, `configs/topology.yaml`).
Next: D and B feature code plus the IBM AML pattern-behavior audit (Phase 3 start).
Engineering gates run with `hard_negatives=False` ("easy background"); recall with hard negatives on
is a reported metric, not a gate. Candidate labels: §3.7 rule in `src/ringtda/eval/labels.py` (X = 80%).

Things to remember in Phase 3:
- Dedup is effectively off: it removes nothing for `max_len` <= 9 (first removals at 10, verified): near-duplicate rings all stay candidates, so CV must
  group by ring identity (including near-duplicates), not just by window.
- The report uses the legacy scalar T features (`topo_*` in `window_candidates`, fixed cap 1.0,
  report-only, do not use in new code); the Phase 3 T table uses the window-max cap.
- `window_candidates` keeps `max_len=5` for the report; `t_feature_table` uses 6 (§3.3).

Dense scenario (Phase 3): the default generator fills most planted dense rings (7 of 24 survive;
filled: 6 inside only, 4 outside only, 1 both, 6 combined). `dense_internal_chords=False` gives 75%
surviving (rest filled by outside accounts, kept as realistic). Report dense rows under both settings;
a planted positive with `T_local_ring_inf = 0` keeps its positive label. The `report` config uses the
default.
