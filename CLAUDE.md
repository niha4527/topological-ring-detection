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
See README "Status". Done: Phases 1-2 (generator, candidates, persistence, T table), Phase 3 (B and D
features, resumable IBM feature builder, pattern-behaviour audit, classification harness, HI-Small and LI-Small
runs, T_local / T_win ablations, HI<->LI transfer, synthetic per-scenario runs, candidate recall) and the results
numbers pack (`paper/numbers_pack.md`, built by `experiments/make_numbers_pack.py`). Remaining: the write-up.
Phases 4-6 (clustering and severity regression, NFT loader and provenance audit, bandit policies and drift
benchmark) are future work.
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

## Phase 3 design decisions (IBM AML)
- Windows 5 days, stride 1 day. Primary evaluation = grouped CV: groups = union-find over candidates
  sharing any edge, merged with attempt id (stricter variant: shared account, as a sensitivity check);
  all scaling/hyperparameter/threshold choices inside training folds; stratified group k-fold, 5
  repeats, pooled out-of-fold PR-AUC with grouped bootstrap CIs, prevalence and group counts reported.
  Only 10-16 positive groups per dataset (144 of 150 fold-repeats over tau 0.02/0.05/0.10 and HI/LI have < 5, 50 of 50 at tau 0.05; `results/phase3_features/ibm_split_counts.csv`): the effective sample size is the group
  count, not the positive-candidate count; report wide CIs as they are and do not tune to rescue them.
- Time-ordered check: tried as specified (train groups first seen days 0-5, test day >= 7, mixed windows
  dropped) and unusable: HI-Small test side has 0 candidates, LI-Small 1, because rings recur and almost
  every window mixes early and late groups. Dropped as an evaluation scheme and reported as unusable;
  `eval/splits.time_ordered_split` and its tests are kept; no post-hoc variant.
- Out-of-sample check = HI<->LI transfer, labelled transfer, not time-leak-free (only 3 of 705,907 LI
  accounts are shared with HI: essentially, not perfectly, disjoint). The earlier split (test windows
  from day 12) had no test positives and was dropped.
- Labeling stays per §3.7 (any single-attempt >= 80% ring is positive); report positive counts by pattern
  type and a cycle-only secondary subset (usable on HI-Small only: LI-Small has 3 cycle positives); do
  not change the rule based on counts.
- `Amount Paid` as is (currency limitation). Self-loops dropped (11.6% of HI-Small; logged; s_bal uses
  inter-account flow only). Laundering rows in no attempt (1,968 of 5,177) make a candidate `ambiguous`;
  report ambiguous excluded and counted positive. Keep `data/ibm_aml/ibm.zip`.
- IBM `tau_quantile` sweep about 0.02-0.15. Recall caps at `max_len` 6: 11 of 15 complete cycle
  attempts in the day 0-5 window (12 at `max_len` 8); 8-9 recovered.
