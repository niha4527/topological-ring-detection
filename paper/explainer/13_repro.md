
## 13. Reproducibility

**Environment.** Conda environment `ringtda`, Python 3.11 (Python 3.13 is unsupported: pyflagser has no wheels for it). Install with `pip install -e ".[tda,dev]"`. Classical machine learning only (scikit-learn); heavy runs use `nice -n 10` and at most 2 jobs.

**Tests.** `pytest -q` runs 204 tests (a few seconds). Engineering gates (cyclic triangle persists, transitive dies, shortcut-filled dies at the fill weight, backends agree, planted rings are candidates, leakage guard, target alignment, bandit delay rule) are hard tests; model performance is never a test.

**Commands, in the order the results were produced** (all from the repository root; outputs go under `results/`, which is git-ignored):

| Result | Command |
|---|---|
| Optimization report (never overwritten; refuses without `--force`) | `python experiments/gd_report.py --config report` |
| Split counts, candidate pickles | `python experiments/ibm_split_report.py` |
| B/D/T feature tables (resumable) | `python experiments/ibm_feature_build.py --dataset HI --tau 0.05 --max-len 6` (and `LI`) |
| Pattern-behaviour audit | `python experiments/ibm_pattern_audit.py --dataset HI` (and `LI`) |
| HI / LI classification | `python experiments/ibm_classify.py --dataset HI --tau 0.05` (LI: `--variants primary ambiguous_positive cycle_only --require-pos-groups 2`; ablation rows: `--append`) |
| HI<->LI transfer | `python experiments/ibm_transfer.py --tau 0.05` |
| Per-type diagnostic | `python experiments/ibm_type_diagnostic.py` |
| Synthetic classification | `python experiments/synth_classify.py` |
| Candidate recall | `python experiments/candidate_recall.py --source ibm` and `--source synthetic` |
| Bandit benchmark | `python experiments/bandit_report.py` |
| Clustering | `python experiments/ibm_cluster.py` |
| Severity and its diagnostic | `python experiments/synth_severity.py`, then `python experiments/severity_diagnostic.py` |
| Cap sensitivity | `python experiments/cap_sensitivity.py --source synthetic` and `--source ibm` |
| Figures from the CSVs | `experiments/ibm_classify_plot.py`, `synth_classify_plot.py`, `plot_lift_per_scenario.py`, `candidate_recall_plot.py`, `bandit_report_plot.py`, `ibm_cluster_plot.py` |
| Pump-and-dump track (synthetic, no topology) | `python experiments/pump_dump_run.py`, then `python experiments/pump_dump_plot.py` |
| Numbers pack, provenance, spot-check | `python experiments/make_numbers_pack.py --spot-check 10` |
| This document and its audit | `python experiments/build_explainer.py` |

**Where each result lives.** `results/phase3_features/ibm/` (features), `.../classify/` (HI, LI, transfer, synthetic, candidate recall), `.../audit/`, `.../cap_sensitivity/`, `results/phase4_cluster/`, `results/phase4_severity/`, `results/phase6_bandit/`, `results/report/` and `results/gd_metrics.json` (the optimization report). Every stored result folder has a `config_used.yaml` with the configuration and seeds.

**Checking that the report numbers are unchanged.** Re-run `python experiments/gd_report.py --config report` from an empty scratch directory outside the repository (so it writes its own `results/`), and `diff -r` that folder against a baseline copy. This was done after every change that could affect the generator, labels, candidates or weights; the diff was empty each time.

**Determinism.** Seeds are stored: CV seed 0, bootstrap seed 12345, model seed 0, synthetic seeds 0 to 5, bandit policy seed 12345. Bootstrap resamples are generated from the seed so the same resamples are reused across feature sets and models.
