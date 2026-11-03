# Scaling to IBM AML Medium and Large: measurements and estimates

Status: **planning document, 3 Nov 2026. No Medium or Large result exists.** Nothing in `results/`, `paper/`, the numbers pack or the report was touched. The only things produced are the timing probes in `extension/probes/` and this file.

The reason for looking: the paper's main limit is the number of positive groups (14 on HI-Small, 10 on LI-Small), so the topology comparison B+D+T versus B+D has wide intervals. Medium and Large contain many more laundering attempts. "Large" below means HI-Large and LI-Large (the larger of the two variants after Medium).

Each number is tagged **measured** (run on this machine), **derived** (arithmetic on measured or file-header numbers) or **estimate** (a projection with stated assumptions).

## 1. The machine

| | |
|---|---|
| Chip | Apple M4, 10 cores |
| RAM | 16 GB |
| Free disk | about 14 GB (disk 94% full) |
| Python | `ringtda` env, 3.11, networkx, pyflagser, scikit-learn |

## 2. How big the data is

| | HI-Small | HI-Medium | HI-Large | LI-Small | LI-Medium | LI-Large |
|---|---|---|---|---|---|---|
| Transactions CSV | 0.5 GB | 3.0 GB | 17.1 GB | 0.7 GB | 3.0 GB | 16.7 GB |
| Rows | 5.08 M (measured) | 31.9 M (measured) | about 180 M (derived from file size) | 6.1 M after self-loops | not measured | not measured |
| Laundering attempts | 370 | 2,756 | 16,467 | 117 | 456 | 2,218 |
| CYCLE attempts | 54 | 367 | 2,086 | 12 | 46 | 298 |
| Laundering dates | 1 to 18 Sep 2022 | 1 to 28 Sep 2022 | 1 Aug 2022 to 12 Jan 2023 | | 1 to 27 Sep 2022 | 1 Aug 2022 to 12 Jan 2023 |
| 5-day windows, 1-day stride | 13 (measured) | 23 (measured) | about 160 (derived from the date range) | 12 (measured) | about 22 | about 160 |

Attempt counts are from the `*_Patterns.txt` files inside `data/ibm_aml/ibm.zip`. Ratios to Small: HI-Medium 7.4x, HI-Large 44x, LI-Medium 3.9x, LI-Large 19x.

Per-day volume is similar across variants (HI-Medium 3.0 GB over about 28 days, HI-Large 17.1 GB over about 165 days: about 105 MB per day each). So one Large window is about the size of one Medium window. Large is about 7 times more windows, not bigger windows. (derived)

## 3. Measured cost on HI-Medium, window 0

Window 0 (days 0 to 5): 8,919,362 transactions, 1,621,499 accounts, 2,224,592 edges, 111,230 edges after sparsifying at the 0.05 quantile, 4,232 candidate rings. HI-Small window 0 had about 2.2 M transactions and 1,113 candidates.

| Step | Time on HI-Medium w0 | How | HI-Small w0 (README) |
|---|---|---|---|
| Load whole CSV and join patterns (once) | 81 s, 6.4 GB peak | measured | not recorded |
| Aggregate, weights, graph | 29.4 s | measured | 5.7 s |
| Candidate enumeration | 3.0 s | measured | at most 0.8 s |
| T, whole-window persistence | 14.3 s | measured | part of 11.5 s |
| T, per-candidate (21 ms each) | 89 s | 150 sampled, extrapolated to 4,232 | part of 11.5 s |
| D features (8.7 ms each) | 37 s | 150 sampled, extrapolated | small |
| B betweenness, 100 sampled sources | 165 s | measured | 17.8 s (k=50) to 34.8 s (k=100) |
| B per-candidate (195 ms each) | 826 s | 150 sampled, extrapolated | about 47 s |
| **Total** | **about 1,163 s, 19.4 min** | sum | about 1.5 min for B+T+D |

A first attempt that ran the whole `full_feature_table` for window 0 in one call was still running after 15 minutes when I stopped it, which fits the breakdown above. Peak memory of the probe was 9.45 GB.

Main finding: per-candidate B features (clustering and neighbourhood around hubs) are 71% of the time, and betweenness is another 14%. Topology (T) is only about 9%. So the slow part is the baseline, not the topology.

## 4. Time estimates

All of these assume the existing code and one process, as the builder is written today. Window time is taken as 19.4 min for every Medium window (estimate: only window 0 was measured, and later windows have fewer transactions: median 7.3 M, max 10.5 M).

| Job | Estimate | Basis |
|---|---|---|
| HI-Medium feature build | about 7.4 h (23 x 19.4 min) | estimate from the measured window |
| LI-Medium feature build | about 7 h | estimate, assuming a similar window size (not measured) |
| Both Medium, sequential | **about 15 h** | |
| Both Medium, with B per-candidate split across processes | about 5 to 8 h | estimate; 826 s per window would drop to about 100 to 200 s on 4 to 8 processes, but each process holds a copy of the window, and 16 GB of RAM allows about 1 to 2 |
| Labels, groups, split counts | tens of minutes | not measured |
| Classification (LR, RF, SVM, 5x5 grouped CV, bootstrap) on about 97k candidates per dataset | hours to days | not measured. Rows grow about 13x over Small (7.5k primary rows). RBF-SVM scales roughly with the square of rows, so it is the risk; dropping it or subsampling negatives is likely needed |
| HI-Large feature build | about 50 h (160 x 19.4 min) | estimate; **not runnable as the loader is written** (section 5) |
| LI-Large feature build | about 50 h | same |

Realistic wall-clock for a Medium-only extension, with a little engineering and some supervision: **about 1 to 2 working days**, mostly unattended build time. Large: **about 4 to 5 days of compute plus loader engineering**, on a bigger machine.

## 5. Hardware and software limits

**Hardware (this machine)**
- **Disk.** The HI-Large CSV is 17.1 GB and only about 14 GB is free, so it cannot even be extracted. Medium (3.0 GB) fits, and the unzip took 11 s.
- **RAM.** The loader reads the whole CSV into pandas. Medium peaked at 6.4 GB while loading and 9.45 GB while building a window. Scaling by rows, Large would need about 36 GB or more for loading alone, against 16 GB installed (estimate).
- **CPU.** The builder is single-process Python. Most of the 10 cores sit idle.

**Software**
- `load_ibm_files` loads everything at once and has no chunked mode. Large needs a streaming, window-by-window loader.
- Betweenness and local clustering are done with networkx in pure Python, and cost grows with the graph (about 5x from Small to Medium window 0).
- The builder has no multiprocessing. It is resumable per window, which helps after a crash.
- Cycle enumeration (3 s) and flagser persistence (14 s whole-window) are not the problem.

## 6. Estimated results

This is a projection, not a result.

**Positive groups.** Small gave 14 (HI) and 10 (LI) groups. Scaling roughly with attempts (HI: 43 of the 370 attempts gave positive candidates):

| | Small (measured) | Medium (estimate) | Large (estimate) |
|---|---|---|---|
| HI positive groups | 14 | about 100 (plausible 60 to 130) | about 600 |
| LI positive groups | 10 | about 39 (plausible 30 to 45) | about 190 |

**Interval width.** The grouped bootstrap interval shrinks roughly with the square root of the number of groups (assumption). The Small interval for logistic regression, B+D+T minus B+D, was [-0.002, +0.171] on HI (width 0.173) and [-0.007, +0.097] on LI (width 0.104).

| | Small | Medium | Large |
|---|---|---|---|
| HI interval width (LR) | 0.173 | about 0.065 | about 0.026 |
| LI interval width (LR) | 0.104 | about 0.053 | about 0.024 |

**What that would mean for the topology question**, if the Small point estimate for LR (+0.104) were the true effect, or half of it, or zero:

| True effect (LR, HI) | Medium interval, roughly | Reading |
|---|---|---|
| +0.10 | +0.07 to +0.13 | a real gain, clearly detected |
| +0.05 | +0.02 to +0.08 | a small gain, just detected |
| 0 | -0.03 to +0.03 | a tight null: no effect larger than about 0.03 |

Random forest and SVM were close to zero on Small (+0.025, +0.003). For them, the likely Medium outcome is a tight interval around a small number. At Large the intervals would be tighter again (about +/-0.013).

**What I cannot predict.** The absolute PR-AUC levels. They depend on prevalence, which could change with the variant, and on how much more training data helps. HI to LI transfer on Small was only 0.12 to 0.15, so levels are dataset-dependent. I would not quote a number.

**Honest summary.** Either outcome is useful: a small real topological gain confirmed, or a tight null that excludes anything above about 0.03. What the extension cannot do is rescue a positive story: if the effect is zero, Medium will show that clearly.

## 7. Suggested order if this is done

1. Speed up B per-candidate (the 826 s step) first, since it is the main cost and is split trivially across candidates.
2. Run HI-Medium with `experiments/ibm_feature_build.py` to a separate output folder (not `results/phase3_features/ibm/HI_tau0.05`), so the report numbers cannot be overwritten.
3. Count positive groups with the split report before classifying. If they are far below the estimate above, the extension will not tighten the intervals as expected.
4. Classify with LR and RF first. Add SVM only on a subsample.
5. Only then decide on LI-Medium and anything bigger.

## 8. How this was measured

- `probes/probe_full_window.py` and its log: first attempt at one window, did not finish after 15 minutes.
- `probes/probe_breakdown.py` and its log: the breakdown in section 3. Candidates sampled with `rng = default_rng(0)`, 150 of 4,232.
- HI-Medium CSV was extracted into a scratch folder outside the repo (`/private/tmp/.../scratchpad/ibm_medium/`), read with `python -I`, and nothing was written to `results/`.
- Attempt counts: `grep -c "BEGIN LAUNDERING ATTEMPT"` and `... - CYCLE` on each patterns file.
