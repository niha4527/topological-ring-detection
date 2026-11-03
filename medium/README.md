# medium/: IBM AML Medium feature build

Self-contained side project. Everything for the Medium extension lives here; nothing in `experiments/`,
`results/`, `paper/` or `src/` is changed. Background and estimates: `extension/medium_large_estimate.md`.

| Path | What |
|---|---|
| `build_features.py` | Resumable B / D / T builder, adapted from `experiments/ibm_feature_build.py`; writes only to `medium/out/` |
| `data/` | `HI-Medium_Trans.csv` (3.0 GB) and `HI-Medium_Patterns.txt`, extracted from `data/ibm_aml/ibm.zip` (git-ignored) |
| `out/HI_tau0.05/` | One `features_window_NN.csv.gz`, `ids_window_NN.csv.gz` and `window_NN.json` per finished window (git-ignored) |
| `out/betweenness_cache/` | Per-window betweenness cache (git-ignored) |
| `logs/hi_build.log` | Progress log |

## Run (from the project root)

```bash
conda activate ringtda
nohup caffeinate -i nice -n 10 python -I medium/build_features.py --dataset HI --tau 0.05 --max-len 6 > medium/logs/hi_build.log 2>&1 &
```

Same command with `--dataset LI` after downloading `LI-Medium_Trans.csv` and `LI-Medium_Patterns.txt` into `data/`
(`unzip -p data/ibm_aml/ibm.zip LI-Medium_Trans.csv > medium/data/LI-Medium_Trans.csv`, same for the patterns file).

## Check progress

```bash
tail -f medium/logs/hi_build.log      # one line per finished window; expect about 19 minutes each (23 windows)
ls medium/out/HI_tau0.05 | grep -c json
```

## Stop and resume

```bash
pkill -f medium/build_features.py     # loses only the window in progress
```

Run the same command again: finished windows (those with a `window_NN.json`) are skipped. `done.json` is written when
all windows are finished.

## Notes

- Uses about 6.5 GB of RAM while loading and up to about 9.5 GB while building a window (16 GB machine). Do not run
  two builds at once.
- `caffeinate -i` stops idle sleep. A closed lid still sleeps the laptop.
- Labels, groups and classification are NOT done by this script (features only). See section 7 of the estimate doc.
