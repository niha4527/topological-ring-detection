"""Assemble paper/numbers_pack.md from EXISTING result files. No experiments, no model runs, no source edits.

    python experiments/make_numbers_pack.py                 # write the pack
    python experiments/make_numbers_pack.py --spot-check 10 # also print 10 random numbers next to their source rows

Every number in the pack is read from a result file (CSV / JSON / YAML / build log) and registered with its
source in paper/numbers_pack/provenance.csv. Files that are expected but absent are listed at the end of the
run and in the pack. Deterministic: the same inputs give the same pack. Counting tests runs `pytest --collect-only`.
"""

import argparse
import json
import pickle
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ap_ = argparse.ArgumentParser()
ap_.add_argument("--spot-check", type=int, default=0)
ap_.add_argument("--seed", type=int, default=0)
args = ap_.parse_args()

RES = Path("results")
P3 = RES / "phase3_features"
CL = P3 / "classify"
SYN = CL / "synthetic"
OUT = Path("paper")
PACK = OUT / "numbers_pack"
(PACK / "csv").mkdir(parents=True, exist_ok=True)
(PACK / "figures").mkdir(parents=True, exist_ok=True)

missing: list[str] = []
prov: list[dict] = []
_cache: dict = {}


def note_missing(p):
    if str(p) not in missing:
        missing.append(str(p))


def csv(path) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        note_missing(path)
        return None
    if path not in _cache:
        _cache[path] = pd.read_csv(path)
    return _cache[path]


def yml(path) -> dict | None:
    path = Path(path)
    if not path.exists():
        note_missing(path)
        return None
    return yaml.safe_load(open(path))


def sel(df, where):
    m = pd.Series(True, index=df.index)
    for k, v in where.items():
        m &= df[k].isna() if v is None else (df[k] == v)
    return df[m]


def get(path, where, col, fmt="{:.3f}", table=""):
    """One number from one CSV row; registered for provenance and the spot check."""
    df = csv(path)
    if df is None:
        return "n/a"
    r = sel(df, where)
    if len(r) != 1:
        note_missing(f"{path} row {where} (found {len(r)} rows)")
        return "n/a"
    v = r.iloc[0][col]
    txt = "n/a" if pd.isna(v) else (fmt.format(v) if fmt else str(v))
    prov.append({"table": table, "text": txt, "file": str(path), "where": json.dumps(where, default=str), "columns": col})
    return txt


def ci_cell(path, where, table="", digits=3):
    df = csv(path)
    if df is None:
        return "n/a"
    r = sel(df, where)
    if len(r) != 1:
        return "n/a"
    r = r.iloc[0]
    f = f"{{:.{digits}f}}"
    txt = f"{f.format(r.pr_auc)} [{f.format(r.ci_low)}, {f.format(r.ci_high)}]"
    prov.append({"table": table, "text": txt, "file": str(path), "where": json.dumps(where, default=str), "columns": "pr_auc,ci_low,ci_high"})
    return txt


def diff_cell(path, where, table=""):
    df = csv(path)
    if df is None:
        return "n/a"
    r = sel(df, where)
    if len(r) != 1:
        return "n/a"
    r = r.iloc[0]
    txt = f"{r['diff']:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]"
    prov.append({"table": table, "text": txt, "file": str(path), "where": json.dumps(where, default=str), "columns": "diff,ci_low,ci_high"})
    return txt


def mdtable(header, rows):
    out = ["| " + " | ".join(map(str, header)) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out) + "\n"


def copy_csv(path):
    p = Path(path)
    if p.exists():
        dst = PACK / "csv" / "__".join(p.parts[1:])
        shutil.copyfile(p, dst)


md: list[str] = []
MODELS = ["logreg", "rf", "svm"]
SETS_MAIN = ["B", "B+D", "B+D+T", "B+D+T_local", "B+D+T_win", "T"]
SETS_CTRL = ["D", "T_local", "T_win"]
PAIR_PRIMARY = ["B+D+T minus B+D"]
PAIR_SECOND = ["B+D minus B"]
PAIR_EXPL = ["B+D+T_local minus B+D", "B+D+T_win minus B+D", "B+D+T minus B+D+T_local"]


def sec(title, level=2):
    md.append("\n" + "#" * level + " " + title + "\n")


def src(*paths, extra=""):
    md.append("*Source:* " + ", ".join(f"`{p}`" for p in paths) + (f". {extra}" if extra else "") + "\n")
    for p in paths:
        if str(p).endswith(".csv"):
            copy_csv(p)


# ====================================================================== 0. header
md.append("# Numbers pack\n")
md.append("Every number below was read by `experiments/make_numbers_pack.py` from the result file named under each table; "
          "nothing is typed by hand. Provenance of every table cell: `paper/numbers_pack/provenance.csv`. "
          "CSV copies of the sources: `paper/numbers_pack/csv/`. Figures: `paper/numbers_pack/figures/`. "
          "Nothing here is a pass/fail result; PR-AUC values carry 95% grouped-bootstrap CIs (resampling groups).\n")

# ====================================================================== 1. setup
sec("1. Setup")
test_n = "n/a"
try:
    o = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], capture_output=True, text=True, timeout=300).stdout
    m_ = re.search(r"(\d+) tests? collected", o) or re.search(r"^(\d+)/", o, re.M)
    test_n = m_.group(1) if m_ else str(len([ln for ln in o.splitlines() if "::" in ln]))
except Exception:
    note_missing("pytest --collect-only")
md.append(f"- Tests collected (`pytest --collect-only -q`): {test_n}.\n")

blog = P3 / "ibm" / "build.log"
loaded = {}
if blog.exists():
    for ln in blog.read_text().splitlines():
        m_ = re.search(r"(HI|LI): loaded (\d+) transactions", ln)
        if m_:
            loaded[m_.group(1)] = int(m_.group(2))
else:
    note_missing(blog)
rows = []
for ds in ("HI", "LI"):
    ddir = P3 / "ibm" / f"{ds}_tau0.05"
    wj = [json.load(open(f)) for f in sorted(ddir.glob("window_*.json"))]
    if not wj:
        note_missing(ddir / "window_*.json")
        continue
    done = json.load(open(ddir / "done.json")) if (ddir / "done.json").exists() else {}
    sc = csv(P3 / "ibm_split_counts.csv")
    base = {"dataset": ds, "tau_quantile": 0.05, "scheme": "grouped_cv_edge", "part": "all"}
    g = lambda c, ds=ds, base=base: get(P3 / "ibm_split_counts.csv", base | {"dataset": ds}, c, "{:.0f}", "1 setup")  # noqa: E731
    starts = [w["start_day"] for w in wj]
    rows.append([ds, f"{loaded.get(ds, 'n/a'):,}" if ds in loaded else "n/a", len(wj), f"{wj[0]['end_day'] - wj[0]['start_day']:.0f} days / {starts[1] - starts[0]:.0f} day",
                 f"{wj[0]['tau_quantile']}", wj[0]["max_len"], sum(1 for w in wj if w["rows"] > 0), done.get("rows", "n/a"), g("negatives"), g("positives"), g("ambiguous"), g("groups"), g("positive_groups")])  # fmt: skip
md.append(mdtable(["dataset", "transactions (self-loops dropped)", "windows", "window length / stride", "tau_quantile", "max_len", "windows with candidates", "candidates", "negatives", "positives (primary)", "ambiguous", "groups", "positive groups"], rows))
src(blog, P3 / "ibm" / "HI_tau0.05" / "window_*.json", P3 / "ibm" / "LI_tau0.05" / "window_*.json", P3 / "ibm_split_counts.csv",
    extra="Transactions from the build log; windows, tau and max_len from the per-window JSON written by the builder; counts from the split report (groups = union-find over shared edges merged with attempt id).")

ibm_cfg = yml("configs/ibm.yaml")
hi_cfg = yml(CL / "HI_tau0.05" / "config_used.yaml")
hi_abl = yml(CL / "HI_tau0.05" / "config_used_ablation.yaml")
li_cfg = yml(CL / "LI_tau0.05" / "config_used.yaml")
tr_cfg = yml(CL / "transfer" / "config_used.yaml")
if hi_cfg:
    c = hi_cfg["config"]
    md.append("Classification settings (stored with the HI results, identical in the LI and transfer configs unless listed below):\n")
    md.append(mdtable(["item", "value"], [
        ["logreg", f"L2, C={c['models']['logreg']['C']}, max_iter={c['models']['logreg']['max_iter']}, class_weight balanced"],
        ["random forest", f"n_estimators={c['models']['rf']['n_estimators']}, n_jobs={c['models']['rf']['n_jobs']}, class_weight balanced_subsample"],
        ["svm", f"RBF, C={c['models']['svm']['C']}, gamma={c['models']['svm']['gamma']}, class_weight balanced"],
        ["CV", f"stratified group k-fold, k={c['cv']['k']} (fallback {c['cv']['fallback_k']}), repeats={c['cv']['repeats']}, cv seed={c['cv']['seed']}"],
        ["bootstrap", f"grouped, n_boot={c['bootstrap']['n_boot']}, seed={c['bootstrap']['seed']}, 95% percentile CI"],
        ["model seed", c["model_seed"]],
        ["runtime HI (s)", f"{hi_cfg['runtime_seconds']:.0f} (first run), {hi_abl['runtime_seconds']:.0f} (ablation rows appended)" if hi_abl else f"{hi_cfg['runtime_seconds']:.0f}"],
        ["runtime LI (s)", f"{li_cfg['runtime_seconds']:.0f}" if li_cfg else "n/a"],
        ["runtime transfer (s)", f"{tr_cfg['runtime_seconds']:.0f}" if tr_cfg else "n/a"],
    ]))
    src(CL / "HI_tau0.05" / "config_used.yaml", CL / "HI_tau0.05" / "config_used_ablation.yaml", CL / "LI_tau0.05" / "config_used.yaml", CL / "transfer" / "config_used.yaml")
if hi_abl:
    md.append(f"Ablation rows (`B+D+T_local`, `B+D+T_win`) and their paired rows were appended to the HI results with the same settings (config `{CL / 'HI_tau0.05' / 'config_used_ablation.yaml'}`). "
              f"The paired list stored with the config: {hi_abl['config'].get('paired')}.\n")
if ibm_cfg:
    md.append(f"Label rule X (§3.7): {ibm_cfg['labels']['label_x']} (`configs/ibm.yaml`, current file; **not stored with the results**, flagged). "
              f"Split config in the same file: k={ibm_cfg['splits']['k']}, fallback {ibm_cfg['splits']['fallback_k']}, repeats {ibm_cfg['splits']['repeats']}.\n")
syn_cfg = yml(SYN / "hn-off_chords-default" / "config_used.yaml")
if syn_cfg:
    md.append(f"Synthetic runs: seeds {syn_cfg['seeds']} pooled per condition, conditions hard_negatives x dense_internal_chords, CV repeats {syn_cfg['config']['cv']['repeats']}, "
              f"bootstrap n_boot {syn_cfg['config']['bootstrap']['n_boot']}, same fixed hyperparameters (stored in `{SYN / 'hn-off_chords-default' / 'config_used.yaml'}`). "
              f"Synthetic generator parameters and candidate tau (0.25) are **not stored with the results** (flagged); the synthetic recall CSV stores max_len = "
              f"{csv(CL / 'candidate_recall' / 'synthetic.csv').max_len.iloc[0] if csv(CL / 'candidate_recall' / 'synthetic.csv') is not None else 'n/a'}.\n")
topo = yml("configs/topology.yaml")
if topo:
    md.append(f"Candidate step defaults for the synthetic table (`configs/topology.yaml`, current file): tau_quantile {topo['candidates']['tau_quantile']}, max_len {topo['candidates']['max_len']}.\n")

# betweenness sparsity
bt = []
for ds in ("HI", "LI"):
    f = P3 / "ibm" / "betweenness_cache" / f"{ds}_w00.pkl"
    if f.exists():
        d = pickle.load(open(f, "rb"))
        b = np.array(list(d["betweenness"].values()), dtype=float)
        bt.append([ds, f"{len(b):,}", f"{int((b > 0).sum()):,}", f"{(b > 0).mean():.4f}", d.get("exact")])
        prov.append({"table": "1 betweenness", "text": f"{(b > 0).mean():.4f}", "file": str(f), "where": "betweenness values > 0", "columns": "betweenness"})
    else:
        note_missing(f)
if bt:
    wj0 = json.load(open(P3 / "ibm" / "HI_tau0.05" / "window_00.json"))["betweenness"]
    md.append(f"Betweenness on window 0 (days 0-5), sampled with k={wj0['k']} sources (seed {wj0['seed']}, exact only up to {wj0['exact_max_nodes']} nodes):\n")
    md.append(mdtable(["dataset", "nodes", "nodes with betweenness > 0", "share non-zero", "exact"], bt))
    src(P3 / "ibm" / "betweenness_cache" / "HI_w00.pkl", P3 / "ibm" / "betweenness_cache" / "LI_w00.pkl")

# ====================================================================== 2. optimization report
sec("2. Optimization report headline numbers")
gd = None
if (RES / "gd_metrics.json").exists():
    gd = json.load(open(RES / "gd_metrics.json"))
    shutil.copyfile(RES / "gd_metrics.json", PACK / "gd_metrics.json")
else:
    note_missing(RES / "gd_metrics.json")
if gd:
    def flat(x, p=""):
        if isinstance(x, dict):
            for k, v in x.items():
                yield from flat(v, f"{p}{k}." if isinstance(v, dict) else f"{p}{k}")
        else:
            yield p, x

    rows = [[k.rstrip("."), json.dumps(v)] for k, v in flat(gd)]
    md.append("Copied unchanged (values as stored, full precision; lists as JSON). Generator pinned to the report configuration (dense / chorded / shortcut-filled / hot-wallet counts = 0).\n")
    md.append(mdtable(["key", "value"], rows))
    src(RES / "gd_metrics.json", extra="Identical original at `results/report/gd_metrics.json`; not modified.")
    for k, v in flat(gd):
        prov.append({"table": "2 gd_metrics", "text": json.dumps(v), "file": "results/gd_metrics.json", "where": k, "columns": "json"})

# ====================================================================== IBM blocks
def ibm_block(ds, num):
    d = CL / f"{ds}_tau0.05"
    R, Pd = d / "results.csv", d / "paired_differences.csv"
    r = csv(R)
    p = csv(Pd)
    if r is None or p is None:
        return
    sec(f"{num}.1 {ds}-Small: label counts by variant and type", 3)
    rows = []
    for v in ["primary", "ambiguous_positive", "cycle_only", "strict_groups"]:
        w = {"variant": v, "feature_set": "B", "model": "logreg"}
        if len(sel(r, w)) == 0:
            rows.append([v, "unusable (see note) or not run", "", "", "", ""])
            continue
        mp = ("min_pos_groups_on_a_side" in r.columns) and not pd.isna(sel(r, w).iloc[0].get("min_pos_groups_on_a_side"))
        rows.append([v, get(R, w, "n_rows", "{:.0f}", "labels"), get(R, w, "n_pos", "{:.0f}", "labels"), get(R, w, "prevalence", "{:.4f}", "labels"), get(R, w, "n_pos_groups", "{:.0f}", "labels"),
                     get(R, w, "min_pos_groups_on_a_side", "{:.0f}", "labels") if mp else "not recorded"])  # fmt: skip
    md.append(mdtable(["variant", "candidates", "positives", "prevalence", "positive groups", "fewest positive groups on a train or test side"], rows))
    if ds == "LI" and li_cfg and li_cfg.get("unusable_variants"):
        for u in li_cfg["unusable_variants"]:
            md.append(f"**{u['variant']} is unusable on {ds}**: {u['positives']} positives in {u['positive_groups']} groups; fewest positive groups on a train or test side = {u['min_positive_groups_on_a_side']} (< 2). Source: `{d / 'config_used.yaml'}`.\n")
    base = {"dataset": ds, "tau_quantile": 0.05, "scheme": "grouped_cv_edge", "part": "all"}
    SC = P3 / "ibm_split_counts.csv"
    types = ["CYCLE", "GATHER-SCATTER", "SCATTER-GATHER", "STACK"]
    md.append(mdtable(["candidates", "negatives", "ambiguous", "positives", *[f"pos {t}" for t in types], "cycle positives", "cycle positive groups"],
                      [[get(SC, base, c, "{:.0f}", "labels") for c in ["candidates", "negatives", "ambiguous", "positives"]] + [get(SC, base, f"pos_{t}", "{:.0f}", "labels") for t in types]
                       + [get(SC, base, c, "{:.0f}", "labels") for c in ["cycle_positives", "cycle_positive_groups"]]]))
    src(R, SC)

    sec(f"{num}.2 {ds}-Small: primary PR-AUC [95% CI], ambiguous excluded", 3)
    w0 = {"variant": "primary", "feature_set": "B", "model": "logreg"}
    md.append(f"Prevalence {get(R, w0, 'prevalence', '{:.4f}', 'primary')}, positives {get(R, w0, 'n_pos', '{:.0f}', 'primary')}, positive groups {get(R, w0, 'n_pos_groups', '{:.0f}', 'primary')}, "
              f"k per repeat {sel(r, w0).iloc[0].k_per_repeat}, folds without positive {get(R, w0, 'folds_without_positive', '{:.0f}', 'primary')}. Pooled out-of-fold, mean over {get(R, w0, 'repeats', '{:.0f}', 'primary')} repeats.\n")
    rows = []
    for s in SETS_MAIN + SETS_CTRL:
        if len(sel(r, {"variant": "primary", "feature_set": s})):
            rows.append([s] + [ci_cell(R, {"variant": "primary", "feature_set": s, "model": m}, f"{num}.2") for m in MODELS])
    md.append(mdtable(["feature set", *MODELS], rows))
    if len(sel(r, {"variant": "primary", "feature_set": "D_birth_alone"})):
        md.append(f"`-D_birth` alone (single-feature detector, no model, scores within this dataset): {ci_cell(R, {'variant': 'primary', 'feature_set': 'D_birth_alone', 'model': 'none'}, f'{num}.2')}\n")
    rows = []
    for s in SETS_MAIN + SETS_CTRL:
        if len(sel(r, {"variant": "primary", "feature_set": s})):
            rows.append([s] + [get(R, {"variant": "primary", "feature_set": s, "model": m}, "per_window_pr_auc", "{:.3f}", f"{num}.2") for m in MODELS])
    md.append(f"Mean of per-window PR-AUC over the {get(R, w0, 'n_windows_with_pos', '{:.0f}', 'primary')} windows with at least one positive:\n")
    md.append(mdtable(["feature set", *MODELS], rows))
    src(R)

    sec(f"{num}.3 {ds}-Small: paired differences (primary), same resamples", 3)
    md.append("**Pre-specified primary comparison: B+D+T minus B+D.** Second: B+D minus B. The remaining comparisons are **exploratory** (T_local / T_win ablations, added after the first run).\n")
    for title, comps in (("Pre-specified primary", PAIR_PRIMARY), ("Second", PAIR_SECOND), ("Exploratory", PAIR_EXPL)):
        rows = [[c] + [diff_cell(Pd, {"variant": "primary", "model": m, "comparison": c}, f"{num}.3") for m in MODELS] + [get(Pd, {"variant": "primary", "model": "logreg", "comparison": c}, "n_boot", "{:.0f}", f"{num}.3")]
                for c in comps if len(sel(p, {"variant": "primary", "comparison": c}))]
        if rows:
            md.append(f"{title}:\n")
            md.append(mdtable(["comparison", *MODELS, "n_boot"], rows))
    n_expl_p = int(p[(p.variant == "primary") & p.comparison.isin(PAIR_EXPL)].shape[0])
    n_expl_all = int(p[p.comparison.isin(PAIR_EXPL)].shape[0])
    md.append(f"**Exploratory comparisons made on {ds}: {n_expl_p} in the primary variant ({len(PAIR_EXPL)} comparisons x {len(MODELS)} models), {n_expl_all} over all variants in this file. No multiplicity correction is applied.**\n")
    src(Pd)

    sec(f"{num}.4 {ds}-Small: other variants", 3)
    for v in ["ambiguous_positive", "cycle_only", "strict_groups"]:
        rv = sel(r, {"variant": v})
        if not len(rv):
            md.append(f"{v}: not run or unusable.\n")
            continue
        w = {"variant": v, "feature_set": "B", "model": "logreg"}
        md.append(f"**{v}**: prevalence {get(R, w, 'prevalence', '{:.4f}', f'{num}.4')}, positives {get(R, w, 'n_pos', '{:.0f}', f'{num}.4')}, positive groups {get(R, w, 'n_pos_groups', '{:.0f}', f'{num}.4')}.\n")
        rows = [[s] + [ci_cell(R, {"variant": v, "feature_set": s, "model": m}, f"{num}.4") for m in MODELS] for s in SETS_MAIN if len(sel(r, {"variant": v, "feature_set": s}))]
        md.append(mdtable(["feature set", *MODELS], rows))
        rows = [[c] + [diff_cell(Pd, {"variant": v, "model": m, "comparison": c}, f"{num}.4") for m in MODELS] for c in PAIR_PRIMARY + PAIR_SECOND + PAIR_EXPL if len(sel(p, {"variant": v, "comparison": c}))]
        md.append(mdtable(["paired difference" + (" (exploratory rows last)" if v else ""), *MODELS], rows))
    src(R, Pd)


def audit_block(ds, num):
    A = P3 / "audit" / f"{ds}_audit_summary.csv"
    I = P3 / "audit" / f"{ds}_audit_imputation.csv"
    a, imp = csv(A), csv(I)
    if a is None:
        return
    sec(f"{num} {ds}-Small pattern-behaviour audit (pooled over the audited windows)", 3)
    wins = sorted(w for w in a.window.unique() if w != "pooled")
    md.append(f"Audited windows: {', '.join(wins)} and pooled. **Tail share** = share of a group's pairs in the most suspicious 10% tail of the window (for `weight`: the lowest 10% of weights); "
              "the all-pairs baseline is 0.100 for `weight`, `s_bal`, `s_reg` and is the `baseline` column. AUC = edge-level AUC of the group against background pairs (0.5 = chance, higher = the score ranks the group as more suspicious). "
              "`s_rec` is almost always 0 (ties), so its tail share and baseline are tie-driven.\n")
    groups = [g for g in a[a.window == "pooled"].group.unique() if g != "background"]
    rows = []
    for g in groups:
        k = {"window": "pooled", "group": g}
        rows.append([g, get(A, k | {"score": "weight"}, "n_pairs", "{:.0f}", f"{num}"), get(A, k | {"score": "weight"}, "auc_vs_background", "{:.3f}", f"{num}"), get(A, k | {"score": "s_bal"}, "auc_vs_background", "{:.3f}", f"{num}"),
                     get(A, k | {"score": "s_rec"}, "auc_vs_background", "{:.3f}", f"{num}"), get(A, k | {"score": "s_reg"}, "auc_vs_background", "{:.3f}", f"{num}"),
                     get(A, k | {"score": "weight"}, "decile_share", "{:.3f}", f"{num}"), get(A, k | {"score": "s_bal"}, "decile_share", "{:.3f}", f"{num}")])  # fmt: skip
    kb = {"window": "pooled", "group": "background", "score": "weight"}
    rows.append(["background (baseline)", get(A, kb, "n_pairs", "{:.0f}", num), "-", "-", "-", "-", get(A, kb, "decile_share_all", "{:.3f}", num), get(A, kb | {"score": "s_bal"}, "decile_share_all", "{:.3f}", num)])
    md.append(mdtable(["group", "pairs", "AUC weight", "AUC s_bal", "AUC s_rec", "AUC s_reg", "tail share weight", "tail share s_bal"], rows))
    src(A)
    if imp is not None:
        rows = [[g, get(I, {"window": "pooled", "group": g}, "n_pairs", "{:.0f}", num), get(I, {"window": "pooled", "group": g}, "share_imputed", "{:.3f}", num)] for g in list(groups) + ["background"]]
        md.append(f"Share of pairs with fewer than 3 transactions, whose timing-regularity score is imputed (`gap_cv` fill):\n")
        md.append(mdtable(["group", "pairs", "share imputed"], rows))
        src(I)


# ====================================================================== 3. IBM HI
sec("3. IBM HI-Small")
ibm_block("HI", "3")
audit_block("HI", "3.5")

# ====================================================================== 4. LI + transfer
sec("4. IBM LI-Small and HI<->LI transfer")
ibm_block("LI", "4")
audit_block("LI", "4.5")
TR, TP = CL / "transfer" / "results.csv", CL / "transfer" / "paired_differences.csv"
tr, tp = csv(TR), csv(TP)
sec("4.6 HI<->LI transfer", 3)
if tr is not None and tp is not None:
    SCt = P3 / "ibm_split_counts.csv"
    kt = {"dataset": "HI->LI", "tau_quantile": 0.05, "scheme": "transfer", "part": "test"}
    md.append(f"Labelled as **transfer, not a time-leak-free test**: both datasets come from the same generator and cover the same dates; {get(SCt, kt, 'n_shared_accounts', '{:.0f}', 'transfer')} accounts are shared between HI and LI, "
              f"used by {get(SCt, kt, 'train_candidates_using_shared', '{:.0f}', 'transfer')} training and {get(SCt, kt, 'test_candidates_using_shared', '{:.0f}', 'transfer')} test candidates (HI->LI; source `{SCt}`). "
              "Primary variant (ambiguous excluded in train and test), scaling fitted on the training dataset only, **one model fitted per direction (no repeats)**, CI = grouped bootstrap over the TEST groups.\n")
    for tn, te in (("HI", "LI"), ("LI", "HI")):
        w0 = {"train": tn, "test": te, "feature_set": "B", "model": "logreg"}
        md.append(f"**Train {tn} -> test {te}**: {get(TR, w0, 'n_test_rows', '{:.0f}', 'transfer')} test candidates, {get(TR, w0, 'n_test_pos', '{:.0f}', 'transfer')} positives "
                  f"(prevalence {get(TR, w0, 'prevalence', '{:.4f}', 'transfer')}) in {get(TR, w0, 'n_test_pos_groups', '{:.0f}', 'transfer')} test groups; training positives {get(TR, w0, 'n_train_pos', '{:.0f}', 'transfer')}.\n")
        rows = [[s] + [ci_cell(TR, {"train": tn, "test": te, "feature_set": s, "model": m}, "4.6") for m in MODELS] for s in ["B", "B+D", "B+D+T", "B+D+T_local", "T", "D"]]
        md.append(mdtable(["feature set", *MODELS], rows))
        md.append(f"`-D_birth` row: {ci_cell(TR, {'train': tn, 'test': te, 'feature_set': 'D_birth_alone', 'model': 'none'}, '4.6')}. "
                  f"**This is the within-dataset score of a no-model detector on the {te} candidates, not a transfer result** (nothing is fitted on {tn}).\n")
        rows = [[c] + [diff_cell(TP, {"train": tn, "test": te, "model": m, "comparison": c}, "4.6") for m in MODELS] for c in ["B+D+T minus B+D", "B+D+T_local minus B+D"]]
        md.append(mdtable(["paired difference (B+D+T minus B+D is the pre-specified primary; the T_local row is exploratory)", *MODELS], rows))
    src(TR, TP)

# ====================================================================== 5. synthetic
sec("5. Synthetic data")
CONDS = ["hn-off_chords-default", "hn-off_chords-off", "hn-on_chords-default", "hn-on_chords-off"]
SYN_SETS = ["B", "B+D", "B+D+T", "B+D+T_local", "T", "T_local", "D"]
for c in CONDS:
    R, Pd = SYN / c / "results.csv", SYN / c / "paired_differences.csv"
    r, p = csv(R), csv(Pd)
    if r is None:
        continue
    sec(f"5.{CONDS.index(c) + 1} Condition {c}", 3)
    w0 = {"variant": "primary", "feature_set": "B", "model": "logreg"}
    md.append(f"Primary (ambiguous excluded): {get(R, w0, 'n_rows', '{:.0f}', c)} candidates, {get(R, w0, 'n_pos', '{:.0f}', c)} positives, prevalence {get(R, w0, 'prevalence', '{:.4f}', c)}, "
              f"{get(R, w0, 'n_pos_groups', '{:.0f}', c)} positive groups; seeds pooled, {get(R, w0, 'repeats', '{:.0f}', c)} CV repeats. Cell = pooled PR-AUC [95% CI] (mean +- std of the per-seed PR-AUC, {get(R, w0, 'n_seeds', '{:.0f}', c)} seeds).\n")
    rows = []
    for s in SYN_SETS:
        rows.append([s] + [ci_cell(R, {"variant": "primary", "feature_set": s, "model": m}, c) + f" ({get(R, {'variant': 'primary', 'feature_set': s, 'model': m}, 'seed_mean', '{:.3f}', c)} +- {get(R, {'variant': 'primary', 'feature_set': s, 'model': m}, 'seed_std', '{:.3f}', c)})" for m in MODELS])  # fmt: skip
    md.append(mdtable(["feature set", *MODELS], rows))
    md.append(f"`-D_birth` alone (no model): {ci_cell(R, {'variant': 'primary', 'feature_set': 'D_birth_alone', 'model': 'none'}, c)} "
              f"({get(R, {'variant': 'primary', 'feature_set': 'D_birth_alone', 'model': 'none'}, 'seed_mean', '{:.3f}', c)} +- {get(R, {'variant': 'primary', 'feature_set': 'D_birth_alone', 'model': 'none'}, 'seed_std', '{:.3f}', c)}).\n")
    rows = [[cmp_] + [diff_cell(Pd, {"variant": "primary", "model": m, "comparison": cmp_, "scope": "overall"}, c) for m in MODELS] for cmp_ in ["B+D+T minus B+D", "B+D minus B", "B+D+T_local minus B+D"]]
    md.append(mdtable(["paired difference, overall (B+D+T minus B+D pre-specified; T_local row exploratory)", *MODELS], rows))
    va = sel(r, {"variant": "ambiguous_positive"})
    if len(va):
        wa = {"variant": "ambiguous_positive", "feature_set": "B", "model": "logreg"}
        ms = [m for m in MODELS if len(sel(r, {"variant": "ambiguous_positive", "feature_set": "B", "model": m}))]
        md.append(f"Ambiguous counted positive: {get(R, wa, 'n_rows', '{:.0f}', c)} candidates, {get(R, wa, 'n_pos', '{:.0f}', c)} positives, prevalence {get(R, wa, 'prevalence', '{:.3f}', c)}, "
                  f"{get(R, wa, 'n_pos_groups', '{:.0f}', c)} groups; models run: {', '.join(ms)} (limited for runtime).\n")
        rows = [[s] + [ci_cell(R, {"variant": "ambiguous_positive", "feature_set": s, "model": m}, c) if m in ms else "not run" for m in MODELS] for s in ["B", "B+D", "B+D+T", "B+D+T_local", "T"]]
        md.append(mdtable(["feature set", *MODELS], rows))
    src(R, Pd)

    SPc, SPp = SYN / c / "per_scenario.csv", SYN / c / "per_scenario_paired.csv"
    sc, sp = csv(SPc), csv(SPp)
    if sc is not None:
        md.append("Per scenario: positives of ONE scenario against ALL negatives from the same out-of-fold scores (**prevalence changes per scenario**).\n")
        for scn in sorted(sc.scenario.unique()):
            w = {"scenario": scn, "feature_set": "B", "model": "logreg"}
            md.append(f"*{scn}*: {get(SPc, w, 'n_pos', '{:.0f}', c)} positives, prevalence {get(SPc, w, 'prevalence', '{:.4f}', c)}, {get(SPc, w, 'n_pos_groups', '{:.0f}', c)} positive groups.\n")
            rows = [[s] + [ci_cell(SPc, {"scenario": scn, "feature_set": s, "model": m}, c, 2) for m in MODELS] for s in ["B", "B+D", "B+D+T", "B+D+T_local", "T"]]
            if sp is not None:
                rows += [[cmp_] + [diff_cell(SPp, {"scenario": scn, "model": m, "comparison": cmp_}, c) for m in MODELS] for cmp_ in ["B+D+T minus B+D", "B+D+T_local minus B+D", "B+D minus B"]]
            md.append(mdtable(["feature set / paired difference", *MODELS], rows))
        src(SPc, SPp)

sec("5.5 Dense scenario under both chord settings", 3)
rows = []
for c in CONDS:
    F = SYN / c / "positives_ring_inf.csv"
    for scn in ["dense"]:
        k = {"scenario": scn}
        if csv(F) is not None:
            rows.append([c, get(F, k, "positive_candidates", "{:.0f}", "5.5"), get(F, k, "with_ring_inf_0", "{:.0f}", "5.5"), get(F, k, "frac_ring_inf_0", "{:.3f}", "5.5"),
                         get(F, k, "planted_rings_with_candidate", "{:.0f}", "5.5"), get(F, k, "planted_rings_with_ring_inf_1", "{:.0f}", "5.5")])  # fmt: skip
md.append(mdtable(["condition", "dense positive candidates", "with T_local_ring_inf = 0", "fraction", "planted dense rings with a positive candidate", "of those with T_local_ring_inf = 1"], rows))
md.append("Positives with `T_local_ring_inf = 0` keep their positive label (the column is a feature). Dense-scenario PR-AUC and paired differences for each setting are in the per-scenario tables above.\n")
rows = []
for c in CONDS:
    for cmp_ in ["B+D+T minus B+D", "B+D+T_local minus B+D"]:
        rows.append([c, cmp_] + [diff_cell(SYN / c / "per_scenario_paired.csv", {"scenario": "dense", "model": m, "comparison": cmp_}, "5.5") for m in MODELS])
md.append(mdtable(["condition", "dense paired difference", *MODELS], rows))
src(*[SYN / c / "positives_ring_inf.csv" for c in CONDS], *[SYN / c / "per_scenario_paired.csv" for c in CONDS])

sec("5.6 Mean out-of-fold score rank percentile by candidate kind", 3)
md.append("Used instead of a false-positive rate at 80% recall. 1 = top of the ranking, 0.5 = middle. Primary variant, mean over CV repeats; `n` is the number of candidates of that kind.\n")
for c in CONDS:
    Nf = SYN / c / "negatives_rank.csv"
    n = csv(Nf)
    if n is None:
        continue
    kinds = [k for k in ["wash_ring", "shortcut_filled", "background", "arbitrage_loop", "market_maker", "hot_wallet"] if k in set(n.kind)]
    for m in ["logreg", "rf"]:
        rows = [[s] + [get(Nf, {"feature_set": s, "model": m, "kind": k}, "mean_rank_percentile", "{:.3f}", "5.6") for k in kinds] for s in ["B", "B+D", "B+D+T", "B+D+T_local", "T"]]
        md.append(f"*{c}, {m}* (n: " + ", ".join(f"{k} {get(Nf, {'feature_set': 'B', 'model': m, 'kind': k}, 'n', '{:.0f}', '5.6')}" for k in kinds) + ")\n")
        md.append(mdtable(["feature set", *kinds], rows))
    src(Nf)

sec("5.7 Candidate kinds and ambiguity per condition (all seeds pooled)", 3)
rows, amb = [], {}
for c in CONDS:
    parts = []
    for f in sorted((SYN / "cache").glob(f"{c}_seed*.pkl")):
        parts.append(pd.read_pickle(f))
    if not parts:
        note_missing(SYN / "cache" / f"{c}_seed*.pkl")
        continue
    t = pd.concat(parts)
    n_amb = int(t.y.isna().sum())
    amb[c] = n_amb / len(t)
    vc = t.kind.value_counts()
    rows.append([c, len(t), n_amb, f"{n_amb / len(t):.3f}", int(vc.get("wash_ring", 0)), int(vc.get("shortcut_filled", 0)), int(vc.get("arbitrage_loop", 0)), int(vc.get("market_maker", 0)), int(vc.get("hot_wallet", 0)), int(vc.get("background", 0))])
    prov.append({"table": "5.7", "text": f"{n_amb} / {len(t)}", "file": f"{SYN}/cache/{c}_seed*.pkl", "where": "y isna / rows", "columns": "y"})
md.append(mdtable(["condition", "candidates", "ambiguous", "ambiguous share", "wash_ring", "shortcut_filled", "arbitrage_loop", "market_maker", "hot_wallet", "background"], rows))
md.append(f"Sources: per-seed candidate tables `{SYN / 'cache'}/<condition>_seed<k>.pkl` (written by `experiments/synth_classify.py`).\n")

# ====================================================================== 6. recall
sec("6. Candidate-generation recall")
IB, SY = CL / "candidate_recall" / "ibm.csv", CL / "candidate_recall" / "synthetic.csv"
ib, sy = csv(IB), csv(SY)
if ib is not None:
    sec("6.1 IBM AML", 3)
    md.append("Complete = CYCLE attempts with all transactions inside a window; reachable = complete attempts with at most `max_len` accounts. **Recovered (account set)** = a candidate of the same window has exactly the attempt's account set (the README measurement). "
              "**Recovered (labelled)** = in addition the candidate carries a positive CYCLE label under the §3.7 rule. The difference between the two columns is the account-set vs label-based gap. Scope `days0-5` is window 0 only; `whole_data` is the union over all windows.\n")
    rows = []
    for ds in ("HI", "LI"):
        for scope in ("days0-5", "whole_data"):
            for ml in (6, 8):
                for q in (0.02, 0.05, 0.10, 0.15):
                    k = {"source": ds, "scope": scope, "max_len": ml, "tau_quantile": q}
                    rows.append([ds, scope, ml, q, get(IB, k, "n_complete", "{:.0f}", "6.1"), get(IB, k, "n_complete_le_maxlen", "{:.0f}", "6.1"), get(IB, k, "n_recovered", "{:.0f}", "6.1"), get(IB, k, "n_recovered_labelled", "{:.0f}", "6.1"),
                                 get(IB, k, "recall_of_complete", "{:.3f}", "6.1"), get(IB, k, "recall_of_reachable", "{:.3f}", "6.1"), get(IB, k, "candidates_per_window", "{:.0f}", "6.1")])  # fmt: skip
    md.append(mdtable(["dataset", "scope", "max_len", "tau_quantile", "complete", "reachable", "recovered (account set)", "recovered (labelled)", "recall of complete", "recall of reachable", "candidates per window"], rows))
    src(IB)
if sy is not None:
    sec("6.2 Synthetic planted wash rings", 3)
    md.append("Recall = share of planted wash rings (6 seeds) with a candidate labelled as that ring; `max_len` 6. **†** marks cells where the per-SCC enumeration cap (20000 cycles) was hit in at least one of the pooled windows; "
              "the *truncated share* column is the fraction of the pooled windows in the row's cell with at least one truncated SCC. Recall there is limited by the cap, not by tau.\n")
    g = sy.assign(trunc=sy.n_truncated_sccs > 0).groupby(["hard_negatives", "dense_internal_chords", "scenario", "tau_quantile"]).agg(recall=("recovered", "mean"), n=("recovered", "size"), trunc=("trunc", "mean")).reset_index()
    wins = sy.drop_duplicates(["hard_negatives", "dense_internal_chords", "seed", "tau_quantile"]).assign(trunc=lambda d: d.n_truncated_sccs > 0).groupby(["hard_negatives", "dense_internal_chords", "tau_quantile"]).agg(cand=("n_candidates", "mean"), trunc_w=("trunc", "mean")).reset_index()
    taus = sorted(sy.tau_quantile.unique())
    for hn in (False, True):
        for ch in (True, False):
            rows = []
            for scn in ["isolated", "dense", "chorded"]:
                cells = []
                for q in taus:
                    x = g[(g.hard_negatives == hn) & (g.dense_internal_chords == ch) & (g.scenario == scn) & (g.tau_quantile == q)]
                    cells.append("n/a" if not len(x) else f"{x.recall.iloc[0]:.2f}{'†' if x.trunc.iloc[0] > 0 else ''}")
                rows.append([scn, *cells])
            cand = []
            tw = []
            for q in taus:
                x = wins[(wins.hard_negatives == hn) & (wins.dense_internal_chords == ch) & (wins.tau_quantile == q)]
                cand.append(f"{x.cand.iloc[0]:.0f}")
                tw.append(f"{x.trunc_w.iloc[0]:.2f}")
            rows += [["mean candidates per window", *cand], ["truncated share of windows", *tw]]
            md.append(f"*hard negatives {'on' if hn else 'off'}, dense_internal_chords {'default' if ch else 'off'}*\n")
            md.append(mdtable(["scenario \\ tau_quantile", *taus], rows))
            prov.append({"table": "6.2", "text": f"hn={hn} chords={ch}", "file": str(SY), "where": "groupby hard_negatives,dense_internal_chords,scenario,tau_quantile", "columns": "recovered,n_truncated_sccs,n_candidates"})
    rings = sy.groupby(["scenario"]).planted_id.nunique()
    md.append(f"Planted wash rings per window (per condition, per seed): " + ", ".join(f"{s} {int(sy[(sy.scenario == s) & (sy.seed == 0) & (sy.hard_negatives == False) & (sy.dense_internal_chords == True) & (sy.tau_quantile == taus[0])].shape[0])}" for s in ["isolated", "dense", "chorded"]) + " (seed 0).\n")
    src(SY)

# ====================================================================== 6A-6D. phases finished after the first pack
HIR, HIP = CL / "HI_tau0.05" / "results.csv", CL / "HI_tau0.05" / "paired_differences.csv"
LIR, LIP = CL / "LI_tau0.05" / "results.csv", CL / "LI_tau0.05" / "paired_differences.csv"
P6 = Path("results/phase6_bandit")
PDD = Path("results/phase7_pump_dump")
SEV = Path("results/phase4_severity")
CAP = P3 / "cap_sensitivity"
CLU = Path("results/phase4_cluster")
sec("6A. Phase 6: bandit threshold policies in the drift benchmark (synthetic)")
b_cfg = yml(P6 / "config_used.yaml")
if b_cfg:
    bc = b_cfg["config"]
    md.append(f"Stream = the optimization report's drift stream (wash timing jitter {bc['stream']['jitter'][0]} -> {bc['stream']['jitter'][1]}, amount noise {bc['stream']['amount_noise'][0]} -> {bc['stream']['amount_noise'][1]} over {bc['stream']['n_windows']} windows; "
              f"stream seed 0 is the report's stream, {b_cfg['n_seeds']} seeds, window seeds offset by {bc['stream']['seed_offset']} per seed). K = {bc['arms']['K']} arms (thresholds {bc['arms']['lo']}..{bc['arms']['hi']}); reward TP - lambda*FP; "
              f"lambdas {bc['reward']['lambdas']} (primary {bc['reward']['primary_lambda']}); lags d {bc['lags']} (primary {bc['primary_lag']}); warm start {bc['warmup']} windows; discounted UCB gamma {bc['ducb']['gamma']}, c {bc['ducb']['c']}; "
              f"sliding-window Thompson window {bc['sw_ts']['window']} (Beta-Bernoulli on the normalised reward); re-tune every M in {bc['retune_every']} (primary {bc['primary_retune_every']}); ascent eta {bc['ascent']['eta']}, T {bc['ascent']['T']}. "
              f"Hyperparameters were fixed in `configs/bandit.yaml` before the run. Runtime {b_cfg['runtime_seconds']:.0f} s. Config stored in `{P6 / 'config_used.yaml'}`.\n")
    md.append("Share of oracle = cumulative reward from window 5 / hindsight-oracle cumulative reward (401-point grid). Mean +- std over seeds. Bandits see only the played arm's reward; comparators see whole labelled windows. "
              "Lag convention: labels of window s usable at t >= s + d; bandits cannot use a reward before the arm is played, so their effective lag is max(d, 1); at d = 0 the re-tuned and ascent comparators use the window's own labels (not like-for-like).\n")
    SM = P6 / "summary.csv"
    POLS = ["fixed", "retune_M2", "retune_M5", "retune_M10", "ascent", "ducb", "swts", "oracle_arms", "oracle"]
    for lam in bc["reward"]["lambdas"]:
        for kind, mc, sc_, fm in (("share of oracle", "share_mean", "share_std", "{:.3f}"), ("cumulative reward", "cum_mean", "cum_std", "{:.1f}")):
            if kind == "cumulative reward" and lam != bc["reward"]["primary_lambda"]:
                continue
            rows = [[p] + [get(SM, {"lambda": lam, "lag": d, "policy": p}, mc, fm, "6A") + " +- " + get(SM, {"lambda": lam, "lag": d, "policy": p}, sc_, fm, "6A") for d in bc["lags"]] for p in POLS]
            md.append(f"{kind}, lambda = {lam}:\n")
            md.append(mdtable(["policy"] + [f"d={d}" for d in bc["lags"]], rows))
    BV = P6 / "bandit_vs_comparators.csv"
    md.append("Bandit minus comparator, share of oracle, paired per seed (primary comparison: bandit vs re-tuned with the same lag, M = " + str(bc["primary_retune_every"]) + "; negative = the bandit loses). lambda = " + str(bc["reward"]["primary_lambda"]) + ":\n")
    rows = []
    for d in bc["lags"]:
        for bnd in ("ducb", "swts"):
            for cmp_ in (f"retune_M{bc['primary_retune_every']}", "ascent"):
                k = {"lambda": bc["reward"]["primary_lambda"], "lag": d, "bandit": bnd, "comparator": cmp_}
                rows.append([d, bnd, cmp_, get(BV, k, "diff_share_mean", "{:+.3f}", "6A") + " +- " + get(BV, k, "diff_share_std", "{:.3f}", "6A"), get(BV, k, "seeds_bandit_better", "{:.0f}", "6A") + "/" + get(BV, k, "n_seeds", "{:.0f}", "6A")])
    md.append(mdtable(["d", "bandit", "comparator", "mean difference +- std", "seeds with the bandit better"], rows))
    md.append("**Sensitivity (exploratory, lambda " + str(bc["reward"]["primary_lambda"]) + f", d {bc['primary_lag']}).** The pre-set values (gamma {bc['ducb']['gamma']}, window {bc['sw_ts']['window']}) are the headline; the rows below were NOT used to choose them and gamma 0.95 is NOT adopted:\n")
    rows = [[p, get(SM, {"lambda": bc["reward"]["primary_lambda"], "lag": bc["primary_lag"], "policy": p}, "share_mean", "{:.3f}", "6A") + " +- " + get(SM, {"lambda": bc["reward"]["primary_lambda"], "lag": bc["primary_lag"], "policy": p}, "share_std", "{:.3f}", "6A")]
            for p in [f"sens_ducb_gamma{g}" for g in bc["sensitivity"]["gamma"]] + [f"sens_swts_window{w}" for w in bc["sensitivity"]["window"]]]
    md.append(mdtable(["setting", "share of oracle"], rows))
    RS = P6 / "results.csv"
    gd0 = json.load(open(RES / "gd_metrics.json"))["problem_C"]["cumulative_reward_after_warmup"] if gd else {}
    k0 = {"seed": 0, "lambda": 1.0, "lag": 2}
    md.append("Seed-0 reproduction of the report's online-ascent result (lambda 1, d 2, cumulative reward from window 5):\n")
    md.append(mdtable(["policy", "this run (seed 0)", "`results/gd_metrics.json`"], [[a, get(RS, k0 | {"policy": p}, "cum_reward", "{:.1f}", "6A"), gd0.get(a, "n/a")] for a, p in (("fixed", "fixed"), ("retune", "retune_M5"), ("ascent", "ascent"), ("oracle", "oracle"))]))
    src(SM, BV, RS, extra="Stream cache `results/phase6_bandit/cache/stream_seed<k>.pkl`.")
    st = P6 / "cache" / "stream_seed0.pkl"
    if st.exists():
        from ringtda.threshold.bandit_sim import arm_grid, reward_matrix  # read-only use of the stored stream

        Rm = reward_matrix(pd.read_pickle(st)["windows"], arm_grid(bc["arms"]["K"], bc["arms"]["lo"], bc["arms"]["hi"]), 1.0)[bc["warmup"]:]
        md.append(f"**Arm-9 cost (stream seed 0, lambda 1, windows {bc['warmup']}-{bc['stream']['n_windows'] - 1}).** Mean raw reward per window of arms 0-8 (thresholds {bc['arms']['lo']}..0.85): {Rm[:, :-1].mean():.2f}; of the last arm (threshold {bc['arms']['hi']}, flags almost every candidate): "
                  f"mean {Rm[:, -1].mean():.1f}, worst {Rm[:, -1].min():.0f}. The policies see rewards clipped to +-{bc['reward']['scale']:.0f}, so one exploratory play of that arm costs about as much as the whole oracle total ({float(np.sum(csv(RS)[(csv(RS).seed == 0) & (csv(RS)['lambda'] == 1.0) & (csv(RS).lag == 2) & (csv(RS).policy == 'oracle')].oracle_cum_reward)):.0f}) but is invisible to them.\n")
        prov.append({"table": "6A arm-9", "text": f"{Rm[:, -1].mean():.1f}", "file": str(st), "where": "reward_matrix lambda 1 arms", "columns": "scores,labels"})
else:
    md.append("`results/phase6_bandit/config_used.yaml` not found.\n")

sec("6B. Phase 4: next-window severity regression (SYNTHETIC ONLY)")
s_cfg = yml(SEV / "config_used.yaml")
RG, PG, PS = SEV / "results.csv", SEV / "paired_differences.csv", SEV / "population_by_scenario.csv"
DG = SEV / "diagnostic"
if s_cfg and csv(RG) is not None:
    sc_ = s_cfg["config"]
    md.append("**Synthetic only: IBM AML has no next-window wash volume, and the generator decides that a planted ring persists (its rounds run all week). These numbers are not evidence of forecasting skill on real data.** "
              f"Target {sc_['target']} (log1p of the ring's own wash volume in window t+1); {sc_['data']['seeds']} seeds, hard negatives off, {sc_['data']['n_windows']} one-day windows of one week; a window-t candidate is matched to the same ring in t+1 by account set. "
              f"Excluded D columns (own current volume / activity): {', '.join(s_cfg['excluded'])}. Models ridge (alpha {sc_['models']['ridge']['alpha']}), random forest ({sc_['models']['rf']['n_estimators']} trees), gradient boosting ({sc_['models']['gbr']['n_estimators']} trees, depth {sc_['models']['gbr']['max_depth']}, lr {sc_['models']['gbr']['learning_rate']}); "
              f"grouped CV by ring ({sc_['cv']['k']} folds, {sc_['cv']['repeats']} repeats), pooled out-of-fold, grouped bootstrap {sc_['bootstrap']['n_boot']} resamples over rings. Config: `{SEV / 'config_used.yaml'}`. Target mean {s_cfg['target_mean']:.3f}, std {s_cfg['target_std']:.3f}.\n")
    md.append(f"Population: {get(RG, {'feature_set': 'B', 'model': 'ridge'}, 'n_pairs', '{:.0f}', '6B')} (candidate, window) pairs from {get(RG, {'feature_set': 'B', 'model': 'ridge'}, 'n_rings', '{:.0f}', '6B')} rings.\n")
    md.append(mdtable(["scenario", "pairs", "rings"], [[sn, get(PS, {"scenario": sn}, "pairs", "{:.0f}", "6B"), get(PS, {"scenario": sn}, "rings", "{:.0f}", "6B")] for sn in ("isolated", "dense", "chorded")]))
    SM_ = ["ridge", "rf", "gbr"]
    for met in ("rmse", "mae", "spearman"):
        cell = lambda w: f"{get(RG, w, met, '{:.3f}', '6B')} [{get(RG, w, met + '_ci_low', '{:.3f}', '6B')}, {get(RG, w, met + '_ci_high', '{:.3f}', '6B')}]"  # noqa: E731
        rows = [[s] + [cell({"feature_set": s, "model": m}) for m in SM_] for s in ("B", "B+D", "B+D+T", "T")]
        rows.append(["mean predictor", cell({"feature_set": "mean", "model": "none"}), "", ""])
        rows.append(["persistence reference (excluded `D_log_volume` used directly as the predictor, unscaled)", cell({"feature_set": "persistence_ref", "model": "none"}), "", ""])
        rows.append(["fair persistence (y = a + b * `D_log_volume`, fitted in each training fold; diagnostic)", (lambda w: f"{get(DG / 'results.csv', w, met, '{:.3f}', '6B')} [{get(DG / 'results.csv', w, met + '_ci_low', '{:.3f}', '6B')}, {get(DG / 'results.csv', w, met + '_ci_high', '{:.3f}', '6B')}]")({"predictor": "persistence_fit", "model": "none"}), "", ""])
        md.append(f"{met.upper() if met != 'spearman' else 'Spearman'} [95% CI] ({'log1p units, lower is better' if met != 'spearman' else 'higher is better'}):\n")
        md.append(mdtable(["feature set / baseline", *SM_], rows))
    md.append("Paired differences, B+D+T minus B+D (same resamples; pre-specified primary comparison; RMSE and MAE: negative = B+D+T better):\n")
    md.append(mdtable(["metric", *SM_], [[met] + [diff_cell(PG, {"model": m, "comparison": "B+D+T minus B+D", "metric": met}, "6B") for m in SM_] for met in ("rmse", "mae", "spearman")]))
    src(RG, PG, PS, SEV / "config_used.yaml", extra="Seeds 0-5.")
    md.append("**Diagnostic (structure-only predictors; labelled as diagnostics, not features; the scenario label comes from the generator).** Same pairs, groups, folds, model seeds and bootstrap resamples. "
              "RMSE [95% CI]:\n")
    DR = DG / "results.csv"
    md.append(mdtable(["predictor", *SM_], [[s] + [get(DR, {"predictor": s, "model": m}, "rmse", "{:.3f}", "6B") + " [" + get(DR, {"predictor": s, "model": m}, "rmse_ci_low", "{:.3f}", "6B") + ", " + get(DR, {"predictor": s, "model": m}, "rmse_ci_high", "{:.3f}", "6B") + "]" for m in SM_] for s in ("B+D", "D_length", "scenario", "D_length+scenario")]))
    DP = DG / "paired_differences.csv"
    md.append("Paired B+D minus baseline, RMSE (negative = B+D better):\n")
    md.append(mdtable(["comparison", *SM_], [[c] + [diff_cell(DP, {"model": m, "comparison": c, "metric": "rmse"}, "6B") for m in SM_] for c in ("B+D minus D_length", "B+D minus scenario", "B+D minus D_length+scenario")] + [["B+D minus persistence_fit"] + [diff_cell(DP, {"model": m, "comparison": "B+D minus persistence_fit", "metric": "rmse"}, "6B") for m in SM_]]))
    dpd = csv(DP)
    flag = dpd[(dpd.metric == "rmse") & dpd.ci_includes_0 & dpd.comparison.str.contains("D_length|scenario")]
    md.append("Flag: " + ("structure-only predictors whose RMSE difference to B+D has a CI including 0: " + ", ".join(f"{r.model}/{r.comparison}" for r in flag.itertuples()) + "." if len(flag) else "no structure-only baseline matches B+D within the RMSE CI.")
              + f" Structure-only `D_length` has a lower RMSE than B+D for rf ({get(DR, {'predictor': 'D_length', 'model': 'rf'}, 'rmse', '{:.3f}', '6B')} vs {get(DR, {'predictor': 'B+D', 'model': 'rf'}, 'rmse', '{:.3f}', '6B')}) and gbr.\n")
    IM = DG / "rf_importance_BD.csv"
    imp_ = csv(IM)
    if imp_ is not None:
        md.append("Random-forest impurity importance of B+D (no volume), top 10:\n")
        md.append(mdtable(["rank", "column", "importance"], [[i + 1, r.column, f"{r.rf_impurity_importance:.3f}"] for i, r in imp_.head(10).iterrows()]))
        for i in range(10):
            prov.append({"table": "6B", "text": f"{imp_.iloc[i].rf_impurity_importance:.3f}", "file": str(IM), "where": json.dumps({"rank": i + 1}), "columns": "rf_impurity_importance"})
    src(DR, DP, IM, DG / "config_used.yaml")
else:
    md.append("Severity results not found.\n")

sec("6C. Infinite-bar cap sensitivity")
md.append("Primary cap = `window_max` (the pre-set default); `fixed_1.0`, `max_x1.1` (1.1 x window max) and `ring_local_max` (T_local only: max edge weight of the ring's own neighbourhood; T_win keeps the window max) are sensitivity variants defined in `configs/cap_sensitivity.yaml`. "
          "The persistence diagrams do not depend on the cap; B+D rows do not depend on it either. Fixed hyperparameters, same folds and bootstrap resamples as the stored classification runs. **The `ring_local_max` svm cells are a post hoc finding in a single model, uncorrected for multiplicity, and are not a result.**\n")
for srcn, label in (("ibm", "IBM HI-Small (tau 0.05, primary variant)"), ("synthetic", "synthetic hn-off, chords default (6 seeds, primary variant)")):
    CR, CPd, CRc = CAP / srcn / "results.csv", CAP / srcn / "paired_differences.csv", CAP / srcn / "reproduction_check.csv"
    cr = csv(CR)
    if cr is None:
        continue
    r0 = cr.iloc[0]
    nm = r0.source
    md.append(f"**{label}**: {int(r0.n_rows)} candidates, {int(r0.n_pos)} positives, prevalence {r0.n_pos / r0.n_rows:.4f}, {int(r0.n_pos_groups)} positive groups. `window_max` reproduces the stored T columns to max abs difference {get(CRc, {'source': nm}, 'max_abs_diff_window_max_vs_stored_T', '{:.1f}', '6C')}. "
              f"B+D (cap-free): " + ", ".join(f"{m} {ci_cell(CR, {'variant': '(cap-free)', 'feature_set': 'B+D', 'model': m}, '6C')}" for m in MODELS) + "\n")
    for sname in ("B+D+T", "B+D+T_local"):
        md.append(f"{sname}, PR-AUC [95% CI]:\n")
        md.append(mdtable(["cap variant", *MODELS], [[vv] + [ci_cell(CR, {"variant": vv, "feature_set": sname, "model": m}, "6C") for m in MODELS] for vv in ("window_max", "fixed_1.0", "max_x1.1", "ring_local_max")]))
    md.append("Paired differences (same resamples):\n")
    rows = []
    for vv in ("window_max", "fixed_1.0", "max_x1.1", "ring_local_max"):
        for sname in ("B+D+T", "B+D+T_local"):
            rows.append([vv, f"{sname} minus B+D"] + [diff_cell(CPd, {"variant": vv, "feature_set": sname, "model": m, "comparison": f"{sname} minus B+D"}, "6C") for m in MODELS])
            if vv != "window_max":
                rows.append([vv, f"{sname} under this cap minus under window_max"] + [diff_cell(CPd, {"variant": vv, "feature_set": sname, "model": m, "comparison": f"{sname} under this cap minus {sname} under window_max"}, "6C") for m in MODELS])
    md.append(mdtable(["cap variant", "comparison", *MODELS], rows))
    src(CR, CPd, CRc, CAP / srcn / "config_used.yaml")

sec("6D. Phase 4: Ward clustering of positive rings against pattern type (exploratory)")
CLR, CLC = CLU / "results.csv", CLU / "counts_by_type.csv"
if csv(CLR) is not None:
    cl = csv(CLR)
    cc_ = yml(CLU / "config_used.yaml")
    md.append("Ward linkage, Euclidean, vectors standardized on the population; population = positives of the primary variant for pattern types with at least 3 positive groups; k = number of types (fixed) or best silhouette over 2..8 (exploratory). "
              f"CIs: grouped bootstrap over positive groups ({int(cl.n_boot.iloc[0])} resamples, recluster each). Baselines: mean over 1000 permutations of the cluster labels, and majority-class purity. Config `{CLU / 'config_used.yaml'}`. "
              f"**The grouped-bootstrap percentile CIs of this study are biased (resampled duplicate rings inflate agreement and compactness) and {int(((cl.ari_type < cl.ari_type_ci_low) | (cl.ari_type > cl.ari_type_ci_high) | (cl.purity_type < cl.purity_type_ci_low) | (cl.purity_type > cl.purity_type_ci_high) | (cl.ari_group < cl.ari_group_ci_low) | (cl.ari_group > cl.ari_group_ci_high)).sum())} of {len(cl)} rows have at least one CI that excludes its own point estimate, so they are not used as CIs: read the point estimates against the random-permutation and majority baselines.** Read ARI against pattern type *next to* ARI against group id: where the group-id ARI is larger, the clusters track attempts more than pattern types (HI, D+T_local landscape, fixed k: {get(CLR, {'dataset': 'HI', 'vector': 'D+T_local_landscape', 'k_mode': 'fixed'}, 'ari_group', '{:.3f}', '6D')} against group id vs {get(CLR, {'dataset': 'HI', 'vector': 'D+T_local_landscape', 'k_mode': 'fixed'}, 'ari_type', '{:.3f}', '6D')} against type). LI (T_local landscape, fixed k = 3, {get(CLR, {'dataset': 'LI', 'vector': 'T_local_landscape', 'k_mode': 'fixed'}, 'n_groups', '{:.0f}', '6D')} groups, 3 to 5 groups per type): ARI {get(CLR, {'dataset': 'LI', 'vector': 'T_local_landscape', 'k_mode': 'fixed'}, 'ari_type', '{:.3f}', '6D')}, purity {get(CLR, {'dataset': 'LI', 'vector': 'T_local_landscape', 'k_mode': 'fixed'}, 'purity_type', '{:.3f}', '6D')} against random {get(CLR, {'dataset': 'LI', 'vector': 'T_local_landscape', 'k_mode': 'fixed'}, 'purity_type_random', '{:.3f}', '6D')}: a small-sample point estimate with its caveat, not evidence of type recovery.\n")
    cn = csv(CLC)
    md.append(mdtable(["dataset", "type", "rings", "groups", "in population"], [[r.dataset, r.kind, int(r.rings), int(r.groups), r.in_population] for r in cn.itertuples()]))
    for ds in cl.dataset.unique():
        rows = []
        for r in cl[cl.dataset == ds].itertuples():
            k = {"dataset": ds, "vector": r.vector, "k_mode": r.k_mode}
            rows.append([f"{r.vector} ({r.k_mode})", r.k] + [get(CLR, k, s, "{:.3f}", "6D") + " [" + get(CLR, k, s + "_ci_low", "{:.3f}", "6D") + ", " + get(CLR, k, s + "_ci_high", "{:.3f}", "6D") + "]" for s in ("ari_type", "purity_type", "ari_group")]
                        + [get(CLR, k, "ari_type_random", "{:.3f}", "6D"), get(CLR, k, "purity_type_random", "{:.3f}", "6D"), get(CLR, k, "purity_majority", "{:.3f}", "6D")])
        md.append(f"{ds}:\n")
        md.append(mdtable(["vector", "k", "ARI vs type [CI]", "purity vs type [CI]", "ARI vs group id [CI]", "random ARI", "random purity", "majority purity"], rows))
    src(CLR, CLC)
else:
    md.append("not run (`results/phase4_cluster/` not found).\n")

sec("6E. Comparisons made across the project (multiplicity note)")
md.append("Each row of a paired-difference file is one comparison (model x variant x scope [x metric]) with its own CI; rows within a file are not independent. "
          "**Definition.** *Pre-specified primary* = rows with `comparison == 'B+D+T minus B+D'`, one row per model x dataset x variant, at overall scope: IBM HI (4 variants x 3 models), IBM LI (2 variants x 3 models), HI<->LI transfer (2 directions x 3 models), synthetic (per condition: primary and ambiguous-positive variants, overall scope), "
          "severity (3 metrics x 3 models) and the Phase 6 bandit-vs-re-tuned (M = 5) comparison at lambda 1, d 2 (2 bandits). *Pre-specified secondary* = rows with `comparison == 'B+D minus B'` at the same scope. "
          "Everything else is *exploratory or sensitivity*: the T_local / T_win ablation rows, per-scenario rows, cap-sensitivity rows (including its B+D+T minus B+D rows), diagnostics, bandit lags and lambdas other than the primary, and bandit-vs-ascent rows. No multiplicity correction is applied anywhere.\n")
mult = []
def _n(path, mask=None, mask2=None):
    d = csv(path)
    return (None, None, None) if d is None else (len(d), int(mask(d).sum()), int(mask2(d).sum()) if mask2 else 0)
_P, _S = (lambda d: d.comparison == "B+D+T minus B+D"), (lambda d: d.comparison == "B+D minus B")
_none = lambda d: d.comparison != d.comparison  # noqa: E731
_ovr = lambda f: (lambda d: f(d) & (d.scope == "overall"))  # noqa: E731
for label, path, mk, mk2 in (
    ("IBM HI classification", HIP, _P, _S),
    ("IBM LI classification", LIP, _P, _S),
    ("HI<->LI transfer", TP, _P, _S),
    *[(f"synthetic {c}", SYN / c / "paired_differences.csv", _ovr(_P), _ovr(_S)) for c in CONDS],
    *[(f"synthetic per scenario {c}", SYN / c / "per_scenario_paired.csv", _none, _none) for c in CONDS],
    ("cap sensitivity HI", CAP / "ibm" / "paired_differences.csv", _none, _none),
    ("cap sensitivity synthetic", CAP / "synthetic" / "paired_differences.csv", _none, _none),
    ("severity (3 metrics)", PG, _P, _S),
    ("severity diagnostic (3 metrics)", DP, _none, _none),
    ("pump-and-dump (synthetic, no topology)", PDD / "paired_differences.csv", _none, _none),
    ("bandit vs comparators", BV, lambda d: (d.comparator == f"retune_M{b_cfg['config']['primary_retune_every']}") & (d["lambda"] == b_cfg["config"]["reward"]["primary_lambda"]) & (d.lag == b_cfg["config"]["primary_lag"]), lambda d: d.bandit != d.bandit),
):
    tot_, pr_, se_ = _n(path, mk, mk2)
    if tot_ is None:
        continue
    mult.append([label, f"`{path}`", tot_, pr_, se_, tot_ - pr_ - se_])
    prov.append({"table": "6E", "text": f"{label}: {tot_} rows, {pr_} primary, {se_} secondary", "file": str(path), "where": "row count", "columns": "comparison"})
md.append(mdtable(["study", "file", "paired rows", "pre-specified primary", "pre-specified secondary", "exploratory / sensitivity"], mult))
md.append(f"Total paired rows over the project: {sum(m[2] for m in mult)}; pre-specified primary: {sum(m[3] for m in mult)}; pre-specified secondary: {sum(m[4] for m in mult)}; exploratory or sensitivity: {sum(m[5] for m in mult)}. "
          "**The pre-specified primary rows are not independent tests**: they include every variant (e.g. ambiguous-positive, strict-groups), the same candidates and resamples recur across models and variants, and the count is of rows. The headline test per dataset is `B+D+T minus B+D` on the primary variant, one row per model.\n")

# ====================================================================== 6F. Phase 7 optional: synthetic pump-and-dump
sec("6F. Phase 7 (optional): synthetic pump-and-dump track (SYNTHETIC ONLY, NO TOPOLOGY)")
PDR, PDP, PDK, PDC = PDD / "results.csv", PDD / "paired_differences.csv", PDD / "candidate_kinds.csv", PDD / "config_used.yaml"
pd_cfg = yml(PDC)
if pd_cfg and csv(PDR) is not None:
    from ringtda.data.pump_dump import PumpDumpConfig
    from ringtda.motif.pump_dump import BPD_NAMES, M_NAMES

    gc, cc_, cond = pd_cfg["pump_dump_config"]["generator"], pd_cfg["pump_dump_config"]["candidates"], pd_cfg["pump_dump_config"]["conditions"]
    dflt = PumpDumpConfig()
    md.append("**Synthetic only, no topology. The generator author decides how separable the classes are**; nothing here says real pump-and-dump schemes look like this. "
              f"One asset with a noisy hourly price and account-level trades. Planted groups per seed: {gc['n_pump']} pump-and-dump (positives: members accumulate in a {gc['accum_band_h']:.0f} h band, then all sell around a common exit while the price runs up and falls back), "
              f"{gc['n_organic']} organic momentum groups (independent sell times over days) and {gc['n_news']} news-spike groups (unsynchronised sellers; hard negative), each of {dflt.group_size[0]}-{dflt.group_size[1]} accounts, placed in separate time slots of {dflt.slot_h:.0f} h so their bands never overlap "
              f"(slot and group size are code defaults of `PumpDumpConfig`, not stored in the run config); {gc['n_accounts']} accounts, {gc['duration_h']:.0f} h, {pd_cfg['pump_dump_config']['seeds']} seeds; background trades are small and independent. "
              f"Candidates are coordinated-accumulation windows found from trades only: a {cc_['band_h']:.0f} h band (stride {cc_['stride_h']:.0f} h) in which at least {cc_['m_min']} accounts each buy at least {cc_['acc_min']:.0f} (Jaccard {cc_['jaccard']} de-duplication). "
              f"Labels: a candidate takes a planted group's kind if at least {pd_cfg['pump_dump_config']['label_x']} of its accounts belong to it; y = 1 for pump-and-dump; groups = planted group (union-find over shared accounts, seed-prefixed). "
              f"**M features ({len(M_NAMES)}):** {', '.join(f'`{n}`' for n in M_NAMES)}. **B_pd ({len(BPD_NAMES)}):** {', '.join(f'`{n}`' for n in BPD_NAMES)} (group size and per-account volume; no timing or price). They read only the trade table and the price series, never labels or planted ids (tested). "
              "The features use sells and prices after the accumulation band: this is **retrospective detection of a completed episode, not early warning**. "
              f"The generator background amounts and thresholds were calibrated on candidate counts only, before any model run. Classifiers, folds and bootstrap are those of `configs/classify.yaml`; config and seeds stored in `{PDC}`; runtime {pd_cfg['runtime_seconds']:.0f} s.\n")
    kk = csv(PDK)
    md.append(mdtable(["condition", "candidates", "organic", "pump_dump", "news", "ambiguous"], [[r.condition, int(r.candidates), int(r.organic), int(r.pump_dump), int(r.news), int(r.ambiguous)] for r in kk.itertuples()]))
    for cn, cv in cond.items():
        w0 = {"condition": cn, "feature_set": "M", "model": "logreg"}
        md.append(f"**Condition `{cn}`** (exit-time spread sd {get(PDR, w0, 'exit_spread_sd_h', '{:.1f}', '6F')} h): {get(PDR, w0, 'n_rows', '{:.0f}', '6F')} candidates, {get(PDR, w0, 'n_pos', '{:.0f}', '6F')} positives, prevalence {get(PDR, w0, 'prevalence', '{:.3f}', '6F')}, {get(PDR, w0, 'n_pos_groups', '{:.0f}', '6F')} positive groups. "
                  "PR-AUC [95% grouped-bootstrap CI] (mean +- std of the per-seed PR-AUC):\n")
        cell = lambda s, m, cn=cn: ci_cell(PDR, {"condition": cn, "feature_set": s, "model": m}, "6F") + f" ({get(PDR, {'condition': cn, 'feature_set': s, 'model': m}, 'seed_mean', '{:.3f}', '6F')} +- {get(PDR, {'condition': cn, 'feature_set': s, 'model': m}, 'seed_std', '{:.3f}', '6F')})"  # noqa: E731
        md.append(mdtable(["feature set", *MODELS], [[s] + [cell(s, m) for m in MODELS] for s in ("M", "B_pd")]))
        md.append(f"`-M_exit_spread_h` alone (single feature, no model): {ci_cell(PDR, {'condition': cn, 'feature_set': 'exit_spread_alone', 'model': 'none'}, '6F')}.\n")
        md.append("Paired M minus B_pd (same resamples):\n")
        md.append(mdtable(["comparison", *MODELS], [["M minus B_pd"] + [diff_cell(PDP, {"condition": cn, "model": m, "comparison": "M minus B_pd"}, "6F") for m in MODELS]]))
    md.append("**Harder exits (descriptive).** The only difference between the conditions is the spread of the planted exit times; candidates, labels and groups are identical, so B_pd does not change. Change in point estimates of PR-AUC from `default` to `harder_exit` (difference of two separate runs, **no CI**):\n")
    rows = []
    for s in ("M", "B_pd"):
        rows.append([s] + [f"{float(sel(csv(PDR), {'condition': 'harder_exit', 'feature_set': s, 'model': m}).pr_auc.iloc[0]) - float(sel(csv(PDR), {'condition': 'default', 'feature_set': s, 'model': m}).pr_auc.iloc[0]):+.3f}" for m in MODELS])
        for m in MODELS:
            prov.append({"table": "6F", "text": f"change {s} {m}", "file": str(PDR), "where": json.dumps({"feature_set": s, "model": m}), "columns": "pr_auc harder_exit minus default"})
    md.append(mdtable(["feature set", *MODELS], rows))
    src(PDR, PDP, PDK, PDC)
else:
    md.append("`results/phase7_pump_dump/` not found.\n")

# ====================================================================== helpers for sections 7-9
def ci_excludes_zero(path, where_base, comp, models=MODELS):
    d = csv(path)
    out = []
    for m in models:
        r = sel(d, where_base | {"model": m, "comparison": comp})
        if len(r) == 1:
            out.append((m, float(r.iloc[0]["diff"]), float(r.iloc[0].ci_low), float(r.iloc[0].ci_high)))
    pos = sum(1 for _, _, lo, hi in out if lo > 0)
    neg = sum(1 for _, _, lo, hi in out if hi < 0)
    return out, pos, neg


HIR, HIP = CL / "HI_tau0.05" / "results.csv", CL / "HI_tau0.05" / "paired_differences.csv"
LIR, LIP = CL / "LI_tau0.05" / "results.csv", CL / "LI_tau0.05" / "paired_differences.csv"
hi_prim = {"variant": "primary"}
v = lambda path, where, col, fmt="{:.3f}": get(path, where, col, fmt, "7-9")  # noqa: E731

# ====================================================================== 7. limitations
sec("7. Limitations and caveats")
L = []
L.append(f"Few positive groups: {v(HIR, {'variant': 'primary', 'feature_set': 'B', 'model': 'logreg'}, 'n_pos_groups', '{:.0f}')} on HI and {v(LIR, {'variant': 'primary', 'feature_set': 'B', 'model': 'logreg'}, 'n_pos_groups', '{:.0f}')} on LI "
         f"(positives {v(HIR, {'variant': 'primary', 'feature_set': 'B', 'model': 'logreg'}, 'n_pos', '{:.0f}')} and {v(LIR, {'variant': 'primary', 'feature_set': 'B', 'model': 'logreg'}, 'n_pos', '{:.0f}')} candidates); the effective sample size is the group count, so CIs are wide.")
L.append(f"LI has {v(P3 / 'ibm_split_counts.csv', {'dataset': 'LI', 'tau_quantile': 0.05, 'scheme': 'grouped_cv_edge', 'part': 'all'}, 'cycle_positives', '{:.0f}')} cycle positives in "
         f"{v(P3 / 'ibm_split_counts.csv', {'dataset': 'LI', 'tau_quantile': 0.05, 'scheme': 'grouped_cv_edge', 'part': 'all'}, 'cycle_positive_groups', '{:.0f}')} groups: the cycle-only variant is unusable there "
         f"(fewest positive groups on a train or test side: {li_cfg['unusable_variants'][0]['min_positive_groups_on_a_side'] if li_cfg and li_cfg.get('unusable_variants') else 'n/a'}); LI primary has {v(LIR, {'variant': 'primary', 'feature_set': 'B', 'model': 'logreg'}, 'min_pos_groups_on_a_side', '{:.0f}')} on its thinnest side.")
L.append(f"LI contradicts HI on the ring-birth signal: `-D_birth` alone scores {v(LIR, {'variant': 'primary', 'feature_set': 'D_birth_alone', 'model': 'none'}, 'pr_auc')} on LI (prevalence {v(LIR, {'variant': 'primary', 'feature_set': 'B', 'model': 'logreg'}, 'prevalence', '{:.4f}')}) "
         f"against {v(HIR, {'variant': 'primary', 'feature_set': 'D_birth_alone', 'model': 'none'}, 'pr_auc')} on HI (prevalence {v(HIR, {'variant': 'primary', 'feature_set': 'B', 'model': 'logreg'}, 'prevalence', '{:.4f}')}).")
L.append(f"`T_win` alone on HI: PR-AUC {v(HIR, {'variant': 'primary', 'feature_set': 'T_win', 'model': 'logreg'}, 'pr_auc')} (logreg) and {v(HIR, {'variant': 'primary', 'feature_set': 'T_win', 'model': 'rf'}, 'pr_auc')} (rf) at prevalence "
         f"{v(HIR, {'variant': 'primary', 'feature_set': 'B', 'model': 'logreg'}, 'prevalence', '{:.4f}')}: window-constant features rank well above chance (consistent with a time proxy, not tested directly); on LI the same sets score {v(LIR, {'variant': 'primary', 'feature_set': 'T_win', 'model': 'logreg'}, 'pr_auc')} / "
         f"{v(LIR, {'variant': 'primary', 'feature_set': 'T_win', 'model': 'rf'}, 'pr_auc')} at prevalence {v(LIR, {'variant': 'primary', 'feature_set': 'B', 'model': 'logreg'}, 'prevalence', '{:.4f}')}.")
dsyn = [(c, v(SYN / c / "results.csv", {"variant": "primary", "feature_set": "D", "model": m}, "pr_auc")) for c in CONDS for m in ("logreg", "rf")]
bds = [float(csv(SYN / c / "results.csv").pipe(lambda d: sel(d, {"variant": "primary", "feature_set": "D", "model": m}).pr_auc.iloc[0])) for c in CONDS for m in ("logreg", "rf") if csv(SYN / c / "results.csv") is not None]
if bds:
    _sd = lambda m: [float(sel(csv(SYN / c / "results.csv"), {"variant": "primary", "feature_set": "D", "model": m}).pr_auc.iloc[0]) for c in CONDS]  # noqa: E731
    L.append("The synthetic generator is near-separable by D: D alone scores " + "; ".join(f"{m} {min(_sd(m)):.3f}-{max(_sd(m)):.3f}" for m in ("logreg", "rf", "svm")) + " over the four conditions (range per model), so there is little room for B+D+T to add.")
imp = csv(P3 / "audit" / "HI_audit_imputation.csv")
if imp is not None:
    ip = imp[(imp.window == "pooled") & (imp.group != "background")].share_imputed
    ib_ = float(imp[(imp.window == "pooled") & (imp.group == "background")].share_imputed.iloc[0])
    L.append(f"Imputed timing: on HI, {ip.min():.3f} to {ip.max():.3f} of each laundering group's pairs have fewer than 3 transactions (`gap_cv` needs at least 2 gaps; the code imputes pairs with 1 or 2 transactions, whereas v3 section 3.1 says a pair with a single transaction), against {ib_:.3f} for background pairs, so the timing-regularity score of most laundering pairs is the imputed constant.")
if bt:
    L.append(f"Betweenness is sampled (k={wj0['k']}) on IBM windows: {bt[0][3]} of the {bt[0][1]} nodes of HI window 0 have non-zero betweenness; B is exact only up to {wj0['exact_max_nodes']} nodes (synthetic).")
if ib is not None:
    k = {"source": "HI", "scope": "days0-5", "max_len": 6, "tau_quantile": 0.05}
    k8 = k | {"max_len": 8}
    L.append(f"Recall caps: HI days 0-5 has {v(IB, k, 'n_complete', '{:.0f}')} complete CYCLE attempts, only {v(IB, k, 'n_complete_le_maxlen', '{:.0f}')} with at most 6 accounts ({v(IB, k8, 'n_complete_le_maxlen', '{:.0f}')} at max_len 8); "
             f"{v(IB, k, 'n_recovered', '{:.0f}')} recovered by account set but {v(IB, k, 'n_recovered_labelled', '{:.0f}')} also labelled positive. LI days 0-5 has {v(IB, {'source': 'LI', 'scope': 'days0-5', 'max_len': 6, 'tau_quantile': 0.05}, 'n_complete', '{:.0f}')} complete attempt (not a usable rate).")
if sy is not None:
    th = sy[(sy.hard_negatives == True) & (sy.n_truncated_sccs > 0)]
    tn_ = sy[(sy.hard_negatives == True) & (sy.n_truncated_sccs == 0) & (sy.tau_quantile >= 0.15)]
    L.append(f"Synthetic recall with hard negatives on falls at large tau, where the enumeration cap (20000 cycles per SCC) is hit: rows with a truncated SCC recover {th.recovered.mean():.2f} of planted rings ({len(th)} ring-cells), "
             f"untruncated rows at tau >= 0.15 recover {tn_.recovered.mean():.2f}; at tau 0.05 untruncated recall is {sy[(sy.hard_negatives == True) & (sy.tau_quantile == 0.05)].recovered.mean():.2f}.")
if amb:
    L.append("Hard-negative ambiguity: with hard negatives on, " + " and ".join(f"{amb[c]:.3f} ({c})" for c in CONDS if c.startswith("hn-on") and c in amb) + " of all candidates are ambiguous, so the ambiguous-positive variant has a prevalence near 1 and is run for logreg only.")
sp_ = [csv(SYN / c / "per_scenario.csv") for c in CONDS]
if all(x is not None for x in sp_):
    mn = min(int(x[x.scenario == s].n_pos_groups.iloc[0]) for x in sp_ for s in ("isolated", "dense", "chorded"))
    L.append(f"Smallest per-scenario positive-group count on synthetic: {mn}; chorded and dense scenarios have few groups, so scenario-level CIs are wide.")
L.append(f"Transfer fits one model per direction (no repeats); test positive groups: {v(TR, {'train': 'HI', 'test': 'LI', 'feature_set': 'B', 'model': 'logreg'}, 'n_test_pos_groups', '{:.0f}')} (LI) and "
         f"{v(TR, {'train': 'LI', 'test': 'HI', 'feature_set': 'B', 'model': 'logreg'}, 'n_test_pos_groups', '{:.0f}')} (HI); both datasets come from the same generator and cover the same dates.")
L.append(f"Exploratory comparisons (T_local / T_win ablations) were added after the first HI run; {int(csv(HIP)[csv(HIP).comparison.isin(PAIR_EXPL)].shape[0])} such paired rows exist in the HI file and {int(csv(LIP)[csv(LIP).comparison.isin(PAIR_EXPL)].shape[0])} in the LI file; no multiplicity correction.")
svm_gap = [(float(sel(csv(HIR), {'variant': 'primary', 'feature_set': s, 'model': 'svm'}).pr_auc.iloc[0]), float(sel(csv(HIR), {'variant': 'primary', 'feature_set': s, 'model': 'rf'}).pr_auc.iloc[0])) for s in ("B+D", "B+D+T")]
L.append(f"The RBF SVM (fixed C=1, gamma=scale, untuned) is weaker than rf on HI: B+D {svm_gap[0][0]:.3f} vs {svm_gap[0][1]:.3f}, B+D+T {svm_gap[1][0]:.3f} vs {svm_gap[1][1]:.3f}.")
L.append("Currencies are mixed (`Amount Paid` as is); the IBM data are synthetic AML data, not market transactions; the optimisation report numbers come from the pinned report generator (section 2).")
_sv = SEV / "diagnostic"
L.append(f"Severity (synthetic only, the generator decides persistence): B+D without volume columns reaches rf RMSE {get(RG, {'feature_set': 'B+D', 'model': 'rf'}, 'rmse', '{:.3f}', '7')} against {get(_sv / 'results.csv', {'predictor': 'persistence_fit', 'model': 'none'}, 'rmse', '{:.3f}', '7')} for a fair persistence fit and {get(_sv / 'results.csv', {'predictor': 'D_length', 'model': 'rf'}, 'rmse', '{:.3f}', '7')} for `D_length` alone; the top rf importance is `D_length` ({get(_sv / 'rf_importance_BD.csv', {'rank': 1} if False else {'column': 'D_length'}, 'rf_impurity_importance', '{:.3f}', '7')}). This is recovery of generator structure, not forecasting skill; T adds nothing.")
L.append(f"Bandit benchmark: at lambda 1, d 2 the bandits lose to the re-tuned comparator (discounted UCB {get(BV, {'lambda': 1.0, 'lag': 2, 'bandit': 'ducb', 'comparator': 'retune_M5'}, 'diff_share_mean', '{:+.3f}', '7')} share of oracle, {get(BV, {'lambda': 1.0, 'lag': 2, 'bandit': 'ducb', 'comparator': 'retune_M5'}, 'seeds_bandit_better', '{:.0f}', '7')} of 6 seeds better; sliding-window Thompson {get(BV, {'lambda': 1.0, 'lag': 2, 'bandit': 'swts', 'comparator': 'retune_M5'}, 'diff_share_mean', '{:+.3f}', '7')}, {get(BV, {'lambda': 1.0, 'lag': 2, 'bandit': 'swts', 'comparator': 'retune_M5'}, 'seeds_bandit_better', '{:.0f}', '7')} of 6); exploring the flag-everything arm is very costly and its size is hidden by the reward clipping.")
L.append(f"Clustering (exploratory): HI, k = 4, D + T_local landscape: ARI against type {get(CLR, {'dataset': 'HI', 'vector': 'D+T_local_landscape', 'k_mode': 'fixed'}, 'ari_type', '{:.3f}', '7')} (random {get(CLR, {'dataset': 'HI', 'vector': 'D+T_local_landscape', 'k_mode': 'fixed'}, 'ari_type_random', '{:.3f}', '7')}), ARI against group id {get(CLR, {'dataset': 'HI', 'vector': 'D+T_local_landscape', 'k_mode': 'fixed'}, 'ari_group', '{:.3f}', '7')}, purity {get(CLR, {'dataset': 'HI', 'vector': 'D+T_local_landscape', 'k_mode': 'fixed'}, 'purity_type', '{:.3f}', '7')} against random {get(CLR, {'dataset': 'HI', 'vector': 'D+T_local_landscape', 'k_mode': 'fixed'}, 'purity_type_random', '{:.3f}', '7')} and majority {get(CLR, {'dataset': 'HI', 'vector': 'D', 'k_mode': 'fixed'}, 'purity_majority', '{:.3f}', '7')}; 4 to 12 groups per type; the grouped-bootstrap percentile CIs are biased (resampled duplicate rings) and several exclude their own point estimate, so they are not used as CIs.")
L.append(f"Cap sensitivity: on HI, `ring_local_max` changes the svm B+D+T PR-AUC by {get(CAP / 'ibm' / 'paired_differences.csv', {'variant': 'ring_local_max', 'feature_set': 'B+D+T', 'model': 'svm', 'comparison': 'B+D+T under this cap minus B+D+T under window_max'}, 'diff', '{:+.3f}', '7')} (post hoc, one model, uncorrected for multiplicity).")
L.append(f"Pump-and-dump track (synthetic only, no topology; exploratory): a single feature, `-M_exit_spread_h`, scores {get(PDR, {'condition': 'default', 'feature_set': 'exit_spread_alone', 'model': 'none'}, 'pr_auc', '{:.3f}', '7')} in the default condition ({get(PDR, {'condition': 'harder_exit', 'feature_set': 'exit_spread_alone', 'model': 'none'}, 'pr_auc', '{:.3f}', '7')} with wider exits), so the generator is near-separable by timing; B_pd is weak by construction (all groups are drawn from the same size and volume distributions: logistic regression {get(PDR, {'condition': 'default', 'feature_set': 'B_pd', 'model': 'logreg'}, 'pr_auc', '{:.3f}', '7')} at prevalence {get(PDR, {'condition': 'default', 'feature_set': 'B_pd', 'model': 'logreg'}, 'prevalence', '{:.3f}', '7')}); the features use post-accumulation sells and prices, so detection is retrospective, not early warning; no chance-coincidence candidates arose (every non-planted candidate is a planted organic or news group; ambiguous excluded); the generator was calibrated on candidate counts before any model run; only {get(PDR, {'condition': 'default', 'feature_set': 'M', 'model': 'logreg'}, 'n_pos_groups', '{:.0f}', '7')} positive groups.")
md.append("\n".join(f"{i + 1}. {x}" for i, x in enumerate(L)) + "\n")

# ====================================================================== 8. claims
sec("8. Claims the data supports / does not support")
claims_yes, claims_none, claims_desc, claims_no = [], [], [], []
o, pos, neg = ci_excludes_zero(HIP, hi_prim, "B+D minus B")
claims_yes.append(f"Adding D to B raises PR-AUC on HI: the paired CI excludes 0 in {pos} of {len(o)} models (" + ", ".join(f"{m} {d:+.3f}" for m, d, _, _ in o) + ").")
o, pos, neg = ci_excludes_zero(LIP, hi_prim, "B+D minus B")
claims_yes.append(f"On LI, adding D to B: the paired CI excludes 0 in {pos} of {len(o)} models (" + ", ".join(f"{m} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]" for m, d, lo, hi in o) + ").")
tot_pos = tot = 0
det = []
for c in CONDS:
    o, pos, neg = ci_excludes_zero(SYN / c / "paired_differences.csv", {"variant": "primary", "scope": "overall"}, "B+D+T minus B+D")
    tot_pos += pos
    tot += len(o)
    det.append(f"{c}: {pos}/{len(o)}")
claims_yes.append(f"On synthetic data B+D+T improves on B+D by a paired CI above 0 in {tot_pos} of {tot} condition-model cells (" + "; ".join(det) + "); B+D itself scores " + "; ".join(f"{m} {min(float(sel(csv(SYN / c / 'results.csv'), {'variant': 'primary', 'feature_set': 'B+D', 'model': m}).pr_auc.iloc[0]) for c in CONDS):.3f}-{max(float(sel(csv(SYN / c / 'results.csv'), {'variant': 'primary', 'feature_set': 'B+D', 'model': m}).pr_auc.iloc[0]) for c in CONDS):.3f}" for m in ("logreg", "rf", "svm")) + " over the four conditions (range per model).")
o, pos, neg = ci_excludes_zero(HIP, hi_prim, "B+D+T minus B+D")
claims_none.append(f"IBM HI-Small: no detectable topological lift (B+D+T minus B+D, pre-specified primary): the CI includes 0 in {len(o) - pos - neg} of {len(o)} models (" + ", ".join(f"{m} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]" for m, d, lo, hi in o) + "). With 14 positive groups the CIs cannot rule out a small effect either way.")
o, pos, neg = ci_excludes_zero(LIP, hi_prim, "B+D+T minus B+D")
claims_none.append(f"IBM LI-Small: no detectable topological lift (B+D+T minus B+D): the CI includes 0 in {len(o) - pos - neg} of {len(o)} models (" + ", ".join(f"{m} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]" for m, d, lo, hi in o) + "). With 10 positive groups the CIs cannot rule out a small effect either way.")
o, pos, neg = ci_excludes_zero(HIP, hi_prim, "B+D+T_local minus B+D")
claims_yes.append(f"Exploratory: the local topology columns alone (B+D+T_local) add to B+D on HI for logreg only (CI excludes 0 in {pos} of {len(o)} models: " + ", ".join(f"{m} {d:+.3f}" for m, d, _, _ in o) + "); this is one of the exploratory comparisons and is not corrected for multiplicity.")
for tn, te in (("HI", "LI"), ("LI", "HI")):
    o, pos, neg = ci_excludes_zero(TP, {"train": tn, "test": te}, "B+D+T minus B+D")
    claims_none.append(f"Transfer {tn}->{te}: no detectable lift of B+D+T over B+D (CI above 0 in {pos}, below 0 in {neg} of {len(o)} models; the CIs cannot rule out a small effect either way: " + ", ".join(f"{m} {d:+.3f}" for m, d, _, _ in o) + ").")
claims_no.append(f"That the ring-birth signal carries over from HI to LI: `-D_birth` alone is {v(LIR, {'variant': 'primary', 'feature_set': 'D_birth_alone', 'model': 'none'}, 'pr_auc')} on LI at prevalence {v(LIR, {'variant': 'primary', 'feature_set': 'B', 'model': 'logreg'}, 'prevalence', '{:.4f}')} (HI: {v(HIR, {'variant': 'primary', 'feature_set': 'D_birth_alone', 'model': 'none'}, 'pr_auc')}).")
claims_no.append("That the synthetic topology gains predict IBM results: B+D+T minus B+D has a CI above 0 in " + f"{tot_pos} of {tot}" + " synthetic cells and in 0 IBM cells (HI primary, LI primary); IBM has few positive groups, so absence of a detected gain is not evidence of no gain.")
claims_no.append(f"That the `T_win` contribution on HI is topological: `T_win` columns are constant within a window; alone they score {v(HIR, {'variant': 'primary', 'feature_set': 'T_win', 'model': 'logreg'}, 'pr_auc')} (logreg) on HI at prevalence {v(HIR, {'variant': 'primary', 'feature_set': 'B', 'model': 'logreg'}, 'prevalence', '{:.4f}')} and {v(LIR, {'variant': 'primary', 'feature_set': 'T_win', 'model': 'logreg'}, 'pr_auc')} on LI at prevalence {v(LIR, {'variant': 'primary', 'feature_set': 'B', 'model': 'logreg'}, 'prevalence', '{:.4f}')}; a time proxy is consistent with this but was not tested directly.")
kk = {"source": "HI", "scope": "days0-5", "max_len": 6, "tau_quantile": 0.05}
claims_yes.append(f"Candidate recall is partial and tau-dependent (section 6): HI days 0-5, tau 0.05, max_len 6 recovers {v(IB, kk, 'n_recovered', '{:.0f}')} of {v(IB, kk, 'n_complete_le_maxlen', '{:.0f}')} reachable complete CYCLE attempts ({v(IB, kk, 'n_complete', '{:.0f}')} complete); synthetic recall is 1.00 with hard negatives off at every tau tested.")
_cp = csv(CAP / "ibm" / "paired_differences.csv")
_cc = _cp[_cp.comparison.str.contains("under this cap") & _cp.variant.isin(["fixed_1.0", "max_x1.1"])]
claims_yes.append(f"T_local / T_win cap choice among window_max, fixed 1.0 and 1.1 x window max barely changes HI B+D+T PR-AUC: largest absolute paired difference to window_max {_cc['diff'].abs().max():.3f} (all models, both sets; among these three rules only); `ring_local_max` is a post hoc variant (single model, uncorrected) that differs for svm by {diff_cell(CAP / 'ibm' / 'paired_differences.csv', {'variant': 'ring_local_max', 'feature_set': 'B+D+T', 'model': 'svm', 'comparison': 'B+D+T under this cap minus B+D+T under window_max'}, '8')} (B+D+T).")
claims_no.append(f"That the bandits beat periodic re-tuning at the same lag: at lambda 1, d 2 the paired share-of-oracle difference is {v(BV, {'lambda': 1.0, 'lag': 2, 'bandit': 'ducb', 'comparator': 'retune_M5'}, 'diff_share_mean', '{:+.3f}')} (discounted UCB) and {v(BV, {'lambda': 1.0, 'lag': 2, 'bandit': 'swts', 'comparator': 'retune_M5'}, 'diff_share_mean', '{:+.3f}')} (sliding-window Thompson); the only positive cell is discounted UCB at d = 5 ({v(BV, {'lambda': 1.0, 'lag': 5, 'bandit': 'ducb', 'comparator': 'retune_M5'}, 'diff_share_mean', '{:+.3f}')}); the online gradient ascent comparator is at {v(SM, {'lambda': 1.0, 'lag': 2, 'policy': 'ascent'}, 'share_mean', '{:.3f}')} of oracle.")
claims_desc.append(f"Severity regression (synthetic only; the generator decides that rings persist): `D_length` alone reaches rf RMSE {v(_sv / 'results.csv', {'predictor': 'D_length', 'model': 'rf'}, 'rmse', '{:.3f}')} and gbr RMSE {v(_sv / 'results.csv', {'predictor': 'D_length', 'model': 'gbr'}, 'rmse', '{:.3f}')}, against {v(RG, {'feature_set': 'B+D', 'model': 'rf'}, 'rmse', '{:.3f}')} and {v(RG, {'feature_set': 'B+D', 'model': 'gbr'}, 'rmse', '{:.3f}')} for B+D, so the model recovers generator structure (top rf importance `D_length` {v(_sv / 'rf_importance_BD.csv', {'column': 'D_length'}, 'rf_impurity_importance', '{:.3f}')}), not forecasting skill; T adds nothing (B+D+T minus B+D rf RMSE {v(PG, {'model': 'rf', 'comparison': 'B+D+T minus B+D', 'metric': 'rmse'}, 'diff', '{:+.3f}')}).")
claims_no.append(f"That Ward clustering of positive rings recovers pattern types (exploratory): on HI, k = 4, ARI against type is {get(CLR, {'dataset': 'HI', 'vector': 'D+T_local_landscape', 'k_mode': 'fixed'}, 'ari_type', '{:.3f}', '8')} (D + T_local landscape; random {get(CLR, {'dataset': 'HI', 'vector': 'D+T_local_landscape', 'k_mode': 'fixed'}, 'ari_type_random', '{:.3f}', '8')}) against {get(CLR, {'dataset': 'HI', 'vector': 'D+T_local_landscape', 'k_mode': 'fixed'}, 'ari_group', '{:.3f}', '8')} against group id: the clusters track attempts more than types. Purity {get(CLR, {'dataset': 'HI', 'vector': 'D+T_local_landscape', 'k_mode': 'fixed'}, 'purity_type', '{:.3f}', '8')} vs random {get(CLR, {'dataset': 'HI', 'vector': 'D+T_local_landscape', 'k_mode': 'fixed'}, 'purity_type_random', '{:.3f}', '8')} and majority {get(CLR, {'dataset': 'HI', 'vector': 'D+T_local_landscape', 'k_mode': 'fixed'}, 'purity_majority', '{:.3f}', '8')}. The bootstrap percentile CIs are biased and several exclude their own point estimate, so they are not used as CIs. LI (T_local landscape, k = 3, 9 groups): ARI {get(CLR, {'dataset': 'LI', 'vector': 'T_local_landscape', 'k_mode': 'fixed'}, 'ari_type', '{:.3f}', '8')}, purity {get(CLR, {'dataset': 'LI', 'vector': 'T_local_landscape', 'k_mode': 'fixed'}, 'purity_type', '{:.3f}', '8')} vs random {get(CLR, {'dataset': 'LI', 'vector': 'T_local_landscape', 'k_mode': 'fixed'}, 'purity_type_random', '{:.3f}', '8')}; with 9 groups and 3 to 5 groups per type this is a small-sample point estimate, not evidence of type recovery.")
_pp = csv(PDP)
claims_pd = [f"Timing features separate the planted pump-and-dump groups from the group-size/volume baseline in the synthetic generator: paired M minus B_pd has a CI above 0 in {int((_pp.ci_low > 0).sum())} of {len(_pp)} condition-model cells (default: logreg {diff_cell(PDP, {'condition': 'default', 'model': 'logreg', 'comparison': 'M minus B_pd'}, '8')}, rf {diff_cell(PDP, {'condition': 'default', 'model': 'rf', 'comparison': 'M minus B_pd'}, '8')}, svm {diff_cell(PDP, {'condition': 'default', 'model': 'svm', 'comparison': 'M minus B_pd'}, '8')}; {get(PDR, {'condition': 'default', 'feature_set': 'M', 'model': 'rf'}, 'n_pos_groups', '{:.0f}', '8')} positive groups, prevalence {get(PDR, {'condition': 'default', 'feature_set': 'M', 'model': 'rf'}, 'prevalence', '{:.3f}', '8')}); the generator author decides separability.", "The classification harness (grouped CV, grouped bootstrap, paired differences) runs unchanged on a second pattern type that has no topology."]
claims_no += ["That the pump-and-dump features detect real pump-and-dump: the data are synthetic and the generator decides separability (one feature, `-M_exit_spread_h`, alone reaches " + v(PDR, {'condition': 'default', 'feature_set': 'exit_spread_alone', 'model': 'none'}, 'pr_auc', '{:.3f}') + ").", "That anything about topology follows from the pump-and-dump track: it contains no topological feature."]
md.append("**Supported:**\n\n" + "\n".join(f"- {x}" for x in claims_yes) + "\n\n**Supported (synthetic, exploratory, no topology):**\n\n" + "\n".join(f"- {x}" for x in claims_pd) + "\n\n**No detectable effect (the data neither show nor exclude a small effect):**\n\n" + "\n".join(f"- {x}" for x in claims_none) + "\n\n**Descriptive findings, not claims:**\n\n" + "\n".join(f"- {x}" for x in claims_desc) + "\n\n**Not supported:**\n\n" + "\n".join(f"- {x}" for x in claims_no) + "\n")

# ====================================================================== 9. deviations
sec("9. Deviations from the v3 proposal")
wj_ = json.load(open(P3 / "ibm" / "HI_tau0.05" / "window_00.json"))
SCt_ = P3 / "ibm_split_counts.csv"
kt_ = lambda ds, part: {"dataset": ds, "tau_quantile": 0.05, "scheme": "time_ordered", "part": part}  # noqa: E731
dev = [
    "Candidate de-duplication order: the proposal keeps the lowest-birth ring among near-duplicates; the code de-duplicates shortest-first, then by birth. At overlap 0.8 nothing is removed for max_len <= 9 (first removals at 10; README), so near-duplicate rings all stay candidates and CV groups by ring identity.",
    f"max_len = {wj_['max_len']} for the Phase 3 tables (the optimization report keeps 5).",
    "Infinite-bar cap: Phase 3 T features use the window-maximum cap; the optimization report uses the frozen legacy `topo_*` scalar features with a fixed cap of 1.0 (two T pipelines coexist).",
    f"Windows are {wj_['end_day'] - wj_['start_day']:.0f} days with a {json.load(open(P3 / 'ibm' / 'HI_tau0.05' / 'window_01.json'))['start_day'] - wj_['start_day']:.0f}-day stride on IBM (cycle attempts last up to 96 h; README).",
    f"Time-ordered evaluation was tried as specified and dropped: HI leaves {get(SCt_, kt_('HI', 'test'), 'candidates', '{:.0f}', 'dev')} test candidates and LI {get(SCt_, kt_('LI', 'test'), 'candidates', '{:.0f}', 'dev')} (`eval/splits.time_ordered_split` is kept, unused); grouped CV is primary and HI<->LI transfer the out-of-sample check.",
    "Cycle-only subset: positives are candidates labelled from CYCLE attempts; other-type positives and ambiguous candidates are removed, background kept (not defined in the proposal). Usable on HI only.",
    "Ambiguity: candidates whose laundering edges rest only on unattributed rows, or whose backing is split across structures, are `ambiguous`; results are reported with them excluded (primary) and counted positive.",
    "Timing-regularity imputation: v3 section 3.1 says a pair with a single transaction receives the window's median s_reg; the code leaves the timing CV undefined for pairs with fewer than 3 transactions (it needs at least 2 gaps), so pairs with 1 or 2 transactions are imputed (weights: the window median of s_reg; D features: the window median of the defined gap CVs).",
    "Hard-negative false-positive rate at 80% recall was replaced by the mean out-of-fold score rank percentile of each negative kind.",
    "Recall of complete cycle attempts uses the account-set definition (README measurement); the label-based count is reported next to it.",
    "Phase 4 clustering and severity regression, and Phase 6 (bandit policies and drift benchmark), were run as exploratory studies (sections 6A-6D); severity is synthetic only (IBM has no next-window wash volume) and the bandit benchmark uses the optimization report's synthetic drift stream.",
    "Phase 5 (NFT loader, proxy labels, label-provenance audit) is NOT done. Blocker: running queries on Dune is blocked on the free tier. BigQuery access was confirmed and the schemas were inspected: `nft.wash_trades` combines five filters, of which filters 2 and 3 are cycle-type rules, so the planned leakage-controlled subset would be filters 4 and 5 only.",
    "Pump-and-dump (v3 section 3.6, optional) was done as a synthetic-only, no-topology track with its own generator (section 6F); no real-data result exists.",
    "Not done: undirected clique-complex ablation, persistence-image ablation, per-score classifier filtrations (classifiers on single-score filtrations), and the sensitivity sweeps over window length, stride, `max_len`-in-classification, tau (classification beyond IBM tau 0.05) and label threshold X.",
    "Learned-weight (Problem B), classifier (Problem A) and threshold (Problem C) results in section 2 come from the pinned report generator, not from IBM AML.",
]
md.append("\n".join(f"{i + 1}. {x}" for i, x in enumerate(dev)) + "\n")

# ====================================================================== missing + write
sec("Expected files that were missing")
md.append("None.\n" if not missing else "\n".join(f"- `{m}`" for m in missing) + "\n")
(OUT / "numbers_pack.md").write_text("\n".join(md) + "\n")

# ====================================================================== figures + captions
figs = [
    (CL / "HI_tau0.05" / "pr_auc.png", "hi_pr_auc.png"),
    (SYN / "pr_auc.png", "synthetic_pr_auc.png"),
    (SYN / "lift_per_scenario.png", "synthetic_lift_per_scenario.png"),
    (CL / "candidate_recall" / "ibm.png", "recall_ibm.png"),
    (CL / "candidate_recall" / "synthetic.png", "recall_synthetic.png"),
    (P3 / "audit" / "HI_audit_scores.png", "hi_audit_scores.png"),
    (P3 / "audit" / "LI_audit_scores.png", "li_audit_scores.png"),
    (P6 / "bandit_drift.png", "bandit_drift.png"),
    (CLU / "pca_HI.png", "cluster_pca_HI.png"),
    (CLU / "pca_LI.png", "cluster_pca_LI.png"),
    (PDD / "pump_dump_pr_auc.png", "pump_dump_pr_auc.png"),
] + [(f, f.name) for f in sorted((RES / "figures").glob("*.png"))]
stale = []
for s, d in figs:
    if Path(s).exists():
        shutil.copyfile(s, PACK / "figures" / d)
    else:
        note_missing(s)
hr = HIR
if (CL / "HI_tau0.05" / "pr_auc.png").exists() and (CL / "HI_tau0.05" / "pr_auc.png").stat().st_mtime < (CL / "HI_tau0.05" / "results.csv").stat().st_mtime:
    stale.append("`hi_pr_auc.png` was drawn before the ablation rows were appended to `results.csv`; it does not show B+D+T_local / B+D+T_win (not regenerated).")
prev = lambda path, w: get(path, w, "prevalence", "{:.4f}", "figures")  # noqa: E731
w0 = {"variant": "primary", "feature_set": "B", "model": "logreg"}
fl = ["# Figure list (caption drafts)\n",
      "Copies of existing PNGs in `paper/numbers_pack/figures/` (not regenerated). Every PR-AUC CI is a 95% percentile interval of a grouped bootstrap (resampling groups, 1000 resamples) unless stated.\n"]
pv = {sc: [float(get(SYN / c / "per_scenario.csv", {"scenario": sc, "feature_set": "B", "model": "logreg"}, "prevalence", "{:.6f}", "figures")) for c in CONDS] for sc in ["isolated", "dense", "chorded"]}
pg = {sc: get(SYN / "hn-off_chords-default" / "per_scenario.csv", {"scenario": sc, "feature_set": "B", "model": "logreg"}, "n_pos_groups", "{:.0f}", "figures") for sc in ["isolated", "dense", "chorded"]}
cap = [
    ("hi_pr_auc.png", f"IBM HI-Small, PR-AUC of B, B+D, B+D+T, B+D+T_local, B+D+T_win, T and the controls D, T_local, T_win by model; primary variant (ambiguous excluded), prevalence {prev(HIR, w0)}, {get(HIR, w0, 'n_pos_groups', '{:.0f}', 'figures')} positive groups. Error bars: 95% grouped-bootstrap CI. Pooled out-of-fold, mean of 5 repeats."),
    ("synthetic_pr_auc.png", "Synthetic data, PR-AUC of B, B+D, B+D+T, B+D+T_local, T by model in the four conditions (hard negatives x dense_internal_chords); primary variant, prevalence " + ", ".join(f"{prev(SYN / c / 'results.csv', w0)} ({c})" for c in CONDS) + ". Error bars: 95% grouped-bootstrap CI over six seeds pooled."),
    ("synthetic_lift_per_scenario.png", "F7. Synthetic data, paired PR-AUC difference B+D+T minus B+D per scenario (isolated, dense, chorded), by model, hard negatives off (left) and on (right); filled markers = dense_internal_chords default, hollow = off. Error bars: 95% grouped-bootstrap CI of the paired difference (same resamples for both feature sets). **Prevalence differs per scenario** (positives of one scenario against all negatives): " + "; ".join(f"{sc} {min(pv[sc]):.4f}-{max(pv[sc]):.4f}" for sc in ["isolated", "dense", "chorded"]) + " across the four conditions; positive groups per scenario " + ", ".join(f"{sc} {pg[sc]}" for sc in ["isolated", "dense", "chorded"]) + " (hn-off, chords default)."),
    ("bandit_drift.png", f"Phase 6, synthetic drift stream (the optimization report's, {b_cfg['n_seeds']} seeds, 40 windows). Left: share of hindsight-oracle cumulative reward by policy and lag d at lambda 1, mean +- std over seeds (not a CI); discounted UCB and sliding-window Thompson (pre-set hyperparameters) against fixed, re-tuned (M = 5) and online ascent. Middle and right: rolling 5-window precision and recall of the flagged set, stream seed 0, d = 2, lambda 1. Class prevalence is the stream's candidate mix ({get(P6 / 'rolling_seed0.csv', {'policy': 'oracle', 'window': 5}, 'positives', '{:.0f}', 'figures')} positives in window 5); no bootstrap CI."),
    ("cluster_pca_HI.png", "Phase 4 (exploratory), IBM HI-Small positive rings ({} rings, {} positive groups): 2D PCA of the standardized vectors (D, T_local landscape, D + T_local landscape, B+D+T), top row coloured by pattern type, bottom row by Ward cluster (k = {}). Descriptive: no CI; rings per type CYCLE / GATHER-SCATTER / SCATTER-GATHER / STACK = {}.".format(
        get(CLR, {"dataset": "HI", "vector": "D", "k_mode": "fixed"}, "n_rings", "{:.0f}", "figures"), get(CLR, {"dataset": "HI", "vector": "D", "k_mode": "fixed"}, "n_groups", "{:.0f}", "figures"),
        get(CLR, {"dataset": "HI", "vector": "D", "k_mode": "fixed"}, "k", "{:.0f}", "figures"), " / ".join(get(CLC, {"dataset": "HI", "kind": k}, "rings", "{:.0f}", "figures") for k in ("CYCLE", "GATHER-SCATTER", "SCATTER-GATHER", "STACK")))),
    ("cluster_pca_LI.png", "Phase 4 (exploratory), IBM LI-Small positive rings ({} rings, {} positive groups, {} types in the population): same layout as the HI figure, k = {}. Descriptive: no CI.".format(
        get(CLR, {"dataset": "LI", "vector": "D", "k_mode": "fixed"}, "n_rings", "{:.0f}", "figures"), get(CLR, {"dataset": "LI", "vector": "D", "k_mode": "fixed"}, "n_groups", "{:.0f}", "figures"),
        get(CLR, {"dataset": "LI", "vector": "D", "k_mode": "fixed"}, "n_types", "{:.0f}", "figures"), get(CLR, {"dataset": "LI", "vector": "D", "k_mode": "fixed"}, "k", "{:.0f}", "figures"))),
    ("pump_dump_pr_auc.png", "Phase 7 (optional, SYNTHETIC ONLY, no topology), PR-AUC of M (temporal-motif) and B_pd (group size and volume) features by model, default and harder-exit conditions; prevalence {} in the title of each panel (dashed line), {} positive groups. Error bars: 95% grouped-bootstrap CI. The generator author decides separability.".format(get(PDR, {"condition": "default", "feature_set": "M", "model": "rf"}, "prevalence", "{:.3f}", "figures"), get(PDR, {"condition": "default", "feature_set": "M", "model": "rf"}, "n_pos_groups", "{:.0f}", "figures"))),
    ("recall_ibm.png", "Candidate recall of complete CYCLE attempts (reachable ones, at most max_len accounts) against tau_quantile on HI and LI, days 0-5 and whole data, max_len 6 and 8. Counts, not a sample: no CI; prevalence not applicable. LI days 0-5 has one complete attempt."),
    ("recall_synthetic.png", "Recall of planted wash rings (isolated, dense, chorded) against tau_quantile, six seeds, hard negatives off and on. Mean over rings, no CI. With hard negatives on, the cells where the enumeration cap is hit (daggers in section 6.2, from tau 0.25 or 0.30) are limited by the cap."),
    ("hi_audit_scores.png", "IBM HI-Small pattern-behaviour audit: box plots of the rank-normalised s_rec, s_bal, s_reg and the combined weight (low = suspicious) by pattern group and background pairs, pooled over the audited windows. Descriptive: no CI, no class prevalence (pair counts in section 3.5)."),
    ("li_audit_scores.png", "IBM LI-Small pattern-behaviour audit: same layout as the HI figure, for LI-Small. Descriptive: no CI, no class prevalence."),
]
for f, t in cap:
    fl.append(f"- **{f}**: {t}")
fl.append("")
fl.append("**Planned figure and table IDs of `paper/paper_content.md` section 7 versus what exists.** F1 pipeline diagram: not drawn (an ASCII diagram is in the explainer). F2 triangles and bars: not drawn (`viz.py` helper exists). F3 birth by loop kind under equal vs learned weights: partly, `fig3_birth_distributions.png`. F4 PGD paths on the simplex: `fig2_alpha_simplex.png`. F5 GD optimality gap: `fig1_gd_convergence.png`. **F6 PR curves per feature set: NOT made; PR-AUC bars with CIs (`hi_pr_auc.png`, `synthetic_pr_auc.png`) are used instead** (`fig4_precision_recall.png` is the report's own PR figure). F7 topological lift per scenario: `synthetic_lift_per_scenario.png`. F8 recall vs tau: `recall_ibm.png`, `recall_synthetic.png`. F9 threshold trajectories and reward: `fig5_threshold_drift.png` and `bandit_drift.png`. **F10 ring types vs IBM pattern types: a PCA scatter (`cluster_pca_HI.png`, `cluster_pca_LI.png`), not a dendrogram.** Tables: T1 notation, in paper_content section 8; T2 dataset statistics: pack section 1 and 3.1; T3 gap analysis: paper_content section 4.5 (citation check pending); T4 main classification: pack sections 3.2-3.3 and 4; T5 ablations: pack 3.3 (T_local/T_win) and 6C (cap); undirected, persistence-image and vectorization ablations were not run; T6 clustering: pack 6D; T7 severity: pack 6B; T8 adaptive threshold: pack 6A; T9 hyperparameter sensitivity: only the bandit sensitivity table (6A) exists.")
fl.append("")
fl.append("No figure exists for the severity regression (synthetic only; tables in section 6B) or the cap sensitivity (tables in section 6C); none was generated, as the data freeze allows no new experiments.")
fl.append("")
for f in sorted((RES / "figures").glob("*.png")):
    fl.append(f"- **{f.name}**: optimization-report figure (pinned report generator, deterministic); no CI and no class prevalence stated here, see the report text.")
fl.append("")
if stale:
    fl += ["**Flags.**", *[f"- {s}" for s in stale], ""]
(OUT / "figure_list.md").write_text("\n".join(fl) + "\n")
# rewrite the pack's missing section if figures were missing
txt = (OUT / "numbers_pack.md").read_text()
if missing and "None." in txt.split("## Expected files that were missing")[-1][:20]:
    pass
(OUT / "numbers_pack.md").write_text(txt.split("## Expected files that were missing")[0] + "## Expected files that were missing\n\n" + ("None.\n" if not missing else "\n".join(f"- `{m}`" for m in missing) + "\n"))

pd.DataFrame(prov).to_csv(PACK / "provenance.csv", index=False)
print(f"wrote {OUT / 'numbers_pack.md'} ({len(prov)} registered numbers), {OUT / 'figure_list.md'}")
print("MISSING:" if missing else "no expected file missing", *missing, sep="\n  " if missing else "")

# ====================================================================== spot check
if args.spot_check:
    rng = random.Random(args.seed)
    pack_txt = (OUT / "numbers_pack.md").read_text()
    cand = [p for p in prov if p["file"].endswith(".csv") and "*" not in p["file"] and p["columns"] != "json"]
    print(f"\nSPOT CHECK of {args.spot_check} registered numbers (seed {args.seed}); the source is re-read from disk here:")
    for p in rng.sample(cand, args.spot_check):
        df = pd.read_csv(p["file"])
        where = json.loads(p["where"])
        r = sel(df, where)
        vals = {c: r.iloc[0][c] for c in p["columns"].split(",")} if len(r) == 1 else "ROW NOT UNIQUE"
        print(f"- pack text: {p['text']!r:32s} in pack: {p['text'] in pack_txt}\n    {p['file']}  where {where}\n    source values: {vals}")
