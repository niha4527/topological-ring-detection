"""Phase 6: bandit threshold policies in the drift benchmark (proposal v3 §4.5). Report only; no pass/fail gate.

    nice -n 10 python experiments/bandit_report.py

Stream = the optimization report's drift stream (experiments/gd_report.py, Problem C: ring birth under a detector
learned pre-drift, wash timing jitter 0.05 -> 0.60 and amount noise 0.01 -> 0.16 over 40 windows, ambiguous
candidates counted as negatives, report generator pins); stream seed 0 is exactly the report's stream, seeds 1-5
use window seeds offset by 1000*k. Reward R = TP - lam*FP, labels late by d windows, comparators from
`ascent.run_policies` (fixed, re-tuned every M windows with the same lag, online gradient ascent, hindsight oracle),
plus the bandits of `bandit_sim` on K fixed thresholds. Configuration: configs/bandit.yaml (fixed before running).
Outputs: results/phase6_bandit/ (never results/report/).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ringtda.data.synthetic import SyntheticConfig, generate
from ringtda.features import window_candidates
from ringtda.models.gd import AlphaLearner
from ringtda.threshold.ascent import reward, run_policies
from ringtda.threshold.bandit_sim import DiscountedUCB, SlidingWindowTS, arm_grid, reward_matrix, run_bandit

ap_ = argparse.ArgumentParser()
ap_.add_argument("--config", default="configs/bandit.yaml")
ap_.add_argument("--out", default="results/phase6_bandit")
ap_.add_argument("--seeds", type=int, default=None)
args = ap_.parse_args()
cfg = yaml.safe_load(open(args.config))
OUT = Path(args.out)
(OUT / "cache").mkdir(parents=True, exist_ok=True)
GEN = dict(dense_rings_per_size=0, n_chorded_rings=0, n_shortcut_filled=0, n_hot_wallets=0)  # the report's generator pin
t_start = time.time()
S = cfg["stream"]
N_SEEDS = args.seeds or S["n_seeds"]


def build_stream(k: int):
    """(windows, n_candidates, positives) of stream seed k; windows = [(scores, labels)]."""
    f = OUT / "cache" / f"stream_seed{k}.pkl"
    if f.exists():
        return pd.read_pickle(f)
    off = S["seed_offset"] * k
    Zs, ys = [], []
    for sd in S["pre_seeds"]:
        j = S["pre_jitter"]
        tx, planted = generate(SyntheticConfig(seed=sd + off, ring_jitter=j, amount_noise=0.01 + 0.25 * j, **GEN))
        df, Z = window_candidates(tx, planted, topo=False)
        df = df.assign(y=df.y.fillna(0.0))
        Zs += Z
        ys.append(df.y.to_numpy(float))
    alpha = AlphaLearner(Zs, np.concatenate(ys)).fit(eta=0.5, iters=1500).alpha
    windows = []
    for t in range(S["n_windows"]):
        frac = t / (S["n_windows"] - 1)
        jit = S["jitter"][0] + (S["jitter"][1] - S["jitter"][0]) * frac
        amn = S["amount_noise"][0] + (S["amount_noise"][1] - S["amount_noise"][0]) * frac
        tx, planted = generate(SyntheticConfig(seed=S["base_seed"] + t + off, ring_jitter=jit, amount_noise=amn, **GEN))
        df, Z = window_candidates(tx, planted, topo=False)
        df = df.assign(y=df.y.fillna(0.0))
        al = AlphaLearner(Z, df.y.to_numpy(float))
        windows.append((al.births(alpha, smooth=False), df.y.to_numpy(float)))
    out = {"windows": windows, "alpha": alpha}
    pd.to_pickle(out, f)
    return out


def tp_fp(s, y, tau):
    flag = s <= tau
    return float(y[flag].sum()), float((1 - y[flag]).sum())


taus = arm_grid(cfg["arms"]["K"], cfg["arms"]["lo"], cfg["arms"]["hi"])
W0, SC = cfg["warmup"], cfg["reward"]["scale"]
rows, roll = [], []
for k in range(N_SEEDS):
    st = build_stream(k)
    wins = st["windows"]
    print(f"[{time.time() - t_start:5.0f}s] seed {k}: {len(wins)} windows, alpha {np.round(st['alpha'], 3).tolist()}, positives per window mean {np.mean([w[1].sum() for w in wins]):.1f}", flush=True)
    for lam in cfg["reward"]["lambdas"]:
        R = reward_matrix(wins, taus, lam)
        for d in cfg["lags"]:
            pol = {}
            for M in cfg["retune_every"]:
                o = run_policies(wins, lam=lam, delay=d, eta=cfg["ascent"]["eta"], T=cfg["ascent"]["T"], retune_every=M, warmup=W0)
                pol[f"retune_M{M}"] = o["retune"]
                if M == cfg["primary_retune_every"]:
                    pol.update({"fixed": o["fixed"], "ascent": o["ascent"], "oracle": o["oracle"]})
            pol["oracle_arms"] = {"tau": [taus[int(np.argmax(R[t]))] for t in range(len(wins))], "reward": [float(R[t].max()) for t in range(len(wins))]}
            for name, mk in (("ducb", lambda: DiscountedUCB(len(taus), cfg["ducb"]["gamma"], cfg["ducb"]["c"])),
                             ("swts", lambda: SlidingWindowTS(len(taus), cfg["sw_ts"]["window"], cfg["policy_seed"] + 100 * k + 10 * int(d)))):  # fmt: skip
                b = run_bandit(R, mk(), d, W0, SC)
                pol[name] = {"tau": [np.nan] * W0 + [taus[a] for a in b["arm"]], "reward": [np.nan] * W0 + b["reward"].tolist()}
            orc = float(np.sum(pol["oracle"]["reward"][W0:]))
            for name, v in pol.items():
                cum = float(np.sum(v["reward"][W0:]))
                rows.append({"seed": k, "lambda": lam, "lag": d, "policy": name, "cum_reward": cum, "oracle_cum_reward": orc, "share_of_oracle": cum / orc if orc else np.nan})
                if k == 0 and lam == cfg["reward"]["primary_lambda"] and d == cfg["primary_lag"]:
                    for t in range(W0, len(wins)):
                        tp, fp = tp_fp(*wins[t], v["tau"][t])
                        roll.append({"policy": name, "window": t, "tau": v["tau"][t], "tp": tp, "fp": fp, "positives": float(wins[t][1].sum()), "reward": v["reward"][t]})
    # sensitivity (primary lag and lambda): discount and window, one small table
    R = reward_matrix(wins, taus, cfg["reward"]["primary_lambda"])
    o = run_policies(wins, lam=cfg["reward"]["primary_lambda"], delay=cfg["primary_lag"], eta=cfg["ascent"]["eta"], T=cfg["ascent"]["T"], retune_every=cfg["primary_retune_every"], warmup=W0)
    orc = float(np.sum(o["oracle"]["reward"][W0:]))
    for g in cfg["sensitivity"]["gamma"]:
        b = run_bandit(R, DiscountedUCB(len(taus), g, cfg["ducb"]["c"]), cfg["primary_lag"], W0, SC)
        rows.append({"seed": k, "lambda": cfg["reward"]["primary_lambda"], "lag": cfg["primary_lag"], "policy": f"sens_ducb_gamma{g}", "cum_reward": float(b["reward"].sum()), "oracle_cum_reward": orc, "share_of_oracle": float(b["reward"].sum()) / orc})
    for w in cfg["sensitivity"]["window"]:
        b = run_bandit(R, SlidingWindowTS(len(taus), w, cfg["policy_seed"] + 100 * k + 10 * cfg["primary_lag"]), cfg["primary_lag"], W0, SC)
        rows.append({"seed": k, "lambda": cfg["reward"]["primary_lambda"], "lag": cfg["primary_lag"], "policy": f"sens_swts_window{w}", "cum_reward": float(b["reward"].sum()), "oracle_cum_reward": orc, "share_of_oracle": float(b["reward"].sum()) / orc})

R_ = pd.DataFrame(rows)
R_.to_csv(OUT / "results.csv", index=False)
pd.DataFrame(roll).to_csv(OUT / "rolling_seed0.csv", index=False)
agg = R_.groupby(["lambda", "lag", "policy"]).agg(cum_mean=("cum_reward", "mean"), cum_std=("cum_reward", "std"), share_mean=("share_of_oracle", "mean"), share_std=("share_of_oracle", "std"), n_seeds=("seed", "nunique")).reset_index()
agg.to_csv(OUT / "summary.csv", index=False)
pairs = []
for (lam, d), g in R_.groupby(["lambda", "lag"]):
    wide = g.pivot(index="seed", columns="policy", values="share_of_oracle")
    for b in ("ducb", "swts"):
        for ref in (f"retune_M{cfg['primary_retune_every']}", "ascent", "fixed"):
            diff = wide[b] - wide[ref]
            pairs.append({"lambda": lam, "lag": d, "bandit": b, "comparator": ref, "diff_share_mean": diff.mean(), "diff_share_std": diff.std(), "seeds_bandit_better": int((diff > 0).sum()), "n_seeds": len(diff)})
P_ = pd.DataFrame(pairs)
P_.to_csv(OUT / "bandit_vs_comparators.csv", index=False)
yaml.safe_dump({"config": cfg, "n_seeds": N_SEEDS, "runtime_seconds": time.time() - t_start}, open(OUT / "config_used.yaml", "w"))

# ---- numbers-only markdown
f = lambda m, s: f"{m:.3f} +- {s:.3f}"  # noqa: E731
md = [f"# Phase 6: bandit threshold policies, drift benchmark\n",
      f"{N_SEEDS} stream seeds (seed 0 = the report's stream), 40 windows, reward counted from window {W0}. Share of oracle = cumulative reward / hindsight-oracle cumulative reward (oracle over a 401-point grid, per window). "
      f"`oracle_arms` = best of the K={cfg['arms']['K']} arms per window. Mean +- std over seeds. Bandits see only the played arm's reward (bandit feedback), comparators see whole labelled windows. "
      "Lag d: labels of window s usable at t >= s + d (run_policies convention); bandits cannot use a reward before the arm is played, so their effective lag is max(d, 1); "
      "at d = 0 run_policies' re-tune and ascent use the window's own labels (not like-for-like, flagged).\n"]  # fmt: skip
ORD = ["fixed", f"retune_M{cfg['retune_every'][0]}", f"retune_M{cfg['primary_retune_every']}", f"retune_M{cfg['retune_every'][-1]}", "ascent", "ducb", "swts", "oracle_arms", "oracle"]
for lam in cfg["reward"]["lambdas"]:
    md.append(f"\n## lambda = {lam}\n")
    for kind, mc, sc_ in (("share of oracle", "share_mean", "share_std"), ("cumulative reward", "cum_mean", "cum_std")):
        t = agg[agg["lambda"] == lam]
        header = "| policy | " + " | ".join(f"d={d}" for d in cfg["lags"]) + " |\n|---|" + "---|" * len(cfg["lags"])
        md.append(f"\n{kind}:\n\n{header}")
        for p in ORD:
            md.append(f"| {p} | " + " | ".join(f(*t[(t.policy == p) & (t.lag == d)][[mc, sc_]].iloc[0]) for d in cfg["lags"]) + " |")
md.append("\n## Bandit minus comparator, share of oracle (paired per seed)\n\n| lambda | d | bandit | comparator | mean diff +- std | seeds with bandit better |\n|---|---|---|---|---|---|")
md += [f"| {r.lambda_} | {r.lag} | {r.bandit} | {r.comparator} | {r.diff_share_mean:+.3f} +- {r.diff_share_std:.3f} | {r.seeds_bandit_better}/{r.n_seeds} |" for r in P_.rename(columns={"lambda": "lambda_"}).itertuples()]
sens = agg[agg.policy.str.startswith("sens_")]
md.append("\n## Sensitivity (lambda " + str(cfg["reward"]["primary_lambda"]) + ", d " + str(cfg["primary_lag"]) + ")\n\n| setting | share of oracle |\n|---|---|")
md += [f"| {r.policy} | {f(r.share_mean, r.share_std)} |" for r in sens[(sens["lambda"] == cfg["reward"]["primary_lambda"]) & (sens.lag == cfg["primary_lag"])].itertuples()]
(OUT / "summary.md").write_text("\n".join(md) + "\n")
print(f"done in {time.time() - t_start:.0f}s -> {OUT}", flush=True)
