"""Experiments for the optimization report (report §4). Run from the project root:

    python experiments/gd_report.py --config report            # the numbers in the report
    python experiments/gd_report.py --config full [--ambiguous exclude|negative]

Configs
    report  The setup the report was written for: the generator without the dense,
            chorded, shortcut-filled and hot-wallet scenarios, and ambiguous candidates
            counted as negatives (report §2.4: y = 1 only for a planted wash ring).
            Writes results/report/ and refuses to overwrite it unless --force.
    full    Every generator scenario. Writes results/full/<ambiguous>/.

Candidates are labelled by the §3.7 ring-labeling rule (ringtda.eval.labels).
Takes a few seconds (report) to a few minutes (full; the drift study generates 40 windows).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, precision_recall_curve

from ringtda.data.synthetic import SyntheticConfig, generate
from ringtda.features import window_candidates
from ringtda.models.gd import (
    AlphaLearner,
    class_weights,
    gradient_descent,
    logistic_loss_grad,
    smoothness_constant,
)
from ringtda.threshold.ascent import run_policies

ap_ = argparse.ArgumentParser()
ap_.add_argument("--config", choices=("report", "full"), required=True)
ap_.add_argument("--ambiguous", choices=("exclude", "negative"), help="full config only (default exclude)")
ap_.add_argument("--force", action="store_true", help="allow overwriting results/report/")
args = ap_.parse_args()
if args.config == "report":
    if args.ambiguous not in (None, "negative"):
        ap_.error("--config report fixes --ambiguous negative")
    AMBIGUOUS = "negative"
    # pin the generator the report was written for (scenarios added later are switched off)
    GEN = dict(dense_rings_per_size=0, n_chorded_rings=0, n_shortcut_filled=0, n_hot_wallets=0)
    OUT = Path("results") / "report"
    if (OUT / "gd_metrics.json").exists() and not args.force:
        ap_.error(f"{OUT} holds the report's numbers; pass --force to overwrite")
else:
    AMBIGUOUS = args.ambiguous or "exclude"
    GEN = {}
    OUT = Path("results") / "full" / AMBIGUOUS
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
GREY, BLUE, RED, GREEN, ORANGE = "#8c959f", "#1f6feb", "#cf222e", "#1a7f37", "#bc4c00"

D_FEATS = ["birth_eq", "length", "mean_rec", "mean_bal", "mean_reg", "min_rec", "min_bal", "min_reg"]
T_FEATS = ["topo_h1_count", "topo_total_pers", "topo_max_pers", "topo_n_infinite", "topo_nbhd_edges"]
TRAIN_SEEDS, TEST_SEEDS = range(0, 12), range(100, 106)
metrics: dict = {"config": args.config, "ambiguous_treatment": AMBIGUOUS, "generator_overrides": GEN}


def treat(df, Z):
    """Ambiguous candidates (§3.7 labelling rule): dropped, or kept as negatives."""
    if AMBIGUOUS == "negative":
        return df.assign(y=df.y.fillna(0.0)), Z
    keep = (df.kind != "ambiguous").to_numpy()
    return df[keep].reset_index(drop=True), [z for z, k in zip(Z, keep) if k]


def build(seeds, jitters, topo=True, n_ambiguous=None):
    """One window per seed; wash-ring timing jitter (and amount noise) varies across windows,
    so that no single score separates wash rings from legitimate loops on its own."""
    tabs, Zs = [], []
    for s, j in zip(seeds, jitters):
        tx, planted = generate(SyntheticConfig(seed=s, ring_jitter=j, amount_noise=0.01 + 0.25 * j, **GEN))
        df, Z = window_candidates(tx, planted, topo=topo)
        if n_ambiguous is not None:
            n_ambiguous.append(int((df.kind == "ambiguous").sum()))
        df, Z = treat(df, Z)
        df["window"] = s
        tabs.append(df)
        Zs += Z
    return pd.concat(tabs, ignore_index=True), Zs


def design(df, cols, mu=None, sd=None):
    X = df[cols].to_numpy(float)
    if "topo_nbhd_edges" in cols:
        j = cols.index("topo_nbhd_edges")
        X[:, j] = np.log1p(X[:, j])
    mu = X.mean(0) if mu is None else mu
    sd = X.std(0) + 1e-9 if sd is None else sd
    return np.column_stack([np.ones(len(X)), (X - mu) / sd]), mu, sd


t0 = time.time()
TRAIN_JIT = np.linspace(0.05, 0.6, len(TRAIN_SEEDS))
TEST_JIT = np.linspace(0.1, 0.55, len(TEST_SEEDS))
amb_train, amb_test = [], []
train, Ztr = build(TRAIN_SEEDS, TRAIN_JIT, n_ambiguous=amb_train)
test, Zte = build(TEST_SEEDS, TEST_JIT, n_ambiguous=amb_test)
ytr, yte = train.y.to_numpy(float), test.y.to_numpy(float)
metrics["data"] = {
    "train_jitter": TRAIN_JIT.round(3).tolist(),
    "test_jitter": TEST_JIT.round(3).tolist(),
    "train_windows": len(TRAIN_SEEDS),
    "test_windows": len(TEST_SEEDS),
    "train_candidates": int(len(train)),
    "test_candidates": int(len(test)),
    "train_by_kind": train.kind.value_counts().to_dict(),
    "test_by_kind": test.kind.value_counts().to_dict(),
    "ambiguous_per_window": {"train": amb_train, "test": amb_test},
}
print(f"data built in {time.time() - t0:.0f}s", metrics["data"])

# ------------------------------------------------------------------ Problem A
lam = 0.01
cols = D_FEATS + T_FEATS
Xtr, mu, sd = design(train, cols)
Xte, _, _ = design(test, cols, mu, sd)
wtr = class_weights(ytr)
L = smoothness_constant(Xtr, wtr, lam)
fun = lambda th: logistic_loss_grad(th, Xtr, ytr, wtr, lam)
ref = minimize(lambda th: fun(th)[0], np.zeros(Xtr.shape[1]), jac=lambda th: fun(th)[1], method="L-BFGS-B", tol=1e-12)
Jstar = ref.fun

# local curvature at the optimum: largest eigenvalue of the Hessian at theta*
p_star = 1 / (1 + np.exp(-Xtr @ ref.x))
Hs = (Xtr * (wtr * p_star * (1 - p_star))[:, None]).T @ Xtr / len(ytr) + lam * np.diag([0.0] + [1.0] * (Xtr.shape[1] - 1))
L_loc = float(np.linalg.eigvalsh(Hs).max())

runs = {}
for label, eta in [("0.2/L", 0.2 / L), ("1/L", 1.0 / L), ("1.9/L", 1.9 / L), ("1/L_loc", 1.0 / L_loc), ("2.2/L_loc", 2.2 / L_loc)]:
    runs[label] = gradient_descent(Xtr, ytr, wtr, lam, eta=eta, iters=3000)
gd = runs["1/L_loc"]
metrics["problem_A"] = {
    "lambda": lam,
    "L": L,
    "eta_1_over_L": 1 / L,
    "L_local_at_optimum": L_loc,
    "J_star_lbfgs": Jstar,
    "final_gap": {k: float(r.loss[-1] - Jstar) for k, r in runs.items()},
    "iters_to_1e-6": {
        k: (int(np.argmax(np.array(r.loss) - Jstar < 1e-6)) if np.any(np.array(r.loss) - Jstar < 1e-6) else None)
        for k, r in runs.items()
    },
    "max_abs_theta_diff_vs_lbfgs": float(np.abs(gd.theta - ref.x).max()),
}

fig, ax = plt.subplots(figsize=(7, 4))
styles = {"0.2/L": (GREY, "-"), "1/L": (BLUE, "-"), "1.9/L": (GREEN, "-"), "1/L_loc": (ORANGE, "--"), "2.2/L_loc": (RED, ":")}
for k, r in runs.items():
    gap = np.maximum(np.array(r.loss) - Jstar, 1e-16)
    ax.semilogy(gap, color=styles[k][0], ls=styles[k][1], lw=1.6, label=f"η = {k.replace('L_loc', 'L_local')}")
ax.set_xlabel("iteration t")
ax.set_ylabel("J(θᵗ) − J*  (log scale)")
ax.set_title(f"Problem A: effect of step size (L = {L:.2f}, L_local = {L_loc:.3f})")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG / "fig1_gd_convergence.png")
plt.close(fig)

# test AP: D only vs D+T, both trained by GD with eta = 1/L
ap = {}
pr_curves = {}
for name, cset in [("D (behavioural)", D_FEATS), ("D + T (behavioural + topological)", cols)]:
    Xa, m_, s_ = design(train, cset)
    Xb, _, _ = design(test, cset, m_, s_)
    La = smoothness_constant(Xa, wtr, lam)
    th = gradient_descent(Xa, ytr, wtr, lam, eta=1 / La, iters=4000).theta
    sc = Xb @ th
    ap[name] = float(average_precision_score(yte, sc))
    pr_curves[name] = precision_recall_curve(yte, sc)
coef = pd.Series(gd.theta[1:], index=cols).sort_values(key=np.abs, ascending=False)
metrics["problem_A"]["test_AP"] = ap
metrics["problem_A"]["coefficients_D+T"] = coef.round(3).to_dict()

# ------------------------------------------------------------------ Problem B
learner = AlphaLearner(Ztr, ytr, beta=30.0)
eq = np.full(3, 1 / 3)
fitB = learner.fit(alpha0=eq, eta=0.5, iters=1500)
starts = {"equal": eq, "rec vertex": np.array([1.0, 0, 0]), "bal vertex": np.array([0, 1.0, 0]), "reg vertex": np.array([0, 0, 1.0])}
restarts = {k: learner.fit(alpha0=a, eta=0.5, iters=1500) for k, a in starts.items()}

# finite-difference check of the alpha gradient (validates the derivation in §2.6)
rng = np.random.default_rng(0)
a0, c0, k0 = np.array([0.5, 0.3, 0.2]), 0.3, 4.0
_, ga, gc, gk = learner.loss_grad(a0, c0, k0)
num = []
for j in range(3):
    e = np.zeros(3)
    e[j] = 1e-6
    num.append((learner.loss_grad(a0 + e, c0, k0)[0] - learner.loss_grad(a0 - e, c0, k0)[0]) / 2e-6)
fd_err = float(np.abs(np.array(num) - ga).max())

_, ga_star, _, _ = learner.loss_grad(fitB.alpha, fitB.c, fitB.k)
lte = AlphaLearner(Zte, yte)
b_eq, b_learn = lte.births(eq, smooth=False), lte.births(fitB.alpha, smooth=False)
metrics["problem_B"] = {
    "alpha_learned": fitB.alpha.round(4).tolist(),
    "c": fitB.c,
    "k": fitB.k,
    "loss_start": fitB.loss[0],
    "loss_end": fitB.loss[-1],
    "restarts": {k: {"alpha": r.alpha.round(4).tolist(), "loss": r.loss[-1]} for k, r in restarts.items()},
    "grad_check_max_abs_error": fd_err,
    "grad_alpha_at_solution": ga_star.tolist(),
    "test_AP_ring_birth_equal_alpha": float(average_precision_score(yte, -b_eq)),
    "test_AP_ring_birth_learned_alpha": float(average_precision_score(yte, -b_learn)),
}
for name, b in [("equal α", b_eq), ("learned α", b_learn)]:
    pr_curves[f"ring birth, {name}"] = precision_recall_curve(yte, -b)

# figure: alpha trajectories on the simplex + loss curves
def tern(a):
    a = np.atleast_2d(a)
    return a[:, 1] + 0.5 * a[:, 2], (np.sqrt(3) / 2) * a[:, 2]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
corners = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
cx, cy = tern(corners)
ax1.plot(cx, cy, color=GREY, lw=1)
for (x, y), lab, ha in zip(zip(cx[:3], cy[:3]), ["α_rec = 1", "α_bal = 1", "α_reg = 1"], ["right", "left", "center"]):
    ax1.text(x + (-0.02 if ha == "right" else 0.02 if ha == "left" else 0), y + (0.03 if ha == "center" else -0.05), lab, ha=ha, fontsize=9)
cols_r = {"equal": BLUE, "rec vertex": GREY, "bal vertex": GREY, "reg vertex": GREY}
for k, r in restarts.items():
    px, py = tern(np.array(r.path))
    ax1.plot(px, py, color=cols_r[k], lw=1.8 if k == "equal" else 1.0)
    ax1.scatter(px[0], py[0], color=cols_r[k], s=18, marker="o")
fx, fy = tern(fitB.alpha)
ax1.scatter(fx, fy, color=BLUE, s=70, marker="*", zorder=5, label=f"learned α = ({fitB.alpha[0]:.2f}, {fitB.alpha[1]:.2f}, {fitB.alpha[2]:.2f})")
ex, ey = tern(eq)
ax1.scatter(ex, ey, color="k", s=25, zorder=5, label="start: equal weights")
ax1.set_aspect("equal")
ax1.axis("off")
ax1.legend(loc="upper left", fontsize=8, frameon=False)
ax1.set_title("Projected gradient descent paths on the simplex")
for k, r in restarts.items():
    ax2.plot(r.loss, color=cols_r[k], lw=1.8 if k == "equal" else 1.0, label=f"start: {k}")
ax2.set_xlabel("iteration t")
ax2.set_ylabel("weighted cross-entropy J")
ax2.set_title("Problem B: loss from four starting points")
ax2.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "fig2_alpha_simplex.png")
plt.close(fig)

# figure: ring-birth distributions by kind, equal vs learned alpha
kinds = [("wash_ring", RED, "wash ring (positive)"), ("market_maker", ORANGE, "market maker"), ("arbitrage_loop", GREEN, "arbitrage loop"), ("background", GREY, "background loop")]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
bins = np.linspace(0, 1, 41)
for ax, b, title in [(axes[0], b_eq, "equal weights α = (1/3, 1/3, 1/3)"), (axes[1], b_learn, f"learned α = ({fitB.alpha[0]:.2f}, {fitB.alpha[1]:.2f}, {fitB.alpha[2]:.2f})")]:
    for kd, c, lab in kinds:
        m = (test.kind == kd).to_numpy()
        ax.hist(b[m], bins=bins, color=c, alpha=0.7, density=True, label=lab)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("ring birth b_R (low = suspicious)")
axes[0].axvline(1 / 3, color="k", ls="--", lw=0.8)
axes[0].text(1 / 3 + 0.01, axes[0].get_ylim()[1] * 0.9, "1/3 floor\n(Prop. 1)", fontsize=8)
axes[0].set_ylabel("density")
axes[1].legend(frameon=False, fontsize=8)
fig.suptitle("Test windows: where each kind of loop lands in the filtration", fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "fig3_birth_distributions.png")
plt.close(fig)

# figure: precision-recall curves on test windows
fig, ax = plt.subplots(figsize=(6.5, 4.4))
style = {
    "ring birth, equal α": (GREY, "--"),
    "ring birth, learned α": (BLUE, "-"),
    "D (behavioural)": (GREEN, "-."),
    "D + T (behavioural + topological)": (RED, ":"),
}
apall = {**{k: v for k, v in ap.items()}, "ring birth, equal α": metrics["problem_B"]["test_AP_ring_birth_equal_alpha"], "ring birth, learned α": metrics["problem_B"]["test_AP_ring_birth_learned_alpha"]}
for k, (p, r, _) in pr_curves.items():
    ax.plot(r, p, color=style[k][0], ls=style[k][1], lw=1.7, label=f"{k}  (AP {apall[k]:.3f})")
ax.set_xlabel("recall")
ax.set_ylabel("precision")
ax.set_ylim(0, 1.03)
ax.set_title("Test windows: precision-recall of four detectors")
ax.legend(frameon=False, fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(FIG / "fig4_precision_recall.png")
plt.close(fig)

# ------------------------------------------------------------------ Problem C
# The deployed detector is learned on pre-drift data only (low timing noise), then tactics drift.
amb_pre = []
pre, Zpre = build(range(300, 304), [0.05] * 4, topo=False, n_ambiguous=amb_pre)
alpha_pre = AlphaLearner(Zpre, pre.y.to_numpy(float)).fit(eta=0.5, iters=1500).alpha
N_WIN = 40
drift_windows, robust_windows, recall_cand, amb_drift = [], [], [], []
for t in range(N_WIN):
    frac = t / (N_WIN - 1)
    tx, planted = generate(SyntheticConfig(seed=500 + t, ring_jitter=0.05 + 0.55 * frac, amount_noise=0.01 + 0.15 * frac, **GEN))
    df, Z = window_candidates(tx, planted, topo=False)
    amb_drift.append(int((df.kind == "ambiguous").sum()))
    df, Z = treat(df, Z)
    al = AlphaLearner(Z, df.y.to_numpy(float))
    drift_windows.append((al.births(alpha_pre, smooth=False), df.y.to_numpy(float)))
    robust_windows.append((al.births(fitB.alpha, smooth=False), df.y.to_numpy(float)))
    recall_cand.append(df.y.sum() / sum(p.pattern == "wash_ring" for p in planted))
pol = run_policies(drift_windows, lam=1.0, delay=2, eta=0.05, T=0.02, retune_every=5, warmup=5)
tot = {k: float(np.sum(v["reward"][5:])) for k, v in pol.items()}
pol_r = run_policies(robust_windows, lam=1.0, delay=2, eta=0.05, T=0.02, retune_every=5, warmup=5)
tot_r = {k: float(np.sum(v["reward"][5:])) for k, v in pol_r.items()}
metrics["problem_C"] = {
    "windows": N_WIN,
    "lambda": 1.0,
    "delay": 2,
    "cumulative_reward_after_warmup": tot,
    "alpha_pre_drift": alpha_pre.round(4).tolist(),
    "share_of_oracle": {k: tot[k] / tot["oracle"] for k in tot},
    "robust_detector_cumulative_reward": tot_r,
    "robust_detector_share_of_oracle": {k: tot_r[k] / tot_r["oracle"] for k in tot_r},
    "drift": "wash timing jitter 0.05 -> 0.60 and amount noise 0.01 -> 0.16 over 40 windows",
    "candidate_recall_first5_last5": [float(np.mean(recall_cand[:5])), float(np.mean(recall_cand[-5:]))],
    "ambiguous_per_window": amb_drift,
    "ambiguous_per_pre_drift_window": amb_pre,
}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
pcol = {"fixed": GREY, "retune": GREEN, "ascent": BLUE, "oracle": "k"}
plab = {"fixed": "fixed (tuned on first 5)", "retune": "re-tuned every 5 windows", "ascent": "online gradient ascent", "oracle": "oracle (hindsight)"}
for k in ("oracle", "fixed", "retune", "ascent"):
    ax1.plot(pol[k]["tau"], color=pcol[k], lw=1.8 if k == "ascent" else 1.1, ls=":" if k == "oracle" else "-", label=plab[k])
    ax2.plot(np.cumsum(pol[k]["reward"]), color=pcol[k], lw=1.8 if k == "ascent" else 1.1, ls=":" if k == "oracle" else "-", label=plab[k])
for ax in (ax1, ax2):
    ax.axvspan(0, 4.5, color=GREY, alpha=0.12)
    ax.set_xlabel("window t (tactic drift increases →)")
ax1.set_ylabel("threshold τ")
ax1.set_title("Threshold chosen by each policy")
ax2.set_ylabel("cumulative reward  Σ (TP − λ·FP)")
ax2.set_title("Problem C: cumulative reward under drift")
ax2.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "fig5_threshold_drift.png")
plt.close(fig)

(OUT / "gd_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
print(json.dumps(metrics, indent=2, default=float))
print(f"done in {time.time() - t0:.0f}s")
