"""Adaptive flagging threshold (report §2.7).

A candidate ring is flagged when its score s (ring birth; low = suspicious) is <= tau.
Reward per window: R(tau) = TP - lam * FP. Online gradient ascent on the smoothed
reward uses labels that arrive `delay` windows late.
"""

from __future__ import annotations

import numpy as np

from ringtda.models.gd import sigmoid


def reward(tau, s, y, lam):
    flag = s <= tau
    return float((y[flag]).sum() - lam * (1 - y[flag]).sum())


def smoothed_reward_grad(tau, s, y, lam, T):
    """d/dtau of sum_i g_i sigmoid((tau - s_i)/T), divided by n (report §2.7)."""
    g = y - lam * (1 - y)
    sg = sigmoid((tau - s) / T)
    return float((g * sg * (1 - sg)).sum() / (T * len(s)))


def best_tau(s, y, lam, grid):
    vals = [reward(t, s, y, lam) for t in grid]
    return float(grid[int(np.argmax(vals))])


def run_policies(windows, lam=1.0, delay=2, eta=0.05, T=0.02, retune_every=5, warmup=5, grid=None):
    """windows: list of (scores, labels) in time order. Returns per-window tau and reward per policy.

    fixed    best tau on the first `warmup` windows, then frozen
    retune   every `retune_every` windows, best tau on the last `warmup` labelled windows (same delay)
    ascent   online gradient ascent on the latest labelled window
    oracle   best tau for each window with its own labels (upper bound, not achievable)
    """
    grid = np.linspace(0, 1, 401) if grid is None else grid
    n = len(windows)
    pooled = lambda idx: (np.concatenate([windows[i][0] for i in idx]), np.concatenate([windows[i][1] for i in idx]))

    s0, y0 = pooled(range(warmup))
    tau_fixed = best_tau(s0, y0, lam, grid)
    out = {k: {"tau": [], "reward": []} for k in ("fixed", "retune", "ascent", "oracle")}
    tau_rt, tau_ga = tau_fixed, tau_fixed
    for t in range(n):
        s, y = windows[t]
        labelled = t - delay  # newest window whose labels are known at time t
        if t >= warmup and (t - warmup) % retune_every == 0 and labelled >= 0:
            lo = max(0, labelled - warmup + 1)
            tau_rt = best_tau(*pooled(range(lo, labelled + 1)), lam, grid)
        if t >= warmup and labelled >= 0:
            sl, yl = windows[labelled]
            for _ in range(20):  # a few ascent steps on the newest labelled window
                tau_ga = float(np.clip(tau_ga + eta * smoothed_reward_grad(tau_ga, sl, yl, lam, T), 0, 1))
        tau_or = best_tau(s, y, lam, grid)
        for name, tau in (("fixed", tau_fixed), ("retune", tau_rt), ("ascent", tau_ga), ("oracle", tau_or)):
            out[name]["tau"].append(tau)
            out[name]["reward"].append(reward(tau, s, y, lam))
    return out
