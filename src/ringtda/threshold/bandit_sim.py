"""Bandit threshold policies for the drift benchmark (proposal v3 §4.5; report §2.7 is the gradient-ascent baseline).

Arms are K fixed thresholds on the classifier score (a candidate is flagged when score <= tau). Per window the arm
played earns R = TP - lam * FP (`ascent.reward`); only the PLAYED arm's reward is observed (bandit feedback), and
it becomes usable `max(d, 1)` windows after the window it was earned in (a reward cannot be known before the arm is
played, so d = 0 behaves as d = 1; `ascent.run_policies` uses the label convention "labels of window s usable at
t >= s + d"). Rewards are normalised to [0, 1] for the policies.

DiscountedUCB: observation of play s weighs gamma^(t - s) at time t; index = mean + c * sqrt(ln(1 + n) / N).
SlidingWindowTS: Beta(1 + successes, 1 + failures) per arm over the plays of the last `window` windows; the
normalised reward u in [0, 1] becomes a Bernoulli(u) trial when it is revealed (seeded); one posterior sample per
arm, argmax. Both use the full-information warm start of the first `warmup` windows (all arms observed there),
the same windows that tune the fixed comparator.
"""

from __future__ import annotations

import math

import numpy as np

from ringtda.threshold.ascent import reward


def arm_grid(K: int, lo: float, hi: float) -> np.ndarray:
    return np.linspace(lo, hi, K)


def reward_matrix(windows, taus, lam: float) -> np.ndarray:
    """R[t, a] = TP - lam*FP of window t when arm a is played."""
    return np.array([[reward(tau, s, y, lam) for tau in taus] for s, y in windows])


def normalise(R, scale: float):
    return (np.clip(np.asarray(R, dtype=float) / scale, -1.0, 1.0) + 1.0) / 2.0


class DiscountedUCB:
    def __init__(self, K: int, gamma: float = 0.9, c: float = 0.5):
        self.K, self.gamma, self.c = K, gamma, c
        self.obs: list[tuple[int, int, float]] = []  # (play time s, arm, normalised reward)

    def update(self, s: int, arm: int, u: float) -> None:
        self.obs.append((s, arm, u))

    def choose(self, t: int, pending=()) -> int:
        N, S = np.zeros(self.K), np.zeros(self.K)
        for s, a, u in self.obs:
            w = self.gamma ** (t - s)
            N[a] += w
            S[a] += w * u
        tried = N > 1e-12
        tried[list(pending)] = True  # an arm whose play is still unrevealed counts as tried
        if not tried.all():
            return int(np.flatnonzero(~tried)[0])
        n = N.sum()
        idx = np.where(N > 1e-12, S / np.maximum(N, 1e-12) + self.c * np.sqrt(math.log(1 + n) / np.maximum(N, 1e-12)), np.inf)
        return int(np.argmax(idx))


class SlidingWindowTS:
    def __init__(self, K: int, window: int = 10, seed: int = 0):
        self.K, self.window = K, window
        self.rng = np.random.default_rng(seed)
        self.obs: list[tuple[int, int, int]] = []  # (play time s, arm, bernoulli outcome)

    def update(self, s: int, arm: int, u: float) -> None:
        self.obs.append((s, arm, int(self.rng.random() < u)))

    def choose(self, t: int, pending=()) -> int:
        a_ = np.ones(self.K)
        b_ = np.ones(self.K)
        for s, a, b in self.obs:
            if s > t - self.window:  # only the last `window` windows count
                a_[a] += b
                b_[a] += 1 - b
        return int(np.argmax(self.rng.beta(a_, b_)))


def run_bandit(R: np.ndarray, policy, d: int, warmup: int, scale: float) -> dict:
    """Play arms for windows warmup..n-1. R[t, a]: raw reward of arm a in window t (only the played arm is observed).

    Returns arms and raw rewards per window t >= warmup, and `used_at`: for each reward the time it first became usable
    (always >= s + max(d, 1), asserted)."""
    n, K = R.shape
    lag = max(d, 1)
    U = normalise(R, scale)
    for s in range(warmup):  # full-information warm start
        for a in range(K):
            policy.update(s, a, U[s, a])
    arms, rew, pending = [], [], []  # pending: (play time s, arm)
    used_at = []
    for t in range(warmup, n):
        ready = [p for p in pending if p[0] + lag <= t]
        for s, a in ready:
            assert s + lag <= t, "reward used before it was revealed"
            policy.update(s, a, U[s, a])
            used_at.append((s, t))
        pending = [p for p in pending if p not in ready]
        a = policy.choose(t, pending=[p[1] for p in pending])
        arms.append(a)
        rew.append(float(R[t, a]))
        pending.append((t, a))
    return {"arm": np.array(arms), "reward": np.array(rew), "used_at": used_at}
