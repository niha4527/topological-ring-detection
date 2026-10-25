"""Gradient-based learning (report §2.5 and §2.6).

Problem A  weighted, L2-regularised logistic regression by batch gradient descent (convex)
Problem B  learning the score-combination weights alpha on the probability simplex
           by projected gradient descent (constrained, non-convex)

Everything is written out in NumPy so every line maps to an equation in the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def sigmoid(u):
    # numerically stable: never exponentiates a large positive number
    u = np.asarray(u, dtype=float)
    out = np.empty_like(u)
    pos = u >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-u[pos]))
    eu = np.exp(u[~pos])
    out[~pos] = eu / (1.0 + eu)
    return out


def class_weights(y: np.ndarray) -> np.ndarray:
    """omega_R = N / (2 N_{y_R}): each class carries half of the total weight."""
    y = np.asarray(y)
    n, n1 = len(y), y.sum()
    return np.where(y == 1, n / (2.0 * n1), n / (2.0 * (n - n1)))


def _bce(p, y, w):
    eps = 1e-12
    return -np.mean(w * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))


# ---------------------------------------------------------------- Problem A


def logistic_loss_grad(theta, X, y, w, lam):
    """J(theta) and grad J (report §2.5). Column 0 of X is the bias (not penalised)."""
    p = sigmoid(X @ theta)
    reg = theta.copy()
    reg[0] = 0.0
    J = _bce(p, y, w) + 0.5 * lam * reg @ reg
    g = X.T @ (w * (p - y)) / len(y) + lam * reg
    return J, g


def smoothness_constant(X, w, lam) -> float:
    """L = lambda_max(X^T Omega X) / (4N) + lambda  (report §2.5)."""
    M = (X * w[:, None]).T @ X / (4.0 * len(w))
    return float(np.linalg.eigvalsh(M).max() + lam)


@dataclass
class GDResult:
    theta: np.ndarray
    loss: list = field(default_factory=list)
    grad_norm: list = field(default_factory=list)


def gradient_descent(X, y, w, lam, eta, iters=2000, theta0=None) -> GDResult:
    """theta_{t+1} = theta_t - eta * grad J(theta_t)."""
    theta = np.zeros(X.shape[1]) if theta0 is None else theta0.astype(float).copy()
    res = GDResult(theta)
    for _ in range(iters):
        J, g = logistic_loss_grad(theta, X, y, w, lam)
        res.loss.append(J)
        res.grad_norm.append(float(np.linalg.norm(g)))
        if not np.isfinite(J) or J > 1e6:
            break
        theta = theta - eta * g
    res.theta = theta
    return res


# ---------------------------------------------------------------- Problem B


def project_simplex(u) -> np.ndarray:
    """Euclidean projection onto {a >= 0, sum a = 1} via the sort-based KKT solution (report §2.6)."""
    u = np.asarray(u, dtype=float)
    s = np.sort(u)[::-1]
    css = np.cumsum(s)
    j = np.arange(1, len(u) + 1)
    rho = np.nonzero(s - (css - 1) / j > 0)[0][-1] + 1
    nu = (css[rho - 1] - 1) / rho
    return np.maximum(u - nu, 0.0)


@dataclass
class AlphaResult:
    alpha: np.ndarray
    c: float
    k: float
    loss: list = field(default_factory=list)
    path: list = field(default_factory=list)  # alpha after every step


class AlphaLearner:
    """min_{alpha in simplex, c, k} weighted BCE of p_R = sigmoid(c - k * softmax_beta birth_R(alpha)).

    rings: list of arrays Z_R with shape (|R|, 3) = the normalised scores z_e of the ring's edges.
    """

    def __init__(self, rings, y, beta=30.0):
        self.Z = [np.asarray(z, dtype=float) for z in rings]
        self.y = np.asarray(y, dtype=float)
        self.w = class_weights(self.y)
        self.beta = beta
        # stack for vectorised evaluation: pad rings to the same length with -inf weights
        self.lens = np.array([len(z) for z in self.Z])
        m = self.lens.max()
        self.Zpad = np.zeros((len(self.Z), m, 3))
        self.mask = np.zeros((len(self.Z), m), dtype=bool)
        for i, z in enumerate(self.Z):
            self.Zpad[i, : len(z)] = z
            self.mask[i, : len(z)] = True

    def births(self, alpha, smooth=True):
        w = 1.0 - self.Zpad @ alpha  # (n, m) edge weights
        if not smooth:
            return np.where(self.mask, w, -np.inf).max(axis=1)
        bw = np.where(self.mask, self.beta * w, -np.inf)
        mx = bw.max(axis=1, keepdims=True)
        e = np.exp(bw - mx)
        S = e.sum(axis=1, keepdims=True)
        b = (mx[:, 0] + np.log(S[:, 0])) / self.beta
        pi = e / S  # softmax weights pi_{R,e}
        return b, pi

    def loss_grad(self, alpha, c, k):
        b, pi = self.births(alpha)
        p = sigmoid(c - k * b)
        J = _bce(p, self.y, self.w)
        r = self.w * (p - self.y) / len(self.y)  # residuals r_R
        gc = r.sum()
        gk = -(r * b).sum()
        zbar = np.einsum("nm,nmj->nj", pi, self.Zpad)  # sum_e pi_{R,e} z_e
        ga = k * (r[:, None] * zbar).sum(axis=0)
        return J, ga, gc, gk

    def fit(self, alpha0=None, c0=0.0, k0=5.0, eta=0.5, iters=1500) -> AlphaResult:
        alpha = np.full(3, 1 / 3) if alpha0 is None else project_simplex(alpha0)
        c, k = float(c0), float(k0)
        res = AlphaResult(alpha.copy(), c, k)
        for _ in range(iters):
            J, ga, gc, gk = self.loss_grad(alpha, c, k)
            res.loss.append(J)
            res.path.append(alpha.copy())
            alpha = project_simplex(alpha - eta * ga)  # projected step on the simplex
            c -= eta * gc
            k -= eta * gk
        res.alpha, res.c, res.k = alpha, c, k
        return res
