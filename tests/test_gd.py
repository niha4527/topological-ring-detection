"""Correctness of the optimization code (report §2.5-2.7)."""

import numpy as np

from ringtda.models.gd import (
    AlphaLearner,
    class_weights,
    gradient_descent,
    logistic_loss_grad,
    project_simplex,
    smoothness_constant,
)
from ringtda.threshold.ascent import reward, smoothed_reward_grad


def test_projection_worked_example():
    # the worked example in report §2.6
    assert np.allclose(project_simplex([0.5, 0.8, -0.1]), [0.35, 0.65, 0.0])


def test_projection_properties():
    rng = np.random.default_rng(0)
    for _ in range(200):
        u = rng.normal(0, 2, 3)
        a = project_simplex(u)
        assert np.all(a >= -1e-12) and np.isclose(a.sum(), 1.0)
        # optimality: no other simplex point is closer (check random candidates)
        for _ in range(20):
            b = project_simplex(rng.normal(0, 2, 3))
            assert np.linalg.norm(a - u) <= np.linalg.norm(b - u) + 1e-9
    assert np.allclose(project_simplex([0.2, 0.3, 0.5]), [0.2, 0.3, 0.5])  # already feasible


def _toy(seed=0, n=200, d=4):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=(n, d))])
    y = (X @ rng.normal(size=d + 1) + rng.normal(0, 0.5, n) > 0.8).astype(float)
    return X, y, class_weights(y)


def test_logistic_gradient_matches_finite_differences():
    X, y, w = _toy()
    th = np.random.default_rng(1).normal(size=X.shape[1])
    _, g = logistic_loss_grad(th, X, y, w, 0.1)
    num = np.array([
        (logistic_loss_grad(th + e, X, y, w, 0.1)[0] - logistic_loss_grad(th - e, X, y, w, 0.1)[0]) / 2e-6
        for e in np.eye(len(th)) * 1e-6
    ])
    assert np.abs(num - g).max() < 1e-6


def test_gd_with_step_1_over_L_decreases_monotonically():
    X, y, w = _toy()
    L = smoothness_constant(X, w, 0.1)
    loss = np.array(gradient_descent(X, y, w, 0.1, eta=1 / L, iters=300).loss)
    assert np.all(np.diff(loss) <= 1e-12)  # descent lemma


def test_alpha_gradient_matches_finite_differences():
    rng = np.random.default_rng(2)
    rings = [rng.uniform(0, 1, (rng.integers(2, 6), 3)) for _ in range(60)]
    y = rng.integers(0, 2, 60)
    learner = AlphaLearner(rings, y, beta=20.0)
    a, c, k = np.array([0.5, 0.3, 0.2]), 0.4, 3.0
    J, ga, gc, gk = learner.loss_grad(a, c, k)
    h = 1e-6
    num_a = [(learner.loss_grad(a + e, c, k)[0] - learner.loss_grad(a - e, c, k)[0]) / (2 * h) for e in np.eye(3) * h]
    assert np.abs(np.array(num_a) - ga).max() < 1e-7
    assert abs((learner.loss_grad(a, c + h, k)[0] - learner.loss_grad(a, c - h, k)[0]) / (2 * h) - gc) < 1e-7
    assert abs((learner.loss_grad(a, c, k + h)[0] - learner.loss_grad(a, c, k - h)[0]) / (2 * h) - gk) < 1e-7


def test_alpha_learner_finds_the_informative_score():
    # only the third score separates the classes -> alpha should move toward (0, 0, 1)
    rng = np.random.default_rng(3)
    rings, y = [], []
    for i in range(300):
        pos = i % 3 == 0
        z = rng.uniform(0, 1, (3, 3))
        z[:, 2] = rng.uniform(0.8, 1.0, 3) if pos else rng.uniform(0.0, 0.5, 3)
        rings.append(z)
        y.append(int(pos))
    res = AlphaLearner(rings, y).fit(iters=800)
    assert res.alpha[2] > 0.8 and np.isclose(res.alpha.sum(), 1.0)
    assert res.loss[-1] < res.loss[0]


def test_smoothed_reward_gradient_and_sign():
    s = np.array([0.1, 0.12, 0.5, 0.55])
    y = np.array([1.0, 1.0, 0.0, 0.0])
    assert reward(0.2, s, y, 1.0) == 2.0
    # just above the positives the gradient pushes tau up (catch more); just below the negatives, down
    assert smoothed_reward_grad(0.13, s, y, 1.0, 0.02) > 0
    assert smoothed_reward_grad(0.49, s, y, 1.0, 0.02) < 0
