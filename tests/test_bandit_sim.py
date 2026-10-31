import numpy as np

from ringtda.threshold.bandit_sim import DiscountedUCB, SlidingWindowTS, arm_grid, normalise, reward_matrix, run_bandit


def stream(n=120, K=4, best=2, seed=0):
    rng = np.random.default_rng(seed)
    R = rng.normal(0, 0.3, size=(n, K))
    R[:, best] += 4.5
    return R


def test_ducb_converges_to_the_best_arm():
    R = stream()
    out = run_bandit(R, DiscountedUCB(4, gamma=0.98, c=0.1), d=1, warmup=3, scale=5.0)
    assert (out["arm"][-40:] == 2).mean() > 0.75


def test_ts_converges_to_the_best_arm():
    R = stream()
    out = run_bandit(R, SlidingWindowTS(4, window=60, seed=1), d=1, warmup=3, scale=5.0)
    assert (out["arm"][-40:] == 2).mean() > 0.75


def test_no_reward_is_used_before_its_lag():
    R = stream(n=30)
    for d in (0, 1, 2, 5):
        out = run_bandit(R, DiscountedUCB(4), d=d, warmup=3, scale=5.0)
        assert out["used_at"] and all(t - s >= max(d, 1) for s, t in out["used_at"])


def test_delayed_feedback_hides_recent_rewards_from_the_policy():
    seen = []

    class Spy(DiscountedUCB):
        def update(self, s, arm, u):
            seen.append(s)
            super().update(s, arm, u)

    run_bandit(stream(n=12), Spy(4), d=5, warmup=2, scale=5.0)
    assert max(seen) <= 12 - 1 - 5  # the last windows' rewards never arrived


def test_sliding_window_forgets_old_data_and_ucb_discounts_it():
    ts = SlidingWindowTS(2, window=5, seed=0)
    for s in range(10):  # arm 0 great long ago, arm 1 great recently
        ts.update(s, 0, 1.0 if s < 5 else 0.0)
        ts.update(s, 1, 0.0 if s < 5 else 1.0)
    picks = [ts.choose(10) for _ in range(30)]
    assert np.mean(np.array(picks) == 1) > 0.8
    uc = DiscountedUCB(2, gamma=0.5, c=0.0)
    for s in range(10):
        uc.update(s, 0, 1.0 if s < 5 else 0.0)
        uc.update(s, 1, 0.0 if s < 5 else 1.0)
    assert uc.choose(10) == 1


def test_determinism_with_a_seed():
    R = stream()
    a = run_bandit(R, SlidingWindowTS(4, window=10, seed=7), d=2, warmup=3, scale=5.0)["arm"]
    b = run_bandit(R, SlidingWindowTS(4, window=10, seed=7), d=2, warmup=3, scale=5.0)["arm"]
    c = run_bandit(R, DiscountedUCB(4), d=2, warmup=3, scale=5.0)["arm"]
    d = run_bandit(R, DiscountedUCB(4), d=2, warmup=3, scale=5.0)["arm"]
    assert (a == b).all() and (c == d).all()


def test_reward_matrix_and_normalisation():
    s = np.array([0.1, 0.2, 0.9])
    y = np.array([1.0, 0.0, 0.0])
    taus = arm_grid(3, 0.15, 0.95)
    R = reward_matrix([(s, y)], taus, lam=2.0)
    assert R.shape == (1, 3) and R[0, 0] == 1.0 and R[0, 1] == -1.0  # tau .15: TP; tau .55: TP + FP(.2) -> 1 - 2
    assert normalise([-100, 0, 100], 10).tolist() == [0.0, 0.5, 1.0]
