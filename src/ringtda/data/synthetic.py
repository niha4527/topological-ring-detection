"""Synthetic transaction generator with planted rings (Phase 1).

Produces a transaction table plus the ground-truth list of planted structures,
so that every later stage can be tested against known answers.

Week-1 scope:
    * background traffic: heavy-tailed account activity, irregular timing,
      lognormal amounts, mostly one-directional;
    * planted wash rings of configurable sizes: near-equal amounts moving
      around a directed cycle on a regular cadence;
    * transitive decoy triangles (a->b, b->c, a->c): regular and similar in
      amount, but one-way flow rather than a loop.

    * hard negatives: arbitrage loops (price-triggered timing, profit per hop)
      and market makers (two-way trading with clients, random or scheduled);
    * drift / evasion knobs: ring_jitter, amount_noise, wash_decoy_frac, mm_regularity.

TODO (proposal v3 §3.7): dense-neighbourhood rings, chorded rings,
shortcut-filled pseudo-rings, exchange hot wallets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

COLUMNS = ["timestamp", "src", "dst", "amount", "pattern_id", "pattern"]


@dataclass
class Planted:
    """A planted structure. `accounts` is in cycle order for rings."""

    pattern_id: int
    pattern: str  # "wash_ring" | "transitive_decoy"
    accounts: tuple[int, ...]
    edges: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class SyntheticConfig:
    n_accounts: int = 400
    n_background: int = 6000
    duration: float = 7 * 86400.0  # seconds (one week)
    ring_sizes: tuple[int, ...] = (2, 3, 5)
    rings_per_size: int = 3
    ring_period: float = 4 * 3600.0  # one round every ~4h
    ring_jitter: float = 0.05  # timing jitter as fraction of period
    amount_noise: float = 0.01  # wash amounts differ by ~1%
    wash_decoy_frac: float = 0.0  # evasion: each round, members also send this fraction of x to outside accounts
    n_transitive_decoys: int = 4
    # hard negatives: legitimate loops (proposal v3 §3.7)
    n_arbitrage_loops: int = 4  # 3-party loops, price-triggered (irregular) timing, small profit per hop
    arb_events_per_day: float = 5.0
    arb_profit: float = 0.03
    n_market_makers: int = 2  # hub trading both ways with its clients, random arrivals and sizes
    mm_clients: int = 5
    mm_trades_per_day: float = 12.0
    mm_regularity: float = 0.0  # 0 = random arrivals; 1 = algorithmic quoting on a fixed schedule
    activity_zipf: float = 1.1  # heavier tail -> a few very active accounts
    seed: int = 0


def _background(cfg: SyntheticConfig, rng: np.random.Generator, accounts: np.ndarray) -> pd.DataFrame:
    ranks = np.arange(1, len(accounts) + 1)
    p = 1.0 / ranks**cfg.activity_zipf
    p /= p.sum()
    src = rng.choice(accounts, size=cfg.n_background, p=p)
    dst = rng.choice(accounts, size=cfg.n_background, p=rng.permutation(p))
    keep = src != dst
    n = int(keep.sum())
    return pd.DataFrame(
        {
            "timestamp": rng.uniform(0, cfg.duration, n),
            "src": src[keep],
            "dst": dst[keep],
            "amount": rng.lognormal(mean=4.0, sigma=1.2, size=n),
            "pattern_id": -1,
            "pattern": "background",
        }
    )


def _regular_times(cfg: SyntheticConfig, rng: np.random.Generator) -> np.ndarray:
    offset = rng.uniform(0, cfg.ring_period)
    base = np.arange(offset, cfg.duration, cfg.ring_period)
    return base + rng.normal(0, cfg.ring_jitter * cfg.ring_period, size=len(base))


def _wash_ring(cfg, rng, pid, members, outsiders=None) -> tuple[pd.DataFrame, Planted]:
    k = len(members)
    edges = [(members[i], members[(i + 1) % k]) for i in range(k)]
    rows = []
    hop = 0.02 * cfg.ring_period  # each hop shortly after the previous one
    for t0 in _regular_times(cfg, rng):
        x = rng.lognormal(mean=5.0, sigma=0.3)
        for h, (a, b) in enumerate(edges):
            amt = x * (1 + rng.normal(0, cfg.amount_noise))
            rows.append((t0 + h * hop, a, b, amt, pid, "wash_ring"))
            if cfg.wash_decoy_frac > 0 and outsiders is not None:
                # evasion tactic: a one-way decoy trade breaks the member's net-zero balance
                o = int(rng.choice(outsiders))
                rows.append((t0 + h * hop + rng.uniform(0, cfg.ring_period), a, o, x * cfg.wash_decoy_frac * rng.uniform(0.5, 1.5), -1, "decoy"))
    return pd.DataFrame(rows, columns=COLUMNS), Planted(pid, "wash_ring", tuple(members), edges)


def _transitive_decoy(cfg, rng, pid, members) -> tuple[pd.DataFrame, Planted]:
    a, b, c = members
    edges = [(a, b), (b, c), (a, c)]
    rows = []
    for t0 in _regular_times(cfg, rng):
        x = rng.lognormal(mean=5.0, sigma=0.3)
        for h, (u, v) in enumerate(edges):
            rows.append((t0 + h * 60.0, u, v, x * (1 + rng.normal(0, cfg.amount_noise)), pid, "transitive_decoy"))
    return pd.DataFrame(rows, columns=COLUMNS), Planted(pid, "transitive_decoy", tuple(members), edges)


def _poisson_times(cfg, rng, per_day) -> np.ndarray:
    n = rng.poisson(per_day * cfg.duration / 86400.0)
    return np.sort(rng.uniform(0, cfg.duration, n))


def _arbitrage_loop(cfg, rng, pid, members) -> tuple[pd.DataFrame, Planted]:
    """Legitimate loop: trades fire on random price events (irregular timing) and
    each hop carries a small profit, so value grows around the loop."""
    k = len(members)
    edges = [(members[i], members[(i + 1) % k]) for i in range(k)]
    rows = []
    for t0 in _poisson_times(cfg, rng, cfg.arb_events_per_day):
        x = rng.lognormal(mean=5.0, sigma=0.6)
        for h, (a, b) in enumerate(edges):
            rows.append((t0 + h * 30.0, a, b, x * (1 + cfg.arb_profit) ** h, pid, "arbitrage_loop"))
    return pd.DataFrame(rows, columns=COLUMNS), Planted(pid, "arbitrage_loop", tuple(members), edges)


def _market_maker(cfg, rng, pid, hub, clients) -> tuple[pd.DataFrame, Planted]:
    """Legitimate two-way trading: clients arrive at random and buy or sell a
    random size, so hub<->client pairs are reciprocal-ish but neither equal nor regular."""
    rows = []
    edges = set()
    rate = cfg.mm_trades_per_day / len(clients) * 2  # trades per day per client
    for c in clients:
        if cfg.mm_regularity > 0:
            period = 86400.0 / rate
            base = np.arange(rng.uniform(0, period), cfg.duration, period)
            times = np.sort(base + rng.normal(0, (1 - cfg.mm_regularity) * period, len(base)))
        else:
            times = _poisson_times(cfg, rng, rate)
        for t in times:
            amt = rng.lognormal(mean=4.5, sigma=0.8)
            a, b = (hub, c) if rng.random() < 0.5 else (c, hub)
            rows.append((t, a, b, amt, pid, "market_maker"))
            edges.add((a, b))
    return pd.DataFrame(rows, columns=COLUMNS), Planted(pid, "market_maker", (hub, *clients), sorted(edges))


def generate(cfg: SyntheticConfig | None = None) -> tuple[pd.DataFrame, list[Planted]]:
    """Return (transactions, planted). Planted accounts are disjoint from each other.

    Planted accounts also take part in background traffic, so rings are not
    trivially isolated.
    """
    cfg = cfg or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    accounts = np.arange(cfg.n_accounts)

    n_needed = (
        sum(cfg.ring_sizes) * cfg.rings_per_size
        + 3 * cfg.n_transitive_decoys
        + 3 * cfg.n_arbitrage_loops
        + (1 + cfg.mm_clients) * cfg.n_market_makers
    )
    if n_needed > cfg.n_accounts // 2:
        raise ValueError("too many planted accounts for n_accounts")
    # draw planted members from the less active half so they are not hubs
    pool = list(rng.permutation(accounts[cfg.n_accounts // 2 :]))

    frames = [_background(cfg, rng, accounts)]
    planted: list[Planted] = []
    pid = 0
    for size in cfg.ring_sizes:
        for _ in range(cfg.rings_per_size):
            members = [int(pool.pop()) for _ in range(size)]
            df, p = _wash_ring(cfg, rng, pid, members, outsiders=accounts[: cfg.n_accounts // 2])
            frames.append(df)
            planted.append(p)
            pid += 1
    for _ in range(cfg.n_transitive_decoys):
        members = [int(pool.pop()) for _ in range(3)]
        df, p = _transitive_decoy(cfg, rng, pid, members)
        frames.append(df)
        planted.append(p)
        pid += 1
    for _ in range(cfg.n_arbitrage_loops):
        members = [int(pool.pop()) for _ in range(3)]
        df, p = _arbitrage_loop(cfg, rng, pid, members)
        frames.append(df)
        planted.append(p)
        pid += 1
    for _ in range(cfg.n_market_makers):
        hub = int(pool.pop())
        clients = [int(pool.pop()) for _ in range(cfg.mm_clients)]
        df, p = _market_maker(cfg, rng, pid, hub, clients)
        frames.append(df)
        planted.append(p)
        pid += 1

    tx = pd.concat(frames, ignore_index=True)
    tx = tx[(tx.timestamp >= 0) & (tx.timestamp < cfg.duration)]
    tx = tx.sort_values("timestamp", kind="stable").reset_index(drop=True)
    tx["src"] = tx["src"].astype(int)
    tx["dst"] = tx["dst"].astype(int)
    return tx, planted
