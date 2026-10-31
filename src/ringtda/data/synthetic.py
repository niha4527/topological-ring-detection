"""Synthetic transaction generator with planted rings (Phase 1, proposal v3 §3.7).

Produces a transaction table plus the ground-truth list of planted structures,
so that every later stage can be tested against known answers.

Background: heavy-tailed account activity, irregular timing, lognormal amounts,
mostly one-directional.

Positives (pattern "wash_ring"; `Planted.scenario` says which kind):
    isolated   near-equal amounts moving around a directed cycle on a regular
               cadence, in a sparse neighbourhood (topology ~ redundant here).
    dense      the same rings embedded among extra accounts with random,
               irregular traffic between them -> many incidental loops.
    chorded    a longer ring plus forward chords; each round takes either the
               full ring or a shortcut through a chord, so every node still
               nets to zero but the loop is split into shorter cycles.

Topology-sensitive negatives:
    transitive_decoy  a->b, b->c, a->c: regular, similar amounts, one-way flow.
    shortcut_filled   a regular cyclic triangle a->b->c->a plus an extra account
                      d paying all three (d->a, d->b, d->c). The transitive
                      triangles through d fill the loop: ring-like under
                      enumeration, but its H1 class dies (§3.4).

Hard negatives (legitimate loops; switch off with `hard_negatives=False` for the
"without hard negatives" runs, §4.9):
    arbitrage_loop  3-party loops fired by random price events (irregular
                    timing); each hop carries a small profit, so value grows
                    around the loop instead of netting to zero.
    market_maker    hub trading both ways with its clients: random arrivals and
                    sizes (mm_regularity 0) up to algorithmic quoting on a fixed
                    schedule (mm_regularity 1).
    hot_wallet      exchange hub: many users deposit and withdraw at random
                    times; aggregate in- and outflow are nearly equal.

Drift / evasion knobs: ring_jitter, amount_noise, wash_decoy_frac, mm_regularity.

Transaction rows carry the structure's `pattern_id`. Rows around a structure
rather than part of it have pattern "context" (dense-neighbourhood traffic) or
"decoy" (wash-ring evasion trades, pattern_id -1).

Draw order: background, isolated rings, transitive decoys, dense, chorded,
shortcut-filled, then hard negatives. With n_dense/chorded/shortcut set to 0 the
output is identical to the generator before those scenarios were added, and
toggling hard negatives leaves everything drawn before them unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

COLUMNS = ["timestamp", "src", "dst", "amount", "pattern_id", "pattern"]
POSITIVE = "wash_ring"
HARD_NEGATIVES = ("arbitrage_loop", "market_maker", "hot_wallet")


@dataclass
class Planted:
    """A planted structure. `accounts` is in cycle order for rings.

    edges          the structure itself; for rings and triangles, the cycle a
                   candidate generator should (or should not) return.
    context_edges  other edges planted alongside it: chords, the filling
                   account's edges, dense-neighbourhood traffic.
    """

    pattern_id: int
    pattern: str  # "wash_ring" | "transitive_decoy" | "shortcut_filled" | one of HARD_NEGATIVES
    accounts: tuple[int, ...]
    edges: list[tuple[int, int]] = field(default_factory=list)
    scenario: str = "isolated"  # wash rings: "isolated" | "dense" | "chorded"; others: their pattern
    context_edges: list[tuple[int, int]] = field(default_factory=list)
    context_accounts: tuple[int, ...] = ()  # e.g. a dense ring's neighbours

    @property
    def is_positive(self) -> bool:
        return self.pattern == POSITIVE


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
    # wash rings in dense neighbourhoods
    dense_ring_sizes: tuple[int, ...] = (3, 5)
    dense_rings_per_size: int = 2
    dense_neighbours: int = 8  # extra accounts around each ring
    dense_edge_prob: float = 0.3  # chance of incidental traffic on each ordered pair
    # False: no incidental edges between two members of the same planted ring (the ring is then
    # embedded among loops of OTHER accounts only, §3.7). The random draws are unchanged, so every
    # other edge and structure is identical to the default; the default reproduces earlier windows.
    dense_internal_chords: bool = True
    # chorded wash rings; chords are (i, j) positions in ring order, i < j - 1
    chorded_ring_size: int = 6
    n_chorded_rings: int = 2
    chords: tuple[tuple[int, int], ...] = ((0, 3),)
    # shortcut-filled pseudo-rings
    n_shortcut_filled: int = 3
    shortcut_fill_scale: float = 0.3  # filler amounts relative to the ring's
    # hard negatives: legitimate loops (proposal v3 §3.7)
    hard_negatives: bool = True
    n_arbitrage_loops: int = 4  # 3-party loops, price-triggered (irregular) timing, small profit per hop
    arb_events_per_day: float = 5.0
    arb_profit: float = 0.03
    n_market_makers: int = 2  # hub trading both ways with its clients, random arrivals and sizes
    mm_clients: int = 5
    mm_trades_per_day: float = 12.0
    mm_regularity: float = 0.0  # 0 = random arrivals; 1 = algorithmic quoting on a fixed schedule
    n_hot_wallets: int = 2
    hot_wallet_users: int = 150
    hot_wallet_imbalance: float = 0.02  # std of (total out / total in - 1)
    activity_zipf: float = 1.1  # heavier tail -> a few very active accounts
    seed: int = 0


def _rows(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=COLUMNS)


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


def _poisson_times(cfg, rng, per_day) -> np.ndarray:
    n = rng.poisson(per_day * cfg.duration / 86400.0)
    return np.sort(rng.uniform(0, cfg.duration, n))


def _ring_edges(members) -> list[tuple[int, int]]:
    k = len(members)
    return [(members[i], members[(i + 1) % k]) for i in range(k)]


def _cycle_rounds(cfg, rng, pid, routes, outsiders=None, pattern=POSITIVE) -> list[tuple]:
    """Regular rounds of near-equal amounts; round r follows routes[r % len(routes)].

    With wash_decoy_frac > 0 and outsiders given, each hop also sends a one-way
    decoy trade to an outside account (evasion: breaks the members' balance).
    """
    rows = []
    hop = 0.02 * cfg.ring_period  # each hop shortly after the previous one
    for r, t0 in enumerate(_regular_times(cfg, rng)):
        x = rng.lognormal(mean=5.0, sigma=0.3)
        for h, (a, b) in enumerate(routes[r % len(routes)]):
            amt = x * (1 + rng.normal(0, cfg.amount_noise))
            rows.append((t0 + h * hop, a, b, amt, pid, pattern))
            if cfg.wash_decoy_frac > 0 and outsiders is not None:
                o = int(rng.choice(outsiders))
                rows.append((t0 + h * hop + rng.uniform(0, cfg.ring_period), a, o, x * cfg.wash_decoy_frac * rng.uniform(0.5, 1.5), -1, "decoy"))
    return rows


def _wash_ring(cfg, rng, pid, members, outsiders=None) -> tuple[pd.DataFrame, Planted]:
    edges = _ring_edges(members)
    return _rows(_cycle_rounds(cfg, rng, pid, [edges], outsiders)), Planted(pid, POSITIVE, tuple(members), edges)


def _dense_ring(cfg, rng, pid, members, neighbours, outsiders=None) -> tuple[pd.DataFrame, Planted]:
    """Wash ring plus irregular traffic among ring members and neighbours.

    Each ordered pair of the combined account set (other than the ring's own
    edges) carries traffic with probability dense_edge_prob, which produces
    many incidental directed cycles around the ring.
    """
    edges = _ring_edges(members)
    rows = _cycle_rounds(cfg, rng, pid, [edges], outsiders)
    ring = set(edges)
    nodes = list(members) + list(neighbours)
    inside = set(members)
    context = []
    for a in nodes:
        for b in nodes:
            if a == b or (a, b) in ring or rng.random() >= cfg.dense_edge_prob:
                continue
            n = 1 + rng.poisson(2)
            times, amts = rng.uniform(0, cfg.duration, n), rng.lognormal(4.0, 1.0, n)  # draw first: keeps the stream fixed
            if not cfg.dense_internal_chords and a in inside and b in inside:
                continue
            context.append((a, b))
            for t, amt in zip(times, amts):
                rows.append((t, a, b, amt, pid, "context"))
    p = Planted(
        pid, POSITIVE, tuple(members), edges, scenario="dense", context_edges=context, context_accounts=tuple(neighbours)
    )
    return _rows(rows), p


def _chorded_ring(cfg, rng, pid, members, outsiders=None) -> tuple[pd.DataFrame, Planted]:
    """Ring with forward chords m_i -> m_j. Routes: the full ring, and for each
    chord the closed shortcut m_i -> m_j -> m_{j+1} -> ... -> m_i. Every route
    is a closed cycle, so all members stay balanced."""
    k = len(members)
    edges = _ring_edges(members)
    routes = [edges]
    chord_edges = []
    for i, j in cfg.chords:
        if not 0 <= i < j - 1 < k - 1:
            raise ValueError(f"chord {(i, j)} must satisfy 0 <= i < j-1 < ring size - 1")
        chord_edges.append((members[i], members[j]))
        path = [members[i]] + [members[(j + s) % k] for s in range(k - (j - i))]
        routes.append(_ring_edges(path))
    p = Planted(pid, POSITIVE, tuple(members), edges, scenario="chorded", context_edges=chord_edges)
    return _rows(_cycle_rounds(cfg, rng, pid, routes, outsiders)), p


def _transitive_decoy(cfg, rng, pid, members) -> tuple[pd.DataFrame, Planted]:
    a, b, c = members
    edges = [(a, b), (b, c), (a, c)]
    rows = []
    for t0 in _regular_times(cfg, rng):
        x = rng.lognormal(mean=5.0, sigma=0.3)
        for h, (u, v) in enumerate(edges):
            rows.append((t0 + h * 60.0, u, v, x * (1 + rng.normal(0, cfg.amount_noise)), pid, "transitive_decoy"))
    return _rows(rows), Planted(pid, "transitive_decoy", tuple(members), edges, scenario="transitive_decoy")


def _shortcut_filled(cfg, rng, pid, members, filler) -> tuple[pd.DataFrame, Planted]:
    """Cyclic triangle a->b->c->a (wash-like rounds) filled by d->a, d->b, d->c."""
    edges = _ring_edges(members)
    rows = _cycle_rounds(cfg, rng, pid, [edges], pattern="shortcut_filled")
    fill = [(filler, m) for m in members]
    for t0 in _regular_times(cfg, rng):
        x = cfg.shortcut_fill_scale * rng.lognormal(mean=5.0, sigma=0.3)
        for h, (u, v) in enumerate(fill):
            rows.append((t0 + h * 60.0, u, v, x * (1 + rng.normal(0, cfg.amount_noise)), pid, "shortcut_filled"))
    p = Planted(pid, "shortcut_filled", (*members, filler), edges, scenario="shortcut_filled", context_edges=fill)
    return _rows(rows), p


def _arbitrage_loop(cfg, rng, pid, members) -> tuple[pd.DataFrame, Planted]:
    """Legitimate loop: trades fire on random price events (irregular timing) and
    each hop carries a small profit, so value grows around the loop."""
    edges = _ring_edges(members)
    rows = []
    for t0 in _poisson_times(cfg, rng, cfg.arb_events_per_day):
        x = rng.lognormal(mean=5.0, sigma=0.6)
        for h, (a, b) in enumerate(edges):
            rows.append((t0 + h * 30.0, a, b, x * (1 + cfg.arb_profit) ** h, pid, "arbitrage_loop"))
    return _rows(rows), Planted(pid, "arbitrage_loop", tuple(members), edges, scenario="arbitrage_loop")


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
    return _rows(rows), Planted(pid, "market_maker", (hub, *clients), sorted(edges), scenario="market_maker")


def _hot_wallet(cfg, rng, pid, hub, users) -> tuple[pd.DataFrame, Planted]:
    """Deposits user -> hub and withdrawals hub -> user at Poisson times, with
    per-user rates; withdrawals are rescaled so total out ~= total in."""
    dep, wd = [], []
    for u in users:
        per_day_d, per_day_w = rng.lognormal(np.log(1.5), 0.8, size=2) / 7  # ~1.5 of each per user per week
        dep += [(t, u, hub, rng.lognormal(4.5, 1.0)) for t in _poisson_times(cfg, rng, per_day_d)]
        wd += [(t, hub, u, rng.lognormal(4.5, 1.0)) for t in _poisson_times(cfg, rng, per_day_w)]
    if dep and wd:
        scale = sum(r[3] for r in dep) / sum(r[3] for r in wd) * (1 + rng.normal(0, cfg.hot_wallet_imbalance))
        wd = [(t, a, b, amt * scale) for t, a, b, amt in wd]
    rows = [(*r, pid, "hot_wallet") for r in dep + wd]
    edges = sorted({(a, b) for _, a, b, _ in dep + wd})
    return _rows(rows), Planted(pid, "hot_wallet", (hub, *users), edges, scenario="hot_wallet")


def _n_planted_accounts(cfg: SyntheticConfig) -> int:
    n = sum(cfg.ring_sizes) * cfg.rings_per_size + 3 * cfg.n_transitive_decoys
    n += (sum(cfg.dense_ring_sizes) + len(cfg.dense_ring_sizes) * cfg.dense_neighbours) * cfg.dense_rings_per_size
    n += cfg.chorded_ring_size * cfg.n_chorded_rings + 4 * cfg.n_shortcut_filled
    if cfg.hard_negatives:
        n += 3 * cfg.n_arbitrage_loops + (1 + cfg.mm_clients) * cfg.n_market_makers + cfg.n_hot_wallets
    return n


def generate(cfg: SyntheticConfig | None = None) -> tuple[pd.DataFrame, list[Planted]]:
    """Return (transactions, planted). Planted accounts are disjoint from each other.

    Planted accounts also take part in background traffic, so rings are not
    trivially isolated. Hot-wallet users are ordinary non-planted accounts.
    """
    cfg = cfg or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    accounts = np.arange(cfg.n_accounts)

    if _n_planted_accounts(cfg) > cfg.n_accounts // 2:
        raise ValueError("too many planted accounts for n_accounts")
    # draw planted members from the less active half so they are not hubs
    pool = list(rng.permutation(accounts[cfg.n_accounts // 2 :]))
    outsiders = accounts[: cfg.n_accounts // 2]  # targets of wash-ring decoy trades

    def take(n: int) -> list[int]:
        return [int(pool.pop()) for _ in range(n)]

    frames = [_background(cfg, rng, accounts)]
    planted: list[Planted] = []

    def add(df: pd.DataFrame, p: Planted) -> None:
        frames.append(df)
        planted.append(p)

    for size in cfg.ring_sizes:
        for _ in range(cfg.rings_per_size):
            add(*_wash_ring(cfg, rng, len(planted), take(size), outsiders))
    for _ in range(cfg.n_transitive_decoys):
        add(*_transitive_decoy(cfg, rng, len(planted), take(3)))
    for size in cfg.dense_ring_sizes:
        for _ in range(cfg.dense_rings_per_size):
            add(*_dense_ring(cfg, rng, len(planted), take(size), take(cfg.dense_neighbours), outsiders))
    for _ in range(cfg.n_chorded_rings):
        add(*_chorded_ring(cfg, rng, len(planted), take(cfg.chorded_ring_size), outsiders))
    for _ in range(cfg.n_shortcut_filled):
        add(*_shortcut_filled(cfg, rng, len(planted), take(3), take(1)[0]))

    if cfg.hard_negatives:
        for _ in range(cfg.n_arbitrage_loops):
            add(*_arbitrage_loop(cfg, rng, len(planted), take(3)))
        for _ in range(cfg.n_market_makers):
            hub = take(1)[0]
            add(*_market_maker(cfg, rng, len(planted), hub, take(cfg.mm_clients)))
        hubs = take(cfg.n_hot_wallets)
        used = {a for p in planted for a in (*p.accounts, *p.context_accounts)} | set(hubs)
        free = np.setdiff1d(accounts, list(used))
        for h in hubs:
            add(*_hot_wallet(cfg, rng, len(planted), h, rng.choice(free, cfg.hot_wallet_users, replace=False).tolist()))

    tx = pd.concat(frames, ignore_index=True)
    tx = tx[(tx.timestamp >= 0) & (tx.timestamp < cfg.duration)]
    tx = tx.sort_values("timestamp", kind="stable").reset_index(drop=True)
    tx["src"] = tx["src"].astype(int)
    tx["dst"] = tx["dst"].astype(int)
    return tx, planted
