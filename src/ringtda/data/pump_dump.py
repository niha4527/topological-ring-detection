"""Synthetic pump-and-dump generator (Phase 7 optional track, proposal v3 §3.6). Own module: synthetic.py is untouched.

One asset with a noisy hourly price series and account-level trade events (buy / sell, amount). Planted coordinated groups:

    pump_dump   a group accumulates inside a short band, then ALL members sell within a narrow window around a common exit
                time while the price runs up during accumulation and falls back after the exit (the positives)
    organic     momentum buyers: the same accumulation, but members sell at independent random times over days; the price
                rises gradually and does not crash (benign negative)
    news        a news-driven spike: accumulation, a sharp price spike and decay, unsynchronised sellers (hard negative)
    background  independent traders with random timing and small sizes (benign negatives; rarely enough to form a candidate)

Planted structures give ground truth (account membership and a structure id) and the groups for cross-validation. THE GENERATOR
AUTHOR DECIDES HOW SEPARABLE THE CLASSES ARE: nothing here shows that real pump-and-dump schemes look like this. No topology.
`exit_spread_sd_h` is the hardness knob: the standard deviation (hours) of planted exit times around the common exit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EVENT_COLUMNS = ["timestamp", "account", "side", "amount"]  # what features may see (side: +1 buy, -1 sell)
KINDS = ("pump_dump", "organic", "news")


@dataclass
class PumpDumpConfig:
    seed: int = 0
    n_accounts: int = 300
    duration_h: float = 24.0 * 14
    price_vol: float = 0.004  # hourly log-return std
    bg_trades_per_day: float = 2.0
    bg_amount_logmean: float = 3.2
    bg_amount_logsd: float = 1.0
    n_pump: int = 8
    n_organic: int = 8
    n_news: int = 5
    group_size: tuple[int, int] = (5, 8)
    accum_band_h: float = 12.0
    accum_buys: int = 4  # buys per member inside the band
    acc_amount_logmean: float = 5.0
    acc_amount_logsd: float = 0.3
    exit_gap_h: float = 4.0  # common exit time = end of accumulation band + this
    exit_spread_sd_h: float = 1.0  # planted exits: normal around the common exit, clipped to after the band
    organic_exit_window_h: float = 96.0  # organic sellers: uniform over this many hours after the band
    news_exit_mean_h: float = 36.0  # news sellers: exponential delay after the band
    pump_strength: float = 0.15  # log-price run-up reached at the exit
    organic_strength: float = 0.10
    news_strength: float = 0.15
    fall_h: float = 24.0  # time scale of the post-exit fall
    slot_h: float = 40.0  # planted groups start in separate time slots so that accumulation bands of different groups do not overlap


@dataclass
class PlantedGroup:
    pattern_id: int
    kind: str
    accounts: tuple[int, ...]
    t_start: float
    t_end: float  # end of the accumulation band


def _bump(cfg: PumpDumpConfig, kind: str, t0: float, t1: float, t: np.ndarray) -> np.ndarray:
    """Deterministic log-price contribution of one planted group."""
    out = np.zeros_like(t)
    if kind == "pump_dump":
        te = t1 + cfg.exit_gap_h
        up = (t >= t0) & (t <= te)
        out[up] = cfg.pump_strength * (t[up] - t0) / (te - t0)
        dn = t > te
        out[dn] = cfg.pump_strength * np.exp(-(t[dn] - te) / (cfg.fall_h / 3.0))
    elif kind == "organic":
        out = cfg.organic_strength * np.clip((t - t0) / 96.0, 0.0, 1.0)
    else:  # news: spike at the end of the band, decays
        d = t >= t1
        out[d] = cfg.news_strength * np.exp(-(t[d] - t1) / 12.0)
    return out


def generate(cfg: PumpDumpConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[PlantedGroup]]:
    """(events, price, planted). events carries `pattern_id` and `kind` (ground truth) next to EVENT_COLUMNS; price has hour, price."""
    cfg = cfg or PumpDumpConfig()
    rng = np.random.default_rng(cfg.seed)
    accounts = np.arange(cfg.n_accounts)
    hours = np.arange(0.0, cfg.duration_h + 1.0, 1.0)

    # background traders: independent Poisson arrivals, random side, small lognormal sizes
    n_bg = rng.poisson(cfg.bg_trades_per_day * cfg.duration_h / 24.0 * cfg.n_accounts)
    rows = pd.DataFrame(
        {
            "timestamp": rng.uniform(0, cfg.duration_h, n_bg),
            "account": rng.choice(accounts, n_bg),
            "side": rng.choice([-1, 1], n_bg),
            "amount": rng.lognormal(cfg.bg_amount_logmean, cfg.bg_amount_logsd, n_bg),
            "pattern_id": -1,
            "kind": "background",
        }
    )
    parts = [rows]

    plan = [("pump_dump", cfg.n_pump), ("organic", cfg.n_organic), ("news", cfg.n_news)]
    n_need = sum(n * cfg.group_size[1] for _, n in plan)
    if n_need > cfg.n_accounts:
        raise ValueError("too many planted accounts for n_accounts")
    pool = list(rng.permutation(accounts))
    planted: list[PlantedGroup] = []
    logp = np.cumsum(rng.normal(0.0, cfg.price_vol, len(hours)))
    pid = 0
    schedule = [kind for kind, n in plan for _ in range(n)]
    slots = 24.0 + cfg.slot_h * rng.permutation(len(schedule))
    if slots.max() + cfg.accum_band_h + cfg.organic_exit_window_h + 4 * cfg.news_exit_mean_h > cfg.duration_h:
        raise ValueError("duration_h too short for the planted time slots")
    for kind, slot in zip(schedule, slots):
        if True:
            size = int(rng.integers(cfg.group_size[0], cfg.group_size[1] + 1))
            members = [int(pool.pop()) for _ in range(size)]
            t0 = float(slot + rng.uniform(0.0, 8.0))
            t1 = t0 + cfg.accum_band_h
            ev = []
            for a in members:
                bt = rng.uniform(t0, t1, cfg.accum_buys)
                ba = rng.lognormal(cfg.acc_amount_logmean, cfg.acc_amount_logsd, cfg.accum_buys)
                total = float(ba.sum())
                ev += [(float(t), a, 1, float(x)) for t, x in zip(bt, ba)]
                if kind == "pump_dump":
                    ts = max(t1 + 0.5, t1 + cfg.exit_gap_h + rng.normal(0.0, cfg.exit_spread_sd_h))
                elif kind == "organic":
                    ts = t1 + rng.uniform(0.0, cfg.organic_exit_window_h)
                else:
                    ts = t1 + rng.exponential(cfg.news_exit_mean_h)
                ev.append((float(ts), a, -1, total))
            parts.append(pd.DataFrame(ev, columns=EVENT_COLUMNS).assign(pattern_id=pid, kind=kind))
            logp = logp + _bump(cfg, kind, t0, t1, hours)
            planted.append(PlantedGroup(pid, kind, tuple(sorted(members)), t0, t1))
            pid += 1
    events = pd.concat(parts, ignore_index=True).sort_values("timestamp", kind="stable").reset_index(drop=True)
    price = pd.DataFrame({"hour": hours, "price": 100.0 * np.exp(logp)})
    return events, price, planted
