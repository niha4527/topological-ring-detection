"""Pump-and-dump temporal-motif candidates and features (proposal v3 §3.6). No topology.

Candidates: coordinated-accumulation windows, found from trades alone. In a band of `band_h` hours (stride `stride_h`), an account
*accumulates* if its buy volume in the band is at least `acc_min`; a window with at least `m_min` accumulating accounts is a
candidate (accounts, band). Near-duplicate candidates (Jaccard of account sets at least `jaccard`, starts within one band) keep the first.

Features (prefix M_) use ONLY the trade table (timestamp, account, side, amount) and the price series; nothing here receives labels,
planted ids or kinds (`EVENT_COLUMNS` is the whole interface, checked in the tests). They describe what happened AFTER the accumulation
too (exits, price fall): this is retrospective detection of a completed episode, not early warning.
    M_acc_conc        share of the group's buy volume in the busiest quarter of the band (accumulation concentration)
    M_exit_spread_h   std (hours) of the members' first sell after the band (missing sell = horizon); exit synchrony
    M_exit_frac_24h   share of members that sell within 24 h after the band
    M_exit_median_h   median delay of the members' first sell
    M_pos_cv          coefficient of variation of the members' accumulated volume (position size dispersion)
    M_runup           log-price change from band start to band end
    M_drawdown        max drawdown of log price from band start to band end + horizon
    M_size            number of accounts
Baseline set B_pd (no timing, no price): B_pd_size, B_pd_vol_mean, B_pd_vol_max (group size and per-account buy volume).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ringtda.data.pump_dump import EVENT_COLUMNS

M_NAMES = ["M_acc_conc", "M_exit_spread_h", "M_exit_frac_24h", "M_exit_median_h", "M_pos_cv", "M_runup", "M_drawdown", "M_size"]
BPD_NAMES = ["B_pd_size", "B_pd_vol_mean", "B_pd_vol_max"]


def accumulation_candidates(events: pd.DataFrame, duration_h: float, band_h: float, stride_h: float, acc_min: float, m_min: int, jaccard: float = 0.8) -> list[dict]:
    """[{accounts: tuple, t0, t1}] from buys only (no labels)."""
    buys = events.loc[events["side"] == 1, ["timestamp", "account", "amount"]]
    t = buys["timestamp"].to_numpy()
    out: list[dict] = []
    s = 0.0
    while s < duration_h:
        m = (t >= s) & (t < s + band_h)
        if m.any():
            vol = buys[m].groupby("account")["amount"].sum()
            acc = tuple(sorted(int(a) for a in vol.index[vol >= acc_min]))
            if len(acc) >= m_min:
                new = set(acc)
                dup = any(abs(c["t0"] - s) < band_h and len(new & set(c["accounts"])) / len(new | set(c["accounts"])) >= jaccard for c in out)
                if not dup:
                    out.append({"accounts": acc, "t0": s, "t1": s + band_h})
        s += stride_h
    return out


def motif_features(events: pd.DataFrame, price: pd.DataFrame, accounts, t0: float, t1: float, horizon_h: float = 120.0) -> dict[str, float]:
    """One candidate's M_ and B_pd_ features. Only EVENT_COLUMNS of `events` are read."""
    ev = events[EVENT_COLUMNS]
    acc = set(accounts)
    mine = ev[ev["account"].isin(acc)]
    buys = mine[(mine["side"] == 1) & (mine["timestamp"] >= t0) & (mine["timestamp"] < t1)]
    vol = buys.groupby("account")["amount"].sum().reindex(sorted(acc), fill_value=0.0).to_numpy()
    # accumulation concentration: busiest quarter of the band
    q = np.minimum(((buys["timestamp"].to_numpy() - t0) / (t1 - t0) * 4).astype(int), 3)
    share = np.bincount(q, weights=buys["amount"].to_numpy(), minlength=4) if len(buys) else np.zeros(4)
    conc = float(share.max() / share.sum()) if share.sum() > 0 else 0.25
    # first sell after the band, per member (horizon if none)
    sells = mine[(mine["side"] == -1) & (mine["timestamp"] >= t1) & (mine["timestamp"] < t1 + horizon_h)]
    first = sells.groupby("account")["timestamp"].min().reindex(sorted(acc))
    delay = (first - t1).fillna(horizon_h).to_numpy(dtype=float)
    p = price.set_index("hour")["price"]
    lp = np.log(p.to_numpy(dtype=float))
    h = p.index.to_numpy(dtype=float)
    i0, i1 = int(np.searchsorted(h, t0)), int(np.searchsorted(h, t1))
    i2 = int(min(np.searchsorted(h, t1 + horizon_h), len(h) - 1))
    runup = float(lp[min(i1, len(lp) - 1)] - lp[min(i0, len(lp) - 1)])
    seg = lp[min(i0, len(lp) - 1) : i2 + 1]
    dd = float(np.max(np.maximum.accumulate(seg) - seg)) if len(seg) else 0.0
    mean = vol.mean()
    return {
        "M_acc_conc": conc, "M_exit_spread_h": float(delay.std()), "M_exit_frac_24h": float((delay <= 24.0).mean()), "M_exit_median_h": float(np.median(delay)),
        "M_pos_cv": float(vol.std() / mean) if mean > 0 else 0.0, "M_runup": runup, "M_drawdown": dd, "M_size": float(len(acc)),
        "B_pd_size": float(len(acc)), "B_pd_vol_mean": float(mean), "B_pd_vol_max": float(vol.max()),
    }  # fmt: skip


def label_candidate(accounts, planted, x: float = 0.8) -> tuple[str, int]:
    """(kind, planted id). The planted group holding at least x of the candidate's accounts gives its kind; none of them is
    `background`; otherwise `ambiguous`. y = 1 only for kind `pump_dump`. Used for labels and CV groups only, never for features."""
    owner = {a: g for g in planted for a in g.accounts}
    cnt: dict[int, int] = {}
    for a in accounts:
        if a in owner:
            cnt[owner[a].pattern_id] = cnt.get(owner[a].pattern_id, 0) + 1
    if not cnt:
        return "background", -1
    pid, n = max(cnt.items(), key=lambda kv: kv[1])
    if n / len(accounts) >= x:
        return next(g.kind for g in planted if g.pattern_id == pid), pid
    return "ambiguous", -1


def pump_dump_table(cfg, cand: dict, label_x: float = 0.8, group_offset: int = 0) -> pd.DataFrame:
    """One row per candidate: M_ and B_pd_ features, kind, pattern_id, y (NaN = ambiguous), group (seed-prefixed), seed."""
    from ringtda.data.pump_dump import generate
    from ringtda.eval.splits import assign_groups

    events, price, planted = generate(cfg)
    clean = events[EVENT_COLUMNS]  # the features never see pattern_id or kind
    cands = accumulation_candidates(clean, cfg.duration_h, cand["band_h"], cand["stride_h"], cand["acc_min"], cand["m_min"], cand["jaccard"])
    rows = []
    for c in cands:
        kind, pid = label_candidate(c["accounts"], planted, label_x)
        rows.append({**motif_features(clean, price, c["accounts"], c["t0"], c["t1"], cand["horizon_h"]), "kind": kind, "pattern_id": pid, "y": float("nan") if kind == "ambiguous" else float(kind == "pump_dump"), "t0": c["t0"]})
    df = pd.DataFrame(rows)
    if len(df):
        df["group"] = assign_groups([tuple(c["accounts"]) for c in cands], df["pattern_id"].to_numpy()) + group_offset
    df["seed"] = cfg.seed
    return df
