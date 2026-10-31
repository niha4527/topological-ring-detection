"""IBM AML (AMLworld, Altman et al. 2023) loader (proposal v3 §3.7, README "Phase 3 design decisions").

Transactions carry no ID, so laundering attempts from the patterns file are joined to them on
(timestamp, from bank, from account, to bank, to account, amount paid). Rows that share a key
(~24 in HI-Small) are paired deterministically: laundering rows first, then original file order.

Output transaction table (one row per kept transaction):
    timestamp   days from the dataset start (first transaction)
    src, dst    int account ids; an account is (bank, account), see IBMData.accounts
    amount      Amount Paid, in its own currency (15 currencies; limitation, README decision 2)
    pattern_id  attempt id (index in the patterns file), -1 if none
    pattern     attempt pattern type (CYCLE, FAN-OUT, ...), "unattributed" for laundering rows that
                belong to no attempt, "background" otherwise
    laundering  the dataset's Is Laundering flag
Self-loops (src == dst) are dropped and counted (README decision 3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ringtda.graph.windows import iter_windows

log = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "ibm.yaml"
TRANS_COLS = ["ts", "fb", "fa", "tb", "ta", "amt_recv", "cur_recv", "amt_paid", "cur_paid", "fmt", "y"]
KEY = ["ts", "fb", "fa", "tb", "ta", "amt"]
AMT_DECIMALS = 8


def load_config(path: str | Path | None = None) -> dict:
    with open(path or CONFIG_PATH) as f:
        return yaml.safe_load(f)


@dataclass
class IBMData:
    name: str
    tx: pd.DataFrame
    accounts: np.ndarray  # accounts[i] = "bank_account" for int id i
    attempts: pd.DataFrame  # attempt_id, pattern, n_rows, start_day, end_day
    stats: dict = field(default_factory=dict)


def parse_patterns(path: str | Path) -> pd.DataFrame:
    """Patterns file -> one row per attempt transaction: attempt_id, pattern, ts, fb, fa, tb, ta, amt."""
    rows, attempt, ptype = [], -1, None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                attempt += 1
                ptype = line.split(" - ", 1)[1].split(":")[0].strip()
            elif line.startswith("END LAUNDERING") or not line:
                continue
            else:
                p = line.split(",")
                rows.append((attempt, ptype, p[0], p[1], p[2], p[3], p[4], float(p[7])))
    df = pd.DataFrame(rows, columns=["attempt_id", "pattern", "ts", "fb", "fa", "tb", "ta", "amt"])
    df["ts"] = pd.to_datetime(df["ts"], format="%Y/%m/%d %H:%M")
    df["amt"] = df["amt"].round(AMT_DECIMALS)
    return df


def join_patterns(trans: pd.DataFrame, pat: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Map each pattern row to a transaction index. Returns (tidx per pattern row, NaN if unmatched; stats).

    `trans` has KEY columns plus `y`. Duplicate keys: laundering transactions are claimed first, then
    original order; the k-th pattern row with a key takes the k-th transaction with that key.
    """
    n_dup = int(trans.duplicated(KEY, keep=False).sum())
    keys = pat[KEY].drop_duplicates()
    sub = trans[KEY + ["y"]].reset_index(names="tidx").merge(keys, on=KEY, how="inner")
    sub = sub.sort_values(["y", "tidx"], ascending=[False, True], kind="stable")
    sub["rank"] = sub.groupby(KEY, sort=False).cumcount()
    p = pat[KEY].copy()
    p["rank"] = p.groupby(KEY, sort=False).cumcount()
    p = p.reset_index(names="prow").merge(sub[KEY + ["rank", "tidx"]], on=KEY + ["rank"], how="left").sort_values("prow")
    tidx = pd.Series(p["tidx"].to_numpy(), index=pat.index)
    stats = {"n_duplicate_key_rows": n_dup, "n_pattern_rows": len(pat), "n_pattern_unmatched": int(tidx.isna().sum())}
    return tidx, stats


def load_ibm_files(trans_path: str | Path, patterns_path: str | Path, name: str = "IBM") -> IBMData:
    df = pd.read_csv(
        trans_path, header=0, names=TRANS_COLS, usecols=["ts", "fb", "fa", "tb", "ta", "amt_paid", "y"],
        dtype={"fb": str, "fa": str, "tb": str, "ta": str}, parse_dates=["ts"], date_format="%Y/%m/%d %H:%M",
    )
    df = df.rename(columns={"amt_paid": "amt"})
    df["amt"] = df["amt"].round(AMT_DECIMALS)
    n_total = len(df)
    t0 = df["ts"].min()

    pat = parse_patterns(patterns_path)
    tidx, jstats = join_patterns(df, pat)
    att_of = np.full(n_total, -1, dtype=np.int64)
    type_of = np.full(n_total, "", dtype=object)
    ok = tidx.notna()
    att_of[tidx[ok].astype(int).to_numpy()] = pat.loc[ok, "attempt_id"].to_numpy()
    type_of[tidx[ok].astype(int).to_numpy()] = pat.loc[ok, "pattern"].to_numpy()
    laundering = df["y"].to_numpy() == 1
    pattern = np.where(att_of >= 0, type_of, np.where(laundering, "unattributed", "background"))

    src = (df["fb"] + "_" + df["fa"]).to_numpy()
    dst = (df["tb"] + "_" + df["ta"]).to_numpy()
    selfloop = src == dst
    keep = ~selfloop
    codes, uni = pd.factorize(np.concatenate([src[keep], dst[keep]]))
    m = int(keep.sum())
    tx = pd.DataFrame(
        {
            "timestamp": ((df["ts"][keep] - t0).dt.total_seconds() / 86400.0).to_numpy(),
            "src": codes[:m].astype(np.int32),
            "dst": codes[m:].astype(np.int32),
            "amount": df["amt"][keep].to_numpy(),
            "pattern_id": att_of[keep],
            "pattern": pattern[keep],
            "laundering": laundering[keep],
        }
    )
    a = tx[tx.pattern_id >= 0].groupby("pattern_id").agg(pattern=("pattern", "first"), n_rows=("src", "size"), start_day=("timestamp", "min"), end_day=("timestamp", "max"))
    attempts = a.reset_index(names="attempt_id")
    stats = {
        **jstats,
        "n_rows": n_total,
        "n_self_loops": int(selfloop.sum()),
        "n_self_loop_laundering": int((selfloop & laundering).sum()),
        "n_laundering": int(laundering.sum()),
        "n_unattributed_laundering": int(((pattern == "unattributed") & ~selfloop).sum()),
        "n_attempts": int(len(attempts)),
        "t_max_day": float(tx["timestamp"].max()),
    }
    log.info("%s: %s", name, stats)
    if stats["n_pattern_unmatched"]:
        log.warning("%s: %d pattern rows have no matching transaction", name, stats["n_pattern_unmatched"])
    if stats["n_duplicate_key_rows"]:
        log.info("%s: %d transactions share a join key with another; paired deterministically", name, stats["n_duplicate_key_rows"])
    return IBMData(name, tx, uni.astype(str), attempts, stats)


def load_ibm(name: str = "HI", root: str | Path | None = None) -> IBMData:
    root = Path(root or load_config()["data_root"])
    return load_ibm_files(root / f"{name}-Small_Trans.csv", root / f"{name}-Small_Patterns.txt", name)


def iter_ibm_windows(data: IBMData, length_days: float = 5.0, stride_days: float = 1.0):
    """Yield (window_id, start_day, end_day, tx) for full-length windows only (existing iter_windows)."""
    tmax = float(data.tx["timestamp"].max())
    for wid, (s, e, w) in enumerate(iter_windows(data.tx, length_days, stride_days, start=0.0, end=tmax - length_days + 1e-9)):
        yield wid, s, e, w
