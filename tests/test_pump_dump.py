import numpy as np
import pandas as pd

from ringtda.data.pump_dump import EVENT_COLUMNS, PumpDumpConfig, generate
from ringtda.motif.pump_dump import BPD_NAMES, M_NAMES, accumulation_candidates, label_candidate, motif_features, pump_dump_table

SMALL = dict(n_accounts=120, duration_h=500.0, n_pump=2, n_organic=2, n_news=1, group_size=(4, 5), organic_exit_window_h=96.0, bg_amount_logmean=3.0, bg_amount_logsd=0.8)
CAND = dict(band_h=12.0, stride_h=6.0, acc_min=300.0, m_min=4, jaccard=0.8, horizon_h=120.0)


def test_generator_is_deterministic_and_plants_disjoint_groups():
    a = generate(PumpDumpConfig(seed=3, **SMALL))
    b = generate(PumpDumpConfig(seed=3, **SMALL))
    assert a[0].equals(b[0]) and a[1].equals(b[1])
    accounts = [x for g in a[2] for x in g.accounts]
    assert len(accounts) == len(set(accounts)) and {g.kind for g in a[2]} == {"pump_dump", "organic", "news"}


def test_planted_pump_exits_are_synchronised_and_organic_exits_are_not():
    ev, _, planted = generate(PumpDumpConfig(seed=1, **SMALL))
    spread = {}
    for g in planted:
        s = ev[(ev.pattern_id == g.pattern_id) & (ev.side == -1)].timestamp
        spread.setdefault(g.kind, []).append(float(s.std()))
    assert max(spread["pump_dump"]) < min(spread["organic"])


def test_candidates_find_planted_groups_without_labels():
    ev, price, planted = generate(PumpDumpConfig(seed=2, **SMALL))
    cands = accumulation_candidates(ev[EVENT_COLUMNS], 500.0, CAND["band_h"], CAND["stride_h"], CAND["acc_min"], CAND["m_min"], CAND["jaccard"])
    found = {label_candidate(c["accounts"], planted)[1] for c in cands}
    assert {g.pattern_id for g in planted} <= found  # every planted group yields a candidate


def test_features_separate_a_synchronised_exit_from_a_spread_one():
    ev, price, planted = generate(PumpDumpConfig(seed=4, **SMALL))
    f = {}
    for g in planted:
        f[g.kind] = motif_features(ev[EVENT_COLUMNS], price, g.accounts, g.t_start, g.t_end)
    assert f["pump_dump"]["M_exit_spread_h"] < f["organic"]["M_exit_spread_h"]
    assert f["pump_dump"]["M_exit_frac_24h"] > f["organic"]["M_exit_frac_24h"]
    assert f["pump_dump"]["M_runup"] > 0 and all(np.isfinite(v) for v in f["pump_dump"].values())


def test_leakage_guard_features_ignore_labels_and_ids():
    ev, price, planted = generate(PumpDumpConfig(seed=5, **SMALL))
    g = planted[0]
    base = motif_features(ev[EVENT_COLUMNS], price, g.accounts, g.t_start, g.t_end)
    scrambled = ev.copy()
    scrambled["pattern_id"] = 7
    scrambled["kind"] = "x"
    assert motif_features(scrambled, price, g.accounts, g.t_start, g.t_end) == base  # only EVENT_COLUMNS are read
    bad = ("label", "kind", "pattern", "planted", "group", "scenario", "attempt")
    assert not any(b in n.lower() for n in M_NAMES + BPD_NAMES for b in bad)
    assert not set(M_NAMES) & set(BPD_NAMES) - {"M_size"}


def test_table_has_labels_groups_and_seed_prefixed_ids():
    a = pump_dump_table(PumpDumpConfig(seed=0, **SMALL), CAND)
    b = pump_dump_table(PumpDumpConfig(seed=1, **SMALL), CAND, group_offset=10_000)
    assert (a.y == 1).any() and a.group.max() < 10_000 <= b.group.min()
    assert set(a.loc[a.y == 1, "kind"]) == {"pump_dump"}
