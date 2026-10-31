"""IBM AML loader, window iterator and ring labels on tiny fixtures (no IBM data needed)."""

import numpy as np
import pandas as pd

from ringtda.data.ibm_aml import iter_ibm_windows, load_ibm_files
from ringtda.eval.labels import label_candidates_ibm, unattributed_edges
from ringtda.topology.candidates import Candidate

HEADER = "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering\n"
TRANS = HEADER + "\n".join(
    [
        "2022/09/01 00:00,010,A1,010,B1,100.00,US Dollar,100.00,US Dollar,ACH,0",  # 0: background
        "2022/09/01 01:00,010,B1,020,C1,100.00,US Dollar,100.00,US Dollar,ACH,1",  # 1: attempt 0
        "2022/09/01 02:00,020,C1,010,A1,100.00,US Dollar,100.00,US Dollar,ACH,1",  # 2: attempt 0
        "2022/09/02 03:00,030,D1,030,D1,5.00,US Dollar,5.00,US Dollar,Reinvestment,0",  # 3: self-loop
        "2022/09/02 04:00,040,E1,040,F1,7.00,Euro,7.00,Euro,ACH,0",  # 4: duplicate key, not laundering
        "2022/09/02 04:00,040,E1,040,F1,7.00,Euro,7.00,Euro,ACH,1",  # 5: duplicate key, laundering
        "2022/09/03 05:00,050,G1,050,H1,9.00,Euro,9.00,Euro,ACH,1",  # 6: laundering, in no attempt
        "2022/09/08 06:00,010,A1,010,B1,100.00,US Dollar,100.00,US Dollar,ACH,0",  # 7: later background
    ]
)
PATTERNS = """BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 2 hops
2022/09/01 01:00,010,B1,020,C1,100.00,US Dollar,100.00,US Dollar,ACH,1
2022/09/01 02:00,020,C1,010,A1,100.00,US Dollar,100.00,US Dollar,ACH,1
END LAUNDERING ATTEMPT - CYCLE

BEGIN LAUNDERING ATTEMPT - SCATTER-GATHER
2022/09/02 04:00,040,E1,040,F1,7.00,Euro,7.00,Euro,ACH,1
END LAUNDERING ATTEMPT - SCATTER-GATHER
"""


def load(tmp_path):
    t, p = tmp_path / "T.csv", tmp_path / "P.txt"
    t.write_text(TRANS)
    p.write_text(PATTERNS)
    return load_ibm_files(t, p, "TOY")


def test_loader_joins_attempts_handles_duplicates_and_drops_self_loops(tmp_path):
    d = load(tmp_path)
    s = d.stats
    assert s["n_rows"] == 8 and s["n_self_loops"] == 1 and len(d.tx) == 7
    assert s["n_duplicate_key_rows"] == 2 and s["n_pattern_unmatched"] == 0
    assert s["n_unattributed_laundering"] == 1 and s["n_attempts"] == 2
    tx = d.tx
    by_pattern = tx.groupby("pattern").size().to_dict()
    assert by_pattern == {"CYCLE": 2, "SCATTER-GATHER": 1, "unattributed": 1, "background": 3}
    # the duplicate pair: the laundering twin takes the attempt, the non-laundering twin stays background
    dup = tx[(tx.timestamp > 1.1) & (tx.timestamp < 1.2)]
    assert sorted(zip(dup.laundering, dup.pattern)) == [(False, "background"), (True, "SCATTER-GATHER")]
    assert d.attempts.set_index("attempt_id").pattern.to_dict() == {0: "CYCLE", 1: "SCATTER-GATHER"}


def test_accounts_are_bank_account_pairs_and_time_is_days_from_start(tmp_path):
    d = load(tmp_path)
    assert d.tx.timestamp.min() == 0.0 and np.isclose(d.tx.timestamp.max(), 7 + 6 / 24)
    ids = set(d.tx.src) | set(d.tx.dst)
    assert {d.accounts[i] for i in ids} >= {"010_A1", "020_C1", "010_B1"}
    assert "030_D1" not in set(d.accounts)  # only appeared in a self-loop


def test_windows_are_full_length_only(tmp_path):
    d = load(tmp_path)
    w = list(iter_ibm_windows(d, length_days=5, stride_days=1))
    assert [round(s) for _, s, _, _ in w] == [0, 1, 2]  # t_max ~ 7.25 -> starts 0, 1, 2
    assert all(e - s == 5 for _, s, e, _ in w)
    assert len(w[0][3]) == 6 and len(w[2][3]) == 1


# ---- labels ----


def cand(nodes):
    n = len(nodes)
    return Candidate(tuple(nodes), tuple((nodes[i], nodes[(i + 1) % n]) for i in range(n)), 0.1)


def tx_rows(rows):
    return pd.DataFrame(rows, columns=["src", "dst", "pattern_id", "pattern", "laundering"]).assign(timestamp=0.0, amount=1.0)


def test_ibm_labels():
    rows = [(0, 1, 0, "CYCLE", True), (1, 2, 0, "CYCLE", True), (2, 0, 0, "CYCLE", True)]  # attempt 0: ring 0-1-2
    rows += [(10, 11, 0, "CYCLE", True), (11, 12, 1, "FAN-IN", True), (12, 10, 1, "FAN-IN", True)]  # mixed
    rows += [(20, 21, -1, "unattributed", True), (21, 22, -1, "unattributed", True), (22, 20, -1, "unattributed", True)]
    rows += [(30, 31, -1, "background", False), (31, 30, -1, "background", False)]
    tx = tx_rows(rows)
    attempts = {0: "CYCLE", 1: "FAN-IN"}
    lab = label_candidates_ibm([cand([0, 1, 2]), cand([10, 11, 12]), cand([20, 21, 22]), cand([30, 31]), cand([40, 41])], tx, attempts)
    assert list(lab.kind) == ["CYCLE", "ambiguous", "ambiguous", "background", "background"]
    assert list(lab.pattern_id) == [0, -1, -1, -1, -1]
    assert lab.y.tolist()[0] == 1.0 and np.isnan(lab.y[1]) and np.isnan(lab.y[2]) and lab.y[3] == 0.0
    assert unattributed_edges(tx) == {(20, 21), (21, 22), (22, 20)}


def test_ibm_label_80_percent_threshold():
    # 5-ring, 4 of 5 edges from attempt 0 (80%) -> positive; 3 of 5 -> ambiguous
    ring = [0, 1, 2, 3, 4]
    four = tx_rows([(i, (i + 1) % 5, 0, "CYCLE", True) for i in range(4)])
    three = tx_rows([(i, (i + 1) % 5, 0, "CYCLE", True) for i in range(3)])
    assert label_candidates_ibm([cand(ring)], four, {0: "CYCLE"}).kind[0] == "CYCLE"
    assert label_candidates_ibm([cand(ring)], three, {0: "CYCLE"}).kind[0] == "ambiguous"
    assert label_candidates_ibm([cand(ring)], four, {0: "CYCLE"}, x=1.0).kind[0] == "ambiguous"
