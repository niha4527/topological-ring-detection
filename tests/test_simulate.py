import pandas as pd
import pytest

from ringtda.data.synthetic import SyntheticConfig
from ringtda.simulate import merge_window, parse_transactions, simulate, train
from ringtda.synthetic_table import synthetic_candidate_table

RING = "\n".join(f"{4 * r + 0.1 * h:.1f} {a} {b} 1000" for r in range(40) for h, (a, b) in enumerate([("A", "B"), ("B", "C"), ("C", "A")]))
CHAIN = "\n".join(f"{4 * r + 0.1 * h:.1f} {a} {b} 1000" for r in range(40) for h, (a, b) in enumerate([("A", "B"), ("B", "C"), ("C", "D")]))


def test_parse_ok_and_comments():
    df = parse_transactions("# header\n1.5 A B 10\n2, B, C, 5.5  # trailing\n")
    assert list(df["from"]) == ["A", "B"] and list(df["amount"]) == [10.0, 5.5]


@pytest.mark.parametrize("bad", ["", "1 A B", "x A B 1", "1 A A 1", "1 A B -3", "200 A B 1"])
def test_parse_rejects(bad):
    with pytest.raises(ValueError):
        parse_transactions(bad)


def test_merge_keeps_user_accounts_apart():
    tx, ids = merge_window(parse_transactions("1 A B 10"))
    assert min(ids.values()) >= 100_000 and tx.src.max() >= 100_000 and (tx.pattern == "user").sum() == 1


@pytest.fixture(scope="module")
def models():
    cfg = dict(n_accounts=200, n_background=1500, rings_per_size=3, dense_ring_sizes=(), n_chorded_rings=0, hard_negatives=False)
    df = pd.concat([synthetic_candidate_table(SyntheticConfig(seed=s, **cfg), group_offset=s * 1_000_000) for s in range(3)], ignore_index=True)
    assert df.y.sum() >= 10
    return {"B+D": train(df, "B+D")}


def test_ring_is_a_candidate_with_open_loop_and_steps(models):
    out = simulate(RING, models, "B+D")
    top = out["rings"][0]
    assert top["nodes"] == ["A", "B", "C"] and top["is_pure_user"]
    assert top["h1"] and top["h1"][0][1] is None  # infinite H1 bar: the loop is never filled
    assert {"pairs", "sparsify", "candidates"} <= set(out["steps"])
    # score = intercept + sum of per-feature contributions is exact for logistic regression
    m = models["B+D"]
    assert abs(top["intercept"] + sum(top["contrib_groups"].values()) - top["score"]) < 1e-2


def test_one_way_chain_forms_no_candidate(models):
    out = simulate(CHAIN, models, "B+D")
    assert out["rings"] == [] and out["verdict"]["flagged"] is False and "no candidate ring" in out["verdict"]["reason"]
