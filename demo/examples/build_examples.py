"""Builds demo/examples/SIMULATOR_EXAMPLES.md: Simulator inputs and what the model does with each.

Run from the project root: python -I demo/examples/build_examples.py
Each case is scored with the same models the Simulator uses (B+D and B+D+T_local), so the verdicts in the file are real.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from demo.server import _rounds, _text, models  # noqa: E402
from ringtda.simulate import FEATURE_SETS, simulate  # noqa: E402

R = random.Random


def chain(cycle, **kw):
    rows = _rounds(cycle, **kw)
    return [r for r in rows if (r[1], r[2]) != (cycle[-1], cycle[0])]


def two_cycles():
    return _rounds(["A", "B", "C"], every_h=4) + _rounds(["P", "Q", "R", "S"], every_h=6, hop_h=0.2, amount=2500)


def hub():
    rows = []
    for r in range(42):
        t = r * 4.0
        for i, x in enumerate("BCDE"):
            rows.append((t + i * 0.1, "A", x, 400.0))
            rows.append((t + 1 + i * 0.1, x, "A", 395.0))
    return rows


CASES = [
    ("Should flag", "Wash ring, 3 accounts, every 4h", "wash3", _text(_rounds(["A", "B", "C"]))),
    ("Should flag", "Wash ring, 4 accounts, every 6h", "wash4", _text(_rounds(["A", "B", "C", "D"], every_h=6, hop_h=0.2))),
    ("Should flag", "Wash ring, 5 accounts, every 6h", "wash5", _text(_rounds(["A", "B", "C", "D", "E"], every_h=6, hop_h=0.2))),
    ("Should flag", "Ping-pong pair, every 3h", "pingpong", _text(_rounds(["A", "B"], every_h=3))),
    ("Should flag", "Fast 3-ring, every 2h, large amounts", "fast3", _text(_rounds(["A", "B", "C"], every_h=2, amount=50000))),
    ("Should flag", "Ring with decoy trades to outsiders", "decoys", _text(_rounds(["A", "B", "C"], decoys=["X", "Y", "Z", "W"], rng=R(3)))),
    ("Should flag", "Two rings in one input", "two", _text(two_cycles())),
    ("Should not flag", "One-way chain, no loop", "chain", _text(chain(["A", "B", "C", "D"]))),
    ("Should not flag", "One-off triangle, 3 trades", "oneshot", "10 A B 500\n11 B C 500\n12 C A 500"),
    ("Should not flag", "Irregular loop, random timing and sizes", "irregular", _text(_rounds(["P", "Q", "R"], every_h=6, time_noise=4.0, jitter=0.2, rng=R(7)))),
    ("Edge case", "Hub that pays and is repaid by 4 accounts", "hub", _text(hub())),
    ("Should not flag", "Single payment", "single", "5 A B 100"),
    ("Should not flag", "Two-way trade, one round each", "tworound", "20 A B 1000\n90 B A 1000"),
    ("Should not flag", "Star: A pays four accounts, no returns", "star", "\n".join(f"{i * 10 + 1} A {x} 300" for i, x in enumerate("BCDE"))),
]


def main():
    M = models()
    out = ["# Simulator examples", "",
           "Paste a case into the Simulator box (tab 2) and press Score. The blocks here show only the first 12 lines; the verdicts are for the full input, which is in `demo/examples/inputs/<name>.txt` (copy from that file).",
           "Format per line: `hour from to amount`. Hour is 0 to 168 (one week). `#` starts a comment.", "",
           "Verdicts below were produced by running each block through the same models the Simulator uses. "
           "Scores are the model's ring score (higher = more suspicious); the cutoff is each model's own tuned threshold.", ""]
    summary = []
    for group, title, key, text in CASES:
        res = {fs: simulate(text, M, fs, 0) for fs in FEATURE_SETS}
        cells = []
        for fs in FEATURE_SETS:
            v = res[fs]["verdict"]
            sc = v.get("top_score")
            cells.append(f"{fs}: {'FLAGGED' if v['flagged'] else 'not flagged'}" + (f" (score {sc:.3f})" if sc is not None else f" ({v.get('reason', 'no candidate ring')})"))
        summary.append((group, title, key, text, cells))
        print(f"{group:16s} {key:10s} " + " | ".join(cells), flush=True)
    notes = {"Edge case": "These look like ordinary behaviour to a person but the model flags them. They are limits of the model, not bugs in the Simulator: repeated, near-equal round trips between the same accounts are exactly what it learned to call wash trading."}
    for grp in ("Should flag", "Should not flag", "Edge case"):
        out += [f"## {grp}", ""] + ([notes[grp], ""] if grp in notes else [])
        for g, title, key, text, cells in summary:
            if g != grp:
                continue
            body = text if len(text.splitlines()) <= 12 else "\n".join(text.splitlines()[:12]) + "\n# ... (continues; use the file in demo/examples/inputs/ for the full text)"
            out += [f"### {title}", "", "Model result: " + "; ".join(cells), "", "```", body, "```", ""]
    d = Path(__file__).parent
    (d / "inputs").mkdir(exist_ok=True)
    for g, title, key, text, cells in summary:
        (d / "inputs" / f"{key}.txt").write_text(f"# {title}\n" + text + "\n")
    (d / "SIMULATOR_EXAMPLES.md").write_text("\n".join(out) + "\n")


main()
