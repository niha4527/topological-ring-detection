"""Demo backend: thin FastAPI wrapper over ringtda. No pipeline logic lives here.

Run from the repo root:  python -m uvicorn demo.server:app --port 8000
"""

from __future__ import annotations

import math
import random
import uuid
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ringtda.data import toy
from ringtda.data.synthetic import SyntheticConfig, generate
from ringtda.eval.labels import label_candidates
from ringtda.features import window_graphs
from ringtda.simulate import FEATURE_SETS, load_models, simulate
from ringtda.topology.candidates import candidate_rings, neighborhood
from ringtda.topology.persistence import directed_flag_persistence
from ringtda.topology.vectorize import ring_survives

ROOT = Path(__file__).resolve().parent.parent
PACK = ROOT / "paper" / "numbers_pack"
TRAIN_CACHE = ROOT / "results" / "phase3_features" / "classify" / "synthetic" / "cache"
MODEL_PKL = ROOT / "demo" / ".cache" / "models.pkl"
app = FastAPI(title="ringtda demo")

TOYS = {
    "cyclic_triangle": ("Cyclic triangle", "A→B→C→A. A loop. Never filled, so one H1 bar that never dies.", toy.cyclic_triangle),
    "transitive_triangle": ("Transitive triangle", "A→B, B→C, A→C. One-way flow. Filled at once, so no H1 bar.", toy.transitive_triangle),
    "reciprocal_pair": ("Reciprocal pair", "A⇄B ping-pong. Two directed edges, one H1 bar.", toy.reciprocal_pair),
    "shortcut_filled": ("Shortcut-filled ring", "Cyclic triangle plus an account sending to all three. The loop is born at 0.1 and dies at 0.6.", toy.shortcut_filled_triangle),
    "square": ("Chordless 4-cycle", "Directed 4-cycle with no chord. One H1 bar.", toy.square),
}


def _num(x: float) -> float | None:
    return None if not math.isfinite(x) else float(x)


def _bars(G: nx.DiGraph) -> dict[str, list]:
    d = directed_flag_persistence(G)
    out = {}
    for dim in (0, 1):
        arr = np.asarray(d.get(dim, np.zeros((0, 2))), dtype=float).reshape(-1, 2)
        out[f"h{dim}"] = [[_num(b), _num(e)] for b, e in arr]
    return out


def _graph_json(G: nx.DiGraph) -> dict:
    return {
        "nodes": [str(n) for n in G.nodes],
        "edges": [{"src": str(u), "dst": str(v), "w": round(float(d["weight"]), 4)} for u, v, d in G.edges(data=True)],
    }


@app.get("/api/toys")
def toys():
    out = []
    for key, (title, text, fn) in TOYS.items():
        G = fn()
        out.append({"key": key, "title": title, "text": text, **_graph_json(G), "bars": _bars(G)})
    return out


class Edge(BaseModel):
    src: str
    dst: str
    w: float = Field(0.1, ge=0.0, le=1.0)


class PersistIn(BaseModel):
    edges: list[Edge] = Field(max_length=60)


@app.post("/api/persistence")
def persistence(body: PersistIn):
    G = nx.DiGraph()
    for e in body.edges:
        if e.src != e.dst:
            G.add_edge(e.src, e.dst, weight=e.w)
    if G.number_of_edges() == 0:
        return {"bars": {"h0": [], "h1": []}}
    return {"bars": _bars(G)}


RUNS: dict[str, dict] = {}


class RunIn(BaseModel):
    seed: int = 0
    tau_quantile: float = Field(0.25, gt=0.0, le=1.0)
    max_len: int = Field(5, ge=2, le=6)
    hard_negatives: bool = True


@app.post("/api/pipeline")
def pipeline(body: RunIn):
    cfg = SyntheticConfig(seed=body.seed, hard_negatives=body.hard_negatives)
    tx, planted = generate(cfg)
    pairs, G, Gs, tau = window_graphs(tx, body.tau_quantile)
    res = candidate_rings(G, tau=tau, max_len=body.max_len)
    labels = label_candidates(res.candidates, tx, planted)
    by_pid = {p.pattern_id: p for p in planted}

    rows = []
    for i, (c, kind, pid) in enumerate(zip(res.candidates, labels["kind"], labels["pattern_id"])):
        p = by_pid.get(int(pid))
        rows.append({"id": i, "nodes": [int(n) for n in c.nodes], "length": c.length, "birth": round(float(c.birth), 4),
                     "kind": kind, "scenario": p.scenario if p else None})
    rows.sort(key=lambda r: r["birth"])  # most suspicious (lowest birth) first

    n_pos_total = sum(r["kind"] == "wash_ring" for r in rows)
    hit, curve = 0, []
    for rank, r in enumerate(rows, 1):
        hit += r["kind"] == "wash_ring"
        curve.append([rank, hit / rank, hit / n_pos_total if n_pos_total else 0.0])
    step = max(1, len(curve) // 150)

    run_id = uuid.uuid4().hex[:8]
    if len(RUNS) > 20:
        RUNS.pop(next(iter(RUNS)))
    RUNS[run_id] = {"G": G, "Gs": Gs, "cands": res.candidates}
    return {
        "run_id": run_id,
        "summary": {
            "transactions": len(tx), "edges": G.number_of_edges(), "accounts": G.number_of_nodes(),
            "planted_total": len(planted), "planted_wash_rings": sum(p.is_positive for p in planted),
            "tau": round(tau, 4), "sccs": res.n_sccs, "candidates": len(rows),
            "by_kind": labels["kind"].value_counts().to_dict(),
            "planted_by_pattern": pd.Series([p.pattern for p in planted]).value_counts().to_dict(),
        },
        "candidates": rows[:300],
        "curve": curve[::step] + ([curve[-1]] if curve and (len(curve) - 1) % step else []),
    }


@app.get("/api/pipeline/{run_id}/candidate/{cid}")
def candidate(run_id: str, cid: int):
    run = RUNS.get(run_id)
    if run is None or not 0 <= cid < len(run["cands"]):
        raise HTTPException(404, "unknown run or candidate; rerun the pipeline")
    c = run["cands"][cid]
    H = neighborhood(run["Gs"], c.nodes, hops=1)  # same local graph the legacy T path uses
    surv = ring_survives(H, c.edges)
    return {
        "ring": [str(n) for n in c.nodes],
        "ring_edges": [[str(u), str(v)] for u, v in c.edges],
        "birth": round(float(c.birth), 4),
        "graph": _graph_json(H),
        "bars": _bars(H),
        "open_loop": None if surv is None else bool(surv),
    }


_MODELS: dict = {}


def models():
    if not _MODELS:
        _MODELS.update(load_models(TRAIN_CACHE, MODEL_PKL))
    return _MODELS


def _rounds(cycle, every_h=4.0, hop_h=0.1, amount=1000.0, jitter=0.01, time_noise=0.0, decoys=None, rng=None):
    rng = rng or random.Random(1)
    rows = []
    for r in range(int(168 // every_h)):
        t0 = r * every_h + rng.uniform(0, time_noise)
        for h, (a, b) in enumerate(zip(cycle, cycle[1:] + cycle[:1])):
            amt = amount * (1 + rng.uniform(-jitter, jitter))
            rows.append((t0 + h * hop_h, a, b, amt))
            if decoys:
                rows.append((t0 + h * hop_h + 0.05, a, rng.choice(decoys), amt * 0.3))
    return rows


def _text(rows):
    return "\n".join(f"{t:.2f} {a} {b} {x:.2f}" for t, a, b, x in sorted(rows))


def presets():
    rng = random.Random(7)
    arb = [(t, a, b, x * (1 - 0.03 * i)) for t, a, b, x in _rounds(["P", "Q", "R"], every_h=6, time_noise=4.0, jitter=0.2, rng=rng) for i in [0]]
    return [
        {"key": "wash3", "title": "Wash ring (3 accounts)", "text": "# A pays B, B pays C, C pays A, every 4h with near-equal amounts\n" + _text(_rounds(["A", "B", "C"]))},
        {"key": "wash2", "title": "Ping-pong pair", "text": "# A and B trade the same amount back and forth\n" + _text(_rounds(["A", "B"], every_h=3))},
        {"key": "wash5", "title": "Wash ring (5 accounts)", "text": "# five accounts cycling value\n" + _text(_rounds(["A", "B", "C", "D", "E"], every_h=6, hop_h=0.2))},
        {"key": "chain", "title": "One-way chain (no loop)", "text": "# A to B to C to D, never returns\n" + _text([(t, a, b, x) for t, a, b, x in _rounds(["A", "B", "C", "D"]) if (a, b) != ("D", "A")])},
        {"key": "irregular", "title": "Irregular loop (arbitrage-like)", "text": "# a loop, but irregular timing and sizes\n" + _text(arb)},
        {"key": "decoys", "title": "Wash ring with decoy trades", "text": "# the ring also sends 30% of each hop to outsiders (evasion)\n" + _text(_rounds(["A", "B", "C"], decoys=["X", "Y", "Z", "W"], rng=random.Random(3)))},
        {"key": "oneshot", "title": "One-off triangle (3 trades)", "text": "10 A B 500\n11 B C 500\n12 C A 500"},
    ]


class SimIn(BaseModel):
    text: str = Field(max_length=60000)
    feature_set: str = "B+D"
    seed: int = Field(0, ge=0, le=1000)


@app.get("/api/sim/presets")
def sim_presets():
    return {"presets": presets(), "feature_sets": list(FEATURE_SETS)}


STAB_SEEDS = (0, 1, 2, 3, 4)  # background weeks every input is scored against


def _shape(out: dict) -> str:
    """One plain-language line on what kind of structure the input has."""
    rings = out["rings"]
    if rings:
        top = rings[0]
        loop = " → ".join(top["nodes"]) + " → " + top["nodes"][0]
        word = "pair trading back and forth" if top["length"] == 2 else f"directed loop of {top['length']} accounts"
        return f"{word.capitalize()}: {loop}."
    g = nx.DiGraph([(p["from"], p["to"]) for p in out["steps"]["pairs"]])
    if nx.is_directed_acyclic_graph(g):
        return "One-way flow only: money never returns to where it started, so there is no loop."
    return "A loop exists, but its edges are not suspicious enough to pass the cutoff, so it is not a candidate ring."


@app.post("/api/sim")
def sim(body: SimIn):
    if body.feature_set not in FEATURE_SETS:
        raise HTTPException(400, f"feature_set must be one of {list(FEATURE_SETS)}")
    try:
        allruns = {s: {fs: simulate(body.text, models(), fs, s) for fs in FEATURE_SETS} for s in STAB_SEEDS}
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    runs = allruns[STAB_SEEDS[0]]
    out = runs[body.feature_set]
    # same input scored by every model: rings matched by their account cycle
    out["compare"] = [
        {
            "ring": ring["nodes"],
            "scores": {
                fs: next(({k: r[k] for k in ("score", "threshold", "flagged", "percentile")} for r in run["rings"] if r["nodes"] == ring["nodes"]), None)
                for fs, run in runs.items()
            },
        }
        for ring in out["rings"]
    ]
    out["shape"] = _shape(out)
    n = len(STAB_SEEDS)
    flagged = sum(allruns[s][body.feature_set]["verdict"]["flagged"] for s in STAB_SEEDS)
    out["stability"] = {"backgrounds": n, "flagged": int(flagged), "stable": flagged in (0, n)}
    return out


FIGURES = [
    ("hi_pr_auc.png", "IBM HI-Small: PR-AUC by feature set (B, B+D, B+D+T, T)"),
    ("synthetic_lift_per_scenario.png", "Synthetic: topology lift per scenario"),
    ("synthetic_pr_auc.png", "Synthetic: PR-AUC by feature set"),
    ("recall_ibm.png", "Candidate-generation recall, IBM"),
    ("recall_synthetic.png", "Candidate-generation recall, synthetic"),
    ("hi_audit_scores.png", "IBM pattern-behaviour audit (HI)"),
    ("cluster_pca_HI.png", "Clustering of IBM positive rings (HI)"),
    ("bandit_drift.png", "Adaptive threshold under drift"),
    ("fig1_gd_convergence.png", "Optimization report: gradient descent convergence"),
    ("fig4_precision_recall.png", "Optimization report: precision/recall"),
    ("pump_dump_pr_auc.png", "Synthetic pump-and-dump (no topology)"),
]


@app.get("/api/results")
def results():
    tables = {}
    for ds in ("HI", "LI"):
        f = PACK / "csv" / f"phase3_features__classify__{ds}_tau0.05__results.csv"
        if f.exists():
            df = pd.read_csv(f)
            df = df[df["variant"] == "primary"]
            tables[ds] = df[["feature_set", "model", "pr_auc", "ci_low", "ci_high", "prevalence", "n_pos_groups"]].round(4).to_dict("records")
    return {
        "figures": [{"file": f, "caption": c} for f, c in FIGURES if (PACK / "figures" / f).exists()],
        "classify": tables,
    }


app.mount("/figures", StaticFiles(directory=PACK / "figures"), name="figures")


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
