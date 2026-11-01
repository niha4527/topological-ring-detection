"""Static checks for paper/ieee/main.tex (no LaTeX needed): cite keys in refs.bib, ref/label pairs,
includegraphics files, balanced braces and begin/end environments.

    python experiments/check_latex_static.py paper/ieee
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def strip_comments(t: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", t)


def main(d: Path) -> int:
    tex = strip_comments((d / "main.tex").read_text())
    bib = (d / "refs.bib").read_text()
    keys = set(re.findall(r"@\w+\{([^,\s]+),", bib))
    errors = []
    cites = [k.strip() for c in re.findall(r"\\cite\w*\{([^}]*)\}", tex) for k in c.split(",")]
    missing = sorted({k for k in cites if k not in keys})
    unused = sorted(keys - set(cites))
    print(f"cite commands: {len(cites)}, distinct keys cited: {len(set(cites))}, bib entries: {len(keys)}")
    if missing:
        errors.append(f"cite keys missing from refs.bib: {missing}")
    print("bib entries never cited:", unused or "none")
    labels = re.findall(r"\\label\{([^}]*)\}", tex)
    refs = re.findall(r"\\(?:ref|eqref)\{([^}]*)\}", tex)
    dup = sorted({l for l in labels if labels.count(l) > 1})
    unres = sorted({r for r in refs if r not in labels})
    print(f"labels: {len(labels)}, refs: {len(refs)}")
    if dup:
        errors.append(f"duplicate labels: {dup}")
    if unres:
        errors.append(f"unresolved refs: {unres}")
    print("labels never referenced:", sorted(set(labels) - set(refs)) or "none")
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", tex)
    for f in figs:
        if not (d / f).exists():
            errors.append(f"missing figure file: {f}")
    print(f"includegraphics: {len(figs)} ->", figs)
    body = re.sub(r"\\[\\{}%$&#_^~]", "", tex)
    if body.count("{") != body.count("}"):
        errors.append(f"unbalanced braces: {body.count('{')} open, {body.count('}')} close")
    envs = re.findall(r"\\(begin|end)\{([^}]*)\}", tex)
    stack = []
    for kind, name in envs:
        if kind == "begin":
            stack.append(name)
        elif not stack or stack.pop() != name:
            errors.append(f"environment mismatch at \\end{{{name}}}")
    if stack:
        errors.append(f"unclosed environments: {stack}")
    print(f"environments checked: {len(envs) // 2}")
    n_fig = len(re.findall(r"\\begin\{figure\*?\}", tex.split("\\appendices")[0]))
    n_tab = len(re.findall(r"\\begin\{table\*?\}", tex.split("\\appendices")[0]))
    print(f"main-text figures: {n_fig}, tables: {n_tab}")
    for ph in ("[VERIFY]", "[citation needed]", "[ASK]", "TODO"):
        if ph in tex:
            errors.append(f"placeholder left in text: {ph}")
    print("ERRORS:" if errors else "OK: no errors")
    for e in errors:
        print(" -", e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "paper/ieee")))
