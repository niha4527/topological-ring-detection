"""Numeric self-audit of paper/ieee/main.tex: every number must appear (as text) in paper/ieee/notes.md,
paper/numbers_pack.md or paper/numbers_pack/csv/*.csv. Signs are ignored; thousands separators are removed.

    python experiments/check_paper_numbers.py paper/ieee/main.tex
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

NUM = re.compile(r"(?<![A-Za-z_\\])\d[\d,{}]*(?:\.\d+)?(?:e-?\d+)?")


def norm(tok: str) -> str:
    return tok.replace("{,}", "").replace(",", "")


def numbers(text: str) -> list[str]:
    return [norm(m.group(0)).rstrip(".") for m in NUM.finditer(text)]


def main(tex_path: Path) -> None:
    tex = tex_path.read_text()
    body = re.sub(r"(?<!\\)%.*", "", tex)
    body = body.split("\\bibliographystyle")[0] + body.split("\\appendices")[-1] if "\\appendices" in body else body
    body = re.sub(r"\\(?:label|ref|cite\w*|includegraphics(?:\[[^\]]*\])?|bibliography|bibliographystyle|documentclass(?:\[[^\]]*\])?|usepackage(?:\[[^\]]*\])?)\{[^}]*\}", " ", body)
    body = re.sub(r"\\(?:section|subsection)\*?\{[^}]*\}", " ", body)
    body = body.replace("\\mi", "-")
    src = [Path("paper/ieee/notes.md"), Path("paper/numbers_pack.md")] + sorted(Path("paper/numbers_pack/csv").glob("*.csv"))
    allowed: set[str] = set()
    for p in src:
        allowed |= set(numbers(p.read_text()))
    found = numbers(body)
    unmatched = sorted({n for n in found if n not in allowed}, key=lambda s: (len(s), s))
    small = [n for n in unmatched if re.fullmatch(r"\d{1,2}", n)]
    other = [n for n in unmatched if n not in small]
    print(f"numbers in text: {len(found)} (distinct {len(set(found))}); matched: {len(found) - sum(n in unmatched for n in found)}; "
          f"unmatched occurrences: {sum(n in unmatched for n in found)}")
    print("unmatched small integers (counts, years, section numbers, list items):", small)
    print("unmatched other:", other)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
