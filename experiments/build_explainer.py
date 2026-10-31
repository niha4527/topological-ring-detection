"""Assemble paper/project_explainer.md from paper/explainer/*.md and audit every number in it. Docs only; no experiments.

    python experiments/build_explainer.py [--no-docx]

Placeholders in the part files:  <<TABLE id|k>>  copies the k-th markdown table under the pack heading whose text starts with `id`
(e.g. 3.2, 6A);  <<BLOCK id>>  copies the whole section body. Both read paper/numbers_pack.md, so the tables are the pack's, unchanged.

Audit: every number in the prose (outside code, URLs, figure paths and section/figure references) must appear in paper/numbers_pack.md or a
result file (results/**/*.csv|json|yaml|md, configs/*.yaml); decimals match if some source value rounds to them. Numbers found only in the
project documents (README.md, CLAUDE.md, architecture.md, prd.md, paper/paper_content.md) are listed separately; the rest are unmatched.
"""

import argparse
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--no-docx", action="store_true")
args = ap.parse_args()
SRC = Path("paper/explainer")
PACK = Path("paper/numbers_pack.md").read_text().split("\n")
ORDER = ["01_summary", "02_problem", "03_existing", "04_concept_a", "05_data", "06_impl_a", "06_impl_b", "07_eval", "08a_results", "08b_results", "09_conclusions", "11_limits", "12_story", "13_repro", "14_glossary"]


def sections():
    """[(level, heading text, start, end)] of the pack."""
    heads = [(i, len(m.group(1)), m.group(2)) for i, l in enumerate(PACK) if (m := re.match(r"^(#+) (.*)$", l))]
    out = []
    for k, (i, lev, txt) in enumerate(heads):
        end = next((j for j, l2, _ in heads[k + 1 :] if l2 <= lev), len(PACK))
        out.append((lev, txt, i, end))
    return out


SEC = sections()


def find(sid):
    for lev, txt, i, end in SEC:
        if re.match(rf"^{re.escape(sid)}(?:[ .:]|$)", txt):
            return i, end
    raise KeyError(f"pack section {sid!r} not found")


def body_until_any_heading(i):
    j = i + 1
    while j < len(PACK) and not PACK[j].startswith("#"):
        j += 1
    return PACK[i + 1 : j]


def table(sid, k):
    i, _ = find(sid)
    n, cur, intbl = 0, [], False
    for l in body_until_any_heading(i):
        if l.startswith("|"):
            if not intbl:
                n, intbl = n + 1, True
            if n == k:
                cur.append(l)
        else:
            intbl = False
    if not cur:
        raise KeyError(f"table {k} of {sid}")
    return "\n".join(cur) + f"\n\n*Source: `paper/numbers_pack.md` section {sid}.*\n"


def block(sid):
    i, end = find(sid)
    return "\n".join(PACK[i + 1 : end]).strip() + f"\n\n*Source: `paper/numbers_pack.md` section {sid}.*\n"


def expand(text):
    text = re.sub(r"<<TABLE ([^|>]+)\|(\d+)>>", lambda m: table(m.group(1).strip(), int(m.group(2))), text)
    return re.sub(r"<<BLOCK ([^>]+)>>", lambda m: block(m.group(1).strip()), text)


parts = [expand((SRC / f"{n}.md").read_text()) for n in ORDER]
doc = "\n".join(parts)
Path("paper/project_explainer.md").write_text(doc)

# ---------------------------------------------------------------- word counts per top-level section
words = {}
cur = "front"
for l in doc.split("\n"):
    m = re.match(r"^(?:# |## )(\d+)\. ?(.*)$", l) or re.match(r"^## (\d+)\. (.*)$", l)
    if m and l.startswith("## "):
        cur = f"{m.group(1)}. {m.group(2)[:50]}"
    words[cur] = words.get(cur, 0) + len(l.split())
total = sum(words.values())
prose_words = sum(len(l.split()) for l in doc.split("\n") if not l.startswith("|") and not l.startswith("*Source:") and not l.startswith("!["))

# ---------------------------------------------------------------- number audit
NUM = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")


def strip_prose(t):
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"arXiv[: ]\s*\d{4}\.\d{4,5}", " ", t)
    t = re.sub(r"DOI\s*10\.\d+/\d+", " ", t)
    t = re.sub(r"\b\d+\(\d+\),? \d+-\d+\b|\b\d+, 20\d\d\b", " ", t)  # volume(issue), pages in references
    t = re.sub(r"`[^`\n]*`", " ", t)
    t = re.sub(r"\*Source:[^\n]*", " ", t)
    t = re.sub(r"(?i)\b(?:sections?|figures?|fig\.|tables?|§)\s*\d+(?:\.\d+)*(?:\s*(?:to|and|,)\s*\d+(?:\.\d+)*)*", " ", t)
    t = re.sub(r"^#+ .*$", " ", t, flags=re.M)
    t = re.sub(r"\b[0-9a-f]{7}\b", " ", t)  # commit hashes
    t = re.sub(r"\b20\d\d-\d\d-\d\d\b|\bF\d+\b|\bT\d+\b|\bG\d\b|\bFR\d+\b|\bH[01]\b", " ", t)
    return t


def num_vals(text):
    return [float(x.replace(",", "").replace("+", "")) for x in NUM.findall(text) if re.search(r"\d", x)]


TOK = set()


def corpus(paths, tok=None):
    vals = []
    for p in paths:
        try:
            if p.stat().st_size > 60_000_000:
                continue
            txt = p.read_text(errors="ignore")
            vals += num_vals(txt)
            if tok is not None:
                tok.update(x.replace(",", "").replace("+", "").lstrip("-") for x in NUM.findall(txt))
        except Exception:
            pass
    return np.array(sorted(set(vals)))


res_files = [p for p in Path("results").rglob("*") if p.suffix in (".csv", ".json", ".yaml", ".md") and "cache" not in p.parts]
res_files += list(Path("configs").glob("*.yaml")) + [Path("paper/numbers_pack.md")]
C_main = corpus(res_files, TOK)
C_docs = corpus([Path(x) for x in ("README.md", "CLAUDE.md", "architecture.md", "prd.md", "paper/paper_content.md", "src/ringtda/data/synthetic.py")])


def present(C, x, d):
    tol = 0.5 * 10 ** (-d) + 1e-12 if d > 0 else 0.5
    i = np.searchsorted(C, x - tol)
    return i < len(C) and C[i] <= x + tol


prose = strip_prose(doc)
seen, unmatched, docs_only, trivial, rounded_only = 0, [], [], 0, []
for m in NUM.finditer(prose):
    raw = m.group(0)
    s = raw.replace(",", "").replace("+", "")
    try:
        x = float(s)
    except ValueError:
        continue
    d = len(s.split(".")[1]) if "." in s else 0
    if d == 0 and abs(x) <= 20:
        trivial += 1
        continue
    seen += 1
    ax = abs(x)
    if s.lstrip("-") in TOK:
        continue
    if present(C_main, ax, d):
        rounded_only.append((raw, prose[max(0, m.start() - 45) : m.end() + 25].replace("\n", " ")))
        continue
    ctx = prose[max(0, m.start() - 45) : m.end() + 25].replace("\n", " ")
    (docs_only if present(C_docs, ax, d) else unmatched).append((raw, ctx))

rep = [f"# Explainer audit\n", f"Words: {total} including the copied pack tables; {prose_words} excluding table rows, figure lines and source lines. Numbers checked (outside code, URLs, references; integers up to 20 skipped: {trivial}): {seen}. "
       f"Matched exactly as a token in the pack or a result file/config: {seen - len(docs_only) - len(unmatched) - len(rounded_only)}; matched only after rounding a source value: {len(rounded_only)}. Found only in project documents: {len(docs_only)}. Unmatched: {len(unmatched)}.\n", "## Words per section\n"]
rep += [f"- {k}: {v}" for k, v in words.items()]
rep += ["\n## Numbers found only in project documents (README, CLAUDE.md, architecture.md, prd.md, paper_content.md, the generator's defaults in src/ringtda/data/synthetic.py)\n"] + [f"- `{a}` in: ...{b}..." for a, b in docs_only]
rep += ["\n## Numbers matched only after rounding a source value (review list)\n"] + [f"- `{a}` in: ...{b}..." for a, b in rounded_only]
rep += ["\n## Unmatched numbers\n"] + ([f"- `{a}` in: ...{b}..." for a, b in unmatched] or ["none"])
(SRC / "audit_report.md").write_text("\n".join(rep) + "\n")
print(f"words {total} (prose only {prose_words}); numbers {seen}; rounded-only {len(rounded_only)}; docs-only {len(docs_only)}; unmatched {len(unmatched)}")
for a, b in unmatched:
    print("  UNMATCHED", a, "|", b)

if not args.no_docx:
    if shutil.which("pandoc"):
        r = subprocess.run(["pandoc", "paper/project_explainer.md", "-o", "paper/project_explainer.docx", "--resource-path=paper", "--toc"], capture_output=True, text=True)
        print("docx", "written" if r.returncode == 0 else f"FAILED {r.stderr[:200]}")
    else:
        print("pandoc not found: no .docx")
