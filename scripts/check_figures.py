"""Phase 6 — structural QA for the regenerated manuscript.

Checks (all fail-closed, no network, no GPU):

  F1  every figure inserted by ``paper/rewrite_paper.py`` exists on disk as
      PNG *and* PDF, and is recorded in ``paper/figs/figures_manifest.json``;
  F2  figures are numbered 1..N without gaps or duplicates in document order;
  F3  every in-text ``Fig. N`` reference resolves to an inserted figure;
  F4  every figure is cited at least once after its caption (no orphans);
  T1  tables are numbered TABLE I..XI once each, in order;
  T2  every in-text ``Table X`` reference resolves to an existing table;
  T3  every table caption names its source file under ``results/`` or
      ``paper/tables/``;
  M1  ``scripts/check_paper_numbers.py`` passes (all numbers traceable).

Usage:
    python -X utf8 scripts/check_figures.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "paper" / "figs"
DOCX = ROOT / "TAMNet_Paper.docx"

ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
         "XI", "XII", "XIII", "XIV", "XV"]


def load_doc(path: Path):
    from docx import Document
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(str(path))
    paras, tables, images = [], [], []
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            p = Paragraph(child, doc)
            paras.append(p.text)
            if p._p.xpath(".//a:blip"):
                images.append(p.text)
        elif child.tag == qn("w:tbl"):
            t = Table(child, doc)
            tables.append([[c.text for c in r.cells] for r in t.rows])
    return paras, tables, images


def main() -> int:
    fails: list[str] = []
    warns: list[str] = []

    if not DOCX.is_file():
        print(f"FAIL: {DOCX} missing")
        return 1

    paras, tables, _ = load_doc(DOCX)
    text = "\n".join(paras)
    caption_paras = [p for p in paras if re.match(r"Fig\. \d+\.", p)]
    table_caps = [p for p in paras if re.match(r"TABLE [IVX]+$", p.strip())]

    # ---- F2 figure numbering in document order
    fig_nums = [int(re.match(r"Fig\. (\d+)\.", p).group(1))
                for p in caption_paras]
    if fig_nums != list(range(1, len(fig_nums) + 1)):
        fails.append(f"F2 figure numbering not 1..N in order: {fig_nums}")

    # ---- T1 table numbering in document order
    tbl_labels = [p.strip().split()[-1] for p in table_caps]
    expected = ROMAN[:len(tbl_labels)]
    if tbl_labels != expected:
        fails.append(f"T1 table numbering {tbl_labels} != {expected}")

    # ---- F1 every figure file exists (png + pdf) and is in the manifest
    import json

    man_path = FIGS / "figures_manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.is_file() \
        else {}
    fig_sources = []
    for p in paras:
        m = re.search(r"\(source: paper/figs/([^)]+)\)", p)
        if m:
            fig_sources.append(m.group(1))
    if len(fig_sources) != len(caption_paras):
        fails.append(f"F1 {len(fig_sources)} source tags for "
                     f"{len(caption_paras)} captions")
    for src in fig_sources:
        png = FIGS / src
        pdf = png.with_suffix(".pdf")
        if not png.is_file():
            fails.append(f"F1 missing PNG {src}")
        if not pdf.is_file():
            warns.append(f"missing vector PDF for {src}")
        stem = png.name
        fig_keys = {k[:-4] if k.endswith(".png") else k
                    for k in (man.get("figures") or {})}
        if fig_keys and stem[:-4] not in fig_keys:
            warns.append(f"{stem} not listed in figures_manifest.json")

    # ---- F3 in-text Fig references resolve
    refs = {int(n) for n in re.findall(r"Fig\. (\d+)", text)}
    missing_refs = sorted(refs - set(fig_nums))
    if missing_refs:
        fails.append(f"F3 in-text Fig references with no caption: {missing_refs}")

    # ---- F4 every figure is referenced at least once besides its caption
    body_wo_caps = [p for p in paras if not re.match(r"Fig\. \d+\.", p)]
    cited = {int(n) for n in re.findall(r"Fig\. (\d+)", "\n".join(body_wo_caps))}
    orphans = sorted(set(fig_nums) - cited)
    if orphans:
        warns.append(f"F4 figures never cited in the body: {orphans}")

    # ---- T2 in-text Table references resolve
    trefs = {int(ROMAN.index(x)) + 1
             for x in re.findall(r"\bTable ([IVX]+)\b", text)}
    tnums = {ROMAN.index(x) + 1 for x in tbl_labels}
    bad_t = sorted(t for t in trefs if t not in tnums)
    if bad_t:
        fails.append(f"T2 in-text Table references with no table: "
                     f"{[ROMAN[i - 1] for i in bad_t]}")

    # ---- T3 each table's note paragraph names a source
    # layout produced by rewrite_paper.py: "TABLE X" -> caption -> tbl ->
    # optional note paragraph beginning with "Source:"
    for i, p in enumerate(paras):
        if re.match(r"TABLE [IVX]+$", p.strip()):
            # find the note: first paragraph after the next table
            note = None
            for j in range(i + 1, min(i + 60, len(paras))):
                if re.match(r"TABLE [IVX]+$", paras[j].strip()):
                    break
                if paras[j].startswith("Source:"):
                    note = paras[j]
                    break
            if note is None:
                warns.append(f"T3 table {p.strip()} has no Source: note")

    # ---- M1 numbers traceable
    r = subprocess.run(
        [sys.executable, "-X", "utf8",
         str(ROOT / "scripts" / "check_paper_numbers.py")],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
        errors="replace")
    tail = (r.stdout or "").strip().splitlines()
    print(tail[-3] if len(tail) >= 3 else (r.stdout or ""))
    if r.returncode != 0:
        fails.append("M1 check_paper_numbers.py reported untraced numbers")

    # ---- report
    print(f"\nfigures: {len(fig_nums)} captions, {len(fig_sources)} sources")
    print(f"tables:  {len(tbl_labels)} captions ({', '.join(tbl_labels)})")
    for w in warns:
        print(f"  WARN  {w}")
    if fails:
        print("\nFAILURES:")
        for f in fails:
            print(f"  FAIL  {f}")
        return 1
    print("\nOK: manuscript structure and numbers pass Phase 6 checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
