#!/usr/bin/env python
"""Render example duplicate pairs from results/leakage_report.json as
side-by-side thumbnails so thresholds and claims can be eyeballed.

Defaults to the hardest threshold case (pHash 4 and 6 for
internal_train vs internal_test); accepts arguments to target any pair.

Usage:
  .venv/Scripts/python scripts/thumb_boundary.py
  .venv/Scripts/python scripts/thumb_boundary.py kaggle_testing__brisc_test exact 2
  .venv/Scripts/python scripts/thumb_boundary.py internal_train__kaggle_testing near 2
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "results" / "leakage_report.json"
OUT_DIR = ROOT / "results" / "leakage_thumbs"


def render(a_rel: str, b_rel: str, label: str, out_path: Path) -> None:
    height = 256
    panels = []
    for rel in (a_rel, b_rel):
        with Image.open(ROOT / rel) as img:
            img = img.convert("RGB")
            width = max(1, int(img.width * height / img.height))
            panels.append(img.resize((width, height), Image.Resampling.BILINEAR))
    canvas = Image.new(
        "RGB", (sum(p.width for p in panels) + 6, height + 22), "white"
    )
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width + 6
    ImageDraw.Draw(canvas).text((4, height + 5), label[:180], fill="black")
    canvas.save(out_path, quality=88)


def pick_entries(pair: dict, kind: str, dist: str | None) -> list[dict]:
    if kind == "exact":
        return pair.get("exact_examples", [])
    entries = pair.get("near_examples", [])
    if dist is not None:
        entries = [e for e in entries if str(e.get("phash")) == dist]
    return entries


def main(argv: list[str]) -> int:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if len(argv) >= 2:
        # targeted mode: pair_key kind [phash_distance]
        pair_key, kind = argv[0], argv[1]
        dist = argv[2] if len(argv) > 2 else None
        pair = report["pairs"][pair_key]
        entries = pick_entries(pair, kind, dist)
        if not entries:
            print(f"no {kind} examples" + (f" at phash={dist}" if dist else ""))
            return 1
        suffix = f"ph{dist}" if kind == "near" and dist else kind
        for index, entry in enumerate(entries[:4], start=1):
            out = OUT_DIR / f"check_{pair_key}_{suffix}_{index}.jpg"
            label = (
                f"{pair_key} [{kind}] {entry.get('phash', '')} "
                f"{entry['a']} | {entry['b']}"
            )
            render(entry["a"], entry["b"], label, out)
            print(f"wrote {out.relative_to(ROOT)}")
        return 0

    # default mode: boundary distances 4 and 6, internal vs internal
    pair = report["pairs"]["internal_train__internal_test"]
    wanted = {"4": None, "6": None}
    for entry in pair["near_examples"]:
        d = str(entry.get("phash"))
        if d in wanted and wanted[d] is None:
            wanted[d] = entry
    rendered = 0
    for dist, entry in wanted.items():
        if entry is None:
            print(f"no phash-{dist} example in near_examples")
            continue
        out = OUT_DIR / f"boundary_ph{dist}.jpg"
        label = (
            f"boundary ph{dist} internal_train vs internal_test: "
            f"{entry['a']} | {entry['b']}"
        )
        render(entry["a"], entry["b"], label, out)
        print(f"wrote {out.relative_to(ROOT)}")
        rendered += 1
    return 0 if rendered else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
