#!/usr/bin/env python
"""Phase 1 data-integrity: cross-dataset leakage audit for TAMNet_Paper.docx.

For every pair of the six classification sources used by the paper, measures:

  1. filename overlap            (case-insensitive basename match)
  2. exact content duplicates    (SHA-256 of raw bytes)
  3. near duplicates             (pHash Hamming <= 6 OR dHash Hamming <= 4,
                                  exact matches excluded)

Plus within-source duplicates, per-source class counts, and a cross-check of
BRISC files against the publisher's manifest.csv sha256 column.

Outputs (everything under results/, nothing under outputs/ is touched):
  results/leakage_report.json           counts, examples, embedded config
  results/leakage_thumbs/*.jpg          up to 20 side-by-side example pairs

Rules honoured:
  * no model metrics are computed here. "Metrics before/after de-duplication"
    requires re-evaluation on de-duplicated test sets -> flagged TODO-RERUN
    (Phase 3) inside the report.
  * every number in the JSON comes from this run; timestamp, git commit and
    config are embedded so the report is self-describing.

Usage:
  .venv/Scripts/python scripts/leakage_check.py [--workers 8] [--no-thumbs]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.fft import dctn

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
THUMBS_DIR = RESULTS_DIR / "leakage_thumbs"
REPORT_PATH = RESULTS_DIR / "leakage_report.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PHASH_MAX = 6          # hamming distance for a pHash near-duplicate
DHASH_MAX = 4          # hamming distance for a dHash near-duplicate
THUMB_MAX_PAIRS = 20   # side-by-side examples written to leakage_thumbs/
HAMMING_CHUNK = 1024   # rows per chunk when building the hamming matrix

SOURCES: dict[str, Path] = {
    "internal_train": ROOT / "data" / "raw" / "classification" / "Train",
    "internal_test": ROOT / "data" / "raw" / "classification" / "Test",
    "kaggle_training": ROOT / "data" / "external" / "kaggle_brain_tumor" / "Training",
    "kaggle_testing": ROOT / "data" / "external" / "kaggle_brain_tumor" / "Testing",
    "brisc_train": ROOT / "data" / "external" / "brisc2025" / "brisc2025" / "classification_task" / "train",
    "brisc_test": ROOT / "data" / "external" / "brisc2025" / "brisc2025" / "classification_task" / "test",
}

# publisher manifest root (for the sha256 cross-check)
BRISC_MANIFEST_ROOT = ROOT / "data" / "external" / "brisc2025" / "brisc2025"
BRISC_MANIFEST_CSV = BRISC_MANIFEST_ROOT / "manifest.csv"

# class-label aliases keyed by alnum-normalised directory name
CLASS_LABEL_ALIASES = {
    "glioma": "glioma",
    "meningioma": "meningioma",
    "pituitary": "pituitary",
    "pituitarytumor": "pituitary",
    "notumor": "no_tumor",
    "notumors": "no_tumor",
    "notumour": "no_tumor",
    "notumours": "no_tumor",
    "healthy": "no_tumor",
}


def label_of(path: Path) -> str | None:
    """Class label from the nearest class-named ancestor directory, if any."""
    for part in reversed(path.parts[:-1]):
        key = "".join(ch for ch in part.lower() if ch.isalnum())
        if key in CLASS_LABEL_ALIASES:
            return CLASS_LABEL_ALIASES[key]
    return None


# --------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------
def phash64(img: Image.Image) -> int:
    """64-bit perceptual hash (imagehash-style: 2D DCT, top-left 8x8, median)."""
    gray = img.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.float64)
    coeff = dctn(arr, type=2, norm="ortho")[:8, :8].flatten()
    bits = (coeff > np.median(coeff)).astype(np.uint8)
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def dhash64(img: Image.Image) -> int:
    """64-bit difference hash (9x8 grayscale, horizontal gradient signs)."""
    gray = img.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.uint8)
    bits = (arr[:, 1:] > arr[:, :-1]).flatten().astype(np.uint8)
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def digest(path: Path) -> dict:
    data = path.read_bytes()
    record = {
        "path": path,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "phash": None,
        "dhash": None,
        "error": None,
    }
    try:
        with Image.open(path) as img:
            img.load()
            record["phash"] = phash64(img)
            record["dhash"] = dhash64(img)
    except Exception as exc:  # corrupt / unreadable image: keep sha, flag it
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def collect_files(source_dir: Path) -> list[Path]:
    files = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if "masks" in {part.lower() for part in path.parts}:
            continue  # segmentation masks are not classification inputs
        files.append(path)
    return files


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def hamming_matches(
    a: np.ndarray, b: np.ndarray, max_distance: int
) -> list[tuple[int, int, int]]:
    """All (i, j, distance) with distance <= max_distance between uint64 arrays."""
    matches: list[tuple[int, int, int]] = []
    for start in range(0, len(a), HAMMING_CHUNK):
        block = a[start : start + HAMMING_CHUNK]
        distances = np.bitwise_count(block[:, None] ^ b[None, :])
        rows, cols = np.nonzero(distances <= max_distance)
        for row, col in zip(rows.tolist(), cols.tolist()):
            matches.append((start + row, col, int(distances[row, col])))
    return matches


def class_counts(files: list[Path], source_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in files:
        try:
            key = path.relative_to(source_dir).parts[0]
        except (ValueError, IndexError):
            key = "(flat)"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def git_metadata() -> dict:
    meta: dict[str, str | bool | None] = {"commit": None, "dirty": None}
    try:
        meta["commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        )
        meta["dirty"] = bool(status.strip())
    except Exception:
        pass
    return meta


def make_thumbnail(path_a: Path, path_b: Path, label: str, out_path: Path) -> None:
    """Side-by-side composite (left = A, right = B) with a caption strip."""
    height = 256
    panels = []
    for path in (path_a, path_b):
        with Image.open(path) as img:
            img = img.convert("RGB")
            width = max(1, int(img.width * height / img.height))
            panels.append(img.resize((width, height), Image.Resampling.BILINEAR))
    gap = 6
    caption = 22
    canvas = Image.new(
        "RGB", (sum(p.width for p in panels) + gap, height + caption), "white"
    )
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width + gap
    draw = ImageDraw.Draw(canvas)
    draw.text((4, height + 5), label[:180], fill="black")
    canvas.save(out_path, quality=88)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-thumbs", action="store_true")
    args = parser.parse_args()

    started = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. collect + digest every image -------------------------------
    sources: dict[str, dict] = {}
    all_tasks: list[tuple[str, Path]] = []
    for name, directory in SOURCES.items():
        if not directory.is_dir():
            print(f"  [missing] {name}: {directory}", file=sys.stderr)
            sources[name] = {"path": rel(directory), "n_images": 0, "missing": True}
            continue
        files = collect_files(directory)
        all_tasks.extend((name, path) for path in files)
        print(f"  {name}: {len(files)} images")

    print(f"digesting {len(all_tasks)} images with {args.workers} workers ...")
    digested: dict[str, list[dict]] = {name: [] for name in SOURCES}
    errors: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(digest, path): name for name, path in all_tasks}
        done = 0
        for future, name in futures.items():
            record = future.result()
            record["source"] = name
            digested[name].append(record)
            if record["error"]:
                errors.append({"path": rel(record["path"]), "error": record["error"]})
            done += 1
            if done % 2500 == 0:
                print(f"    {done}/{len(all_tasks)}")

    for name in SOURCES:
        digested[name].sort(key=lambda r: rel(r["path"]))

    # ---- 2. per-source statistics ---------------------------------------
    sha_index: dict[str, list[tuple[str, str]]] = {}   # sha -> [(source, relpath)]
    name_index: dict[str, list[tuple[str, str]]] = {}  # basename -> [...]
    for name, recs in digested.items():
        for record in recs:
            sha_index.setdefault(record["sha256"], []).append((name, rel(record["path"])))
            name_index.setdefault(Path(rel(record["path"])).name.lower(), []).append(
                (name, rel(record["path"]))
            )

    for name, directory in SOURCES.items():
        recs = digested[name]
        if not recs:
            continue
        sha_counts: dict[str, int] = {}
        for record in recs:
            sha_counts[record["sha256"]] = sha_counts.get(record["sha256"], 0) + 1
        sources[name] = {
            "path": rel(directory),
            "n_images": len(recs),
            "n_unique_content": len(sha_counts),
            "n_within_source_duplicates": sum(c - 1 for c in sha_counts.values() if c > 1),
            "n_unreadable": sum(1 for r in recs if r["error"]),
            "class_counts": class_counts([r["path"] for r in recs], directory),
            "n_te_prefix_files": sum(
                1 for r in recs if r["path"].name.lower().startswith("te-")
            ),
        }

    # ---- 3. pairwise analysis -------------------------------------------
    pairs: dict[str, dict] = {}
    thumb_candidates: list[dict] = []
    source_names = [n for n in SOURCES if digested.get(n)]

    for src_a, src_b in combinations(source_names, 2):
        recs_a, recs_b = digested[src_a], digested[src_b]
        key = f"{src_a}__{src_b}"

        # (a) filename overlap
        names_a = {Path(rel(r["path"])).name.lower() for r in recs_a}
        names_b = {Path(rel(r["path"])).name.lower() for r in recs_b}
        shared_names = sorted(names_a & names_b)

        # (b) exact content duplicates (sha groups spanning both sources)
        exact_pairs: list[dict] = []
        exact_a: set[int] = set()
        exact_b: set[int] = set()
        pos_a: dict[str, list[int]] = {}
        pos_b: dict[str, list[int]] = {}
        for i, record in enumerate(recs_a):
            pos_a.setdefault(record["sha256"], []).append(i)
        for j, record in enumerate(recs_b):
            pos_b.setdefault(record["sha256"], []).append(j)
        for sha in set(pos_a) & set(pos_b):
            for i in pos_a[sha]:
                exact_a.add(i)
                for j in pos_b[sha]:
                    exact_b.add(j)
                    exact_pairs.append(
                        {
                            "a": rel(recs_a[i]["path"]),
                            "b": rel(recs_b[j]["path"]),
                            "sha256": sha,
                        }
                    )

        # (c) near-duplicates, exact matches excluded.
        # Unreadable images have no hash: give them a path-derived sentinel so
        # two hashless files can never match each other by accident.
        def _hash_array(records: list[dict], field: str) -> np.ndarray:
            values = []
            for record in records:
                value = record[field]
                if value is None:
                    value = int(record["sha256"][:16], 16)
                values.append(value)
            return np.array(values, dtype=np.uint64)

        ph_a = _hash_array(recs_a, "phash")
        ph_b = _hash_array(recs_b, "phash")
        dh_a = _hash_array(recs_a, "dhash")
        dh_b = _hash_array(recs_b, "dhash")
        near: dict[tuple[int, int], dict] = {}
        for i, j, dist in hamming_matches(ph_a, ph_b, PHASH_MAX):
            near.setdefault((i, j), {"phash": dist})["phash"] = dist
        for i, j, dist in hamming_matches(dh_a, dh_b, DHASH_MAX):
            near.setdefault((i, j), {})["dhash"] = dist
        near_pairs = []
        for (i, j), dists in sorted(near.items()):
            if i in exact_a and j in exact_b and recs_a[i]["sha256"] == recs_b[j]["sha256"]:
                continue  # already counted as exact
            entry = {
                "a": rel(recs_a[i]["path"]),
                "b": rel(recs_b[j]["path"]),
                **{k: v for k, v in dists.items()},
            }
            near_pairs.append(entry)
            thumb_candidates.append(
                {
                    "pair": key,
                    "a_path": recs_a[i]["path"],
                    "b_path": recs_b[j]["path"],
                    "phash": dists.get("phash"),
                    "dhash": dists.get("dhash"),
                    "kind": "near",
                }
            )
        for entry in exact_pairs:
            thumb_candidates.append(
                {
                    "pair": key,
                    "a_path": ROOT / entry["a"],
                    "b_path": ROOT / entry["b"],
                    "phash": 0,
                    "dhash": 0,
                    "kind": "exact",
                }
            )

        # label agreement across exact duplicates: a duplicate whose two
        # copies sit under different class directories is a data-integrity bug
        label_agree = label_disagree = label_unknown = 0
        label_disagreement_examples: list[dict] = []
        for entry in exact_pairs:
            label_a = label_of(ROOT / entry["a"])
            label_b = label_of(ROOT / entry["b"])
            if label_a is None or label_b is None:
                label_unknown += 1
            elif label_a == label_b:
                label_agree += 1
            else:
                label_disagree += 1
                if len(label_disagreement_examples) < 10:
                    label_disagreement_examples.append(
                        {
                            "a": entry["a"],
                            "label_a": label_a,
                            "b": entry["b"],
                            "label_b": label_b,
                        }
                    )

        overlap_n = len(exact_pairs) + len(near_pairs)
        denominator = min(len(recs_a), len(recs_b))
        # unique images involved (a single image can match several partners,
        # so pair counts can exceed the dataset size — contamination rates
        # must use unique images)
        exact_a_rel = {e["a"] for e in exact_pairs}
        exact_b_rel = {e["b"] for e in exact_pairs}
        near_a_rel = {e["a"] for e in near_pairs}
        near_b_rel = {e["b"] for e in near_pairs}
        hist: dict[str, int] = {}
        for entry in near_pairs:
            dist = entry.get("phash")
            if dist is not None:
                hist[str(dist)] = hist.get(str(dist), 0) + 1
        pairs[key] = {
            "n_filename_overlap": len(shared_names),
            "filename_examples": shared_names[:10],
            "n_exact_duplicates": len(exact_pairs),
            "exact_examples": exact_pairs[:20],
            "n_unique_a_exact": len(exact_a_rel),
            "n_unique_b_exact": len(exact_b_rel),
            "n_near_duplicates_phash": sum(1 for e in near_pairs if "phash" in e),
            "n_near_duplicates_dhash": sum(1 for e in near_pairs if "dhash" in e),
            "n_near_duplicates_union": len(near_pairs),
            "near_phash_distance_histogram": dict(sorted(hist.items())),
            "n_unique_a_overlap": len(exact_a_rel | near_a_rel),
            "n_unique_b_overlap": len(exact_b_rel | near_b_rel),
            "near_examples": near_pairs[:20],
            "n_overlap_union": overlap_n,
            "n_exact_label_agree": label_agree,
            "n_exact_label_disagree": label_disagree,
            "n_exact_label_unknown": label_unknown,
            "label_disagreement_examples": label_disagreement_examples,
            "overlap_rate_vs_smaller_source": round(overlap_n / denominator, 6)
            if denominator
            else None,
        }
        print(
            f"  {key}: filename={len(shared_names)} exact={len(exact_pairs)} "
            f"near={len(near_pairs)} unique_b={len(exact_b_rel | near_b_rel)}"
        )

    # ---- 4. BRISC publisher-manifest sha256 cross-check ------------------
    brisc_manifest_check = None
    if BRISC_MANIFEST_CSV.is_file():
        manifest_sha: dict[str, str] = {}
        with BRISC_MANIFEST_CSV.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("task") == "classification" and row.get("is_mask") == "False":
                    manifest_sha[row["relative_path"].replace("\\", "/").lower()] = row["sha256"]
        matched = mismatched = missing = 0
        for name in ("brisc_train", "brisc_test"):
            for record in digested.get(name, []):
                key = rel(record["path"]).replace("\\", "/")
                # manifest paths are rooted at brisc2025/
                key = key[key.lower().index("classification_task/"):]
                expected = manifest_sha.get(key.lower())
                if expected is None:
                    missing += 1
                elif expected == record["sha256"]:
                    matched += 1
                else:
                    mismatched += 1
        brisc_manifest_check = {
            "manifest": rel(BRISC_MANIFEST_CSV),
            "matched": matched,
            "mismatched": mismatched,
            "not_in_manifest": missing,
        }

    # ---- 5. thumbnails ---------------------------------------------------
    thumb_files: list[str] = []
    if not args.no_thumbs and thumb_candidates:
        THUMBS_DIR.mkdir(parents=True, exist_ok=True)
        for stale in THUMBS_DIR.glob("[0-9][0-9]__*.jpg"):
            stale.unlink()  # only our own numbered outputs; keep boundary_*/check_*
        # headline pairs first, then alphabetically; split the budget so both
        # exact and near-duplicate examples are represented
        priority = {
            pair: index
            for index, pair in enumerate(
                [
                    "internal_train__kaggle_testing",
                    "internal_train__kaggle_training",
                    "internal_test__kaggle_testing",
                    "internal_train__brisc_train",
                    "internal_test__brisc_test",
                    "internal_test__brisc_train",
                    "kaggle_training__kaggle_testing",
                    "kaggle_testing__brisc_test",
                    "brisc_train__brisc_test",
                    "internal_train__internal_test",
                ]
            )
        }

        def _order(candidate: dict) -> tuple:
            return (
                priority.get(candidate["pair"], 99),
                candidate["pair"],
                candidate["phash"] if candidate["phash"] is not None else 99,
            )

        exact_cands = sorted(
            [c for c in thumb_candidates if c["kind"] == "exact"], key=_order
        )
        near_cands = sorted(
            [c for c in thumb_candidates if c["kind"] == "near"], key=_order
        )
        half = THUMB_MAX_PAIRS // 2
        selected = exact_cands[:half] + near_cands[: THUMB_MAX_PAIRS - half]
        for index, candidate in enumerate(selected, start=1):
            distance = (
                f"ph{candidate['phash']}"
                if candidate["phash"] is not None
                else f"dh{candidate['dhash']}"
            )
            out_path = THUMBS_DIR / f"{index:02d}__{candidate['pair']}__{distance}.jpg"
            try:
                make_thumbnail(
                    candidate["a_path"],
                    candidate["b_path"],
                    f"{index:02d} {candidate['pair']} [{candidate['kind']}] "
                    f"{out_path.stem.split('__')[-1]}",
                    out_path,
                )
                thumb_files.append(rel(out_path))
            except Exception as exc:
                print(f"    thumb failed: {exc}", file=sys.stderr)

    # ---- 6. verdicts -----------------------------------------------------
    kaggle_key = "internal_train__kaggle_testing"
    kaggle_verdict = None
    if kaggle_key in pairs:
        pair = pairs[kaggle_key]
        n_kaggle = sources.get("kaggle_testing", {}).get("n_images", 0)
        contaminated = pair["n_unique_b_overlap"]
        kaggle_verdict = {
            "statement": (
                "Kaggle Testing is NOT an independent external set: "
                f"{contaminated} of {n_kaggle} Kaggle Testing images "
                f"({contaminated / n_kaggle:.1%}) also occur in internal_train "
                f"(byte-identical: {pair['n_unique_b_exact']}, "
                f"perceptual near-duplicate only: "
                f"{contaminated - pair['n_unique_b_exact']})."
                if n_kaggle
                else "kaggle_testing missing"
            ),
            "contaminated_unique_images": contaminated,
            "byte_identical_unique_images": pair["n_unique_b_exact"],
            "total": n_kaggle,
            "contamination_rate": round(contaminated / n_kaggle, 6)
            if n_kaggle
            else None,
        }

    # all pairs with any content overlap, unique-image rates, for quick reading
    contamination_summary = []
    for key, pair in sorted(pairs.items()):
        src_a, src_b = key.split("__")
        n_a = sources.get(src_a, {}).get("n_images", 0)
        n_b = sources.get(src_b, {}).get("n_images", 0)
        contamination_summary.append(
            {
                "pair": key,
                "exact_pairs": pair["n_exact_duplicates"],
                "near_pairs": pair["n_near_duplicates_union"],
                f"unique_{src_a}": pair["n_unique_a_overlap"],
                f"unique_{src_b}": pair["n_unique_b_overlap"],
                f"rate_{src_a}": round(pair["n_unique_a_overlap"] / n_a, 4)
                if n_a
                else None,
                f"rate_{src_b}": round(pair["n_unique_b_overlap"] / n_b, 4)
                if n_b
                else None,
            }
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/leakage_check.py",
        "git": git_metadata(),
        "config": {
            "phash_hamming_max": PHASH_MAX,
            "dhash_hamming_max": DHASH_MAX,
            "exact": "sha256 of raw file bytes",
            "image_extensions": sorted(IMAGE_EXTENSIONS),
            "sources": {name: rel(path) for name, path in SOURCES.items()},
        },
        "runtime_seconds": round(time.time() - started, 1),
        "sources": sources,
        "pairs": pairs,
        "unreadable_images": errors[:50],
        "n_unreadable_images": len(errors),
        "brisc_manifest_sha256_check": brisc_manifest_check,
        "verdicts": {
            "kaggle_testing_independence": kaggle_verdict,
            "all_pairs": contamination_summary,
        },
        "thumbnails": thumb_files,
        "metrics_before_after_deduplication": (
            "TODO-RERUN (Phase 3): requires model re-evaluation on "
            "de-duplicated test sets; not computed by this script."
        ),
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {rel(REPORT_PATH)} ({len(thumb_files)} thumbnails)")
    if kaggle_verdict:
        print(f"  {kaggle_verdict['statement']}")
    print(f"runtime: {report['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
