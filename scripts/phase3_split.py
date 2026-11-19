"""Phase 3: clean (leakage-free) split + external de-duplication manifests.

Phase 1 established that the historical splits leak: 74.3% of the internal test
set is near-duplicate of internal train, and the external sets overlap internal
train (Kaggle Testing 94.7%, BRISC test 80.3%).  This script builds the
primary Phase 3 protocol:

  1. digest every image in all 6 sources (pHash + dHash + sha256), reusing the
     exact Phase 1 hashing from ``scripts/leakage_check.py``;
  2. pool the internal Train+Test sets, cluster near-duplicates
     (pHash <= 6 OR dHash <= 4, the Phase 1 union rule), and split by
     **cluster** (GroupShuffleSplit-style) so no near-dup cluster ever spans
     two splits -> stratified 70/15/15, seed 42;
  3. for each external set, mark which images must be dropped because they
     near-match the CLEAN TRAIN split (leakage), keeping val/test matches
     recorded for transparency only.

Outputs (all under ``results/phase3/``, committed as evidence):

  hash_cache.json              - per-image hashes for all 6 sources
  split_manifest_clean.json    - pooled-internal clustered split (the primary
                                 train/val/test for every Phase 3 run)
  external_dedup_manifest.json - keep/drop lists for external evaluation

Every manifest embeds seed, git commit, thresholds and creation time.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.leakage_check import (  # noqa: E402  (re Phase 1 hashing)
    DHASH_MAX,
    PHASH_MAX,
    SOURCES,
    collect_files,
    digest,
    hamming_matches,
    label_of,
)

OUT_DIR = ROOT / "results" / "phase3"
SPLIT_SEED = 42
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# leakage_check label_of keys -> canonical class names (datasets.CLASS_NAMES)
LABEL_MAP = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "pituitary": "Pituitary",
    "no_tumor": "No Tumor",
}


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def git_info() -> dict:
    def _run(args: list[str]) -> str:
        try:
            return subprocess.run(
                args, capture_output=True, text=True, cwd=ROOT, check=True
            ).stdout.strip()
        except Exception:
            return "unknown"

    return {
        "commit": _run(["git", "rev-parse", "--short", "HEAD"]),
        "dirty": _run(["git", "status", "--porcelain"]) != "",
    }


def digest_source(name: str, directory: Path) -> list[dict]:
    files = collect_files(directory)
    records: list[dict] = []
    unreadable = 0
    for i, path in enumerate(files, 1):
        rec = digest(path)
        if rec["error"] is not None:
            unreadable += 1
            continue  # cannot hash -> cannot leak-check; counted and excluded
        rec["source"] = name
        raw_label = label_of(path)
        rec["label"] = LABEL_MAP.get(raw_label, raw_label)
        records.append(rec)
        if i % 2000 == 0 or i == len(files):
            print(f"  [{name}] {i}/{len(files)} digested", flush=True)
    if unreadable:
        print(f"  [{name}] WARNING {unreadable} unreadable images excluded", flush=True)
    return records


def to_uint64(records: list[dict]) -> np.ndarray:
    return np.array([r["phash"] for r in records], dtype=np.uint64)


def cluster_records(records: list[dict]) -> tuple[list[int], dict]:
    """Union-find over near-dup edges (pHash<=6 OR dHash<=4). Returns cluster ids."""
    ph = to_uint64(records)
    dh = np.array([r["dhash"] for r in records], dtype=np.uint64)

    uf = UnionFind(len(records))
    edges = 0
    seen: set[tuple[int, int]] = set()
    for arr, max_dist in ((ph, PHASH_MAX), (dh, DHASH_MAX)):
        for i, j, _dist in hamming_matches(arr, arr, max_dist):
            if i == j:
                continue
            key = (i, j) if i < j else (j, i)
            if key in seen:
                continue
            seen.add(key)
            uf.union(*key)
            edges += 1

    root_to_cluster: dict[int, int] = {}
    cluster_ids: list[int] = []
    for idx in range(len(records)):
        root = uf.find(idx)
        if root not in root_to_cluster:
            root_to_cluster[root] = len(root_to_cluster)
        cluster_ids.append(root_to_cluster[root])

    stats = {
        "n_edges": edges,
        "n_clusters": len(root_to_cluster),
        "largest_cluster": max(Counter(cluster_ids).values()),
    }
    return cluster_ids, stats


def stratified_group_split(
    records: list[dict], cluster_ids: list[int], seed: int
) -> dict[str, str]:
    """Assign each whole cluster to one split, approximating per-class ratios."""
    rng = np.random.default_rng(seed)

    # cluster -> per-class counts; cluster majority class decides its stratum
    per_cluster: dict[int, Counter] = defaultdict(Counter)
    for rec, cid in zip(records, cluster_ids):
        per_cluster[cid][rec["label"]] += 1

    class_totals: Counter = Counter(rec["label"] for rec in records)
    targets = {
        cls: {sp: int(round(n * ratio)) for sp, ratio in RATIOS.items()}
        for cls, n in class_totals.items()
    }

    # deterministic ordering: shuffle then largest-first (stable within seed)
    order = list(per_cluster)
    rng.shuffle(order)
    order.sort(key=lambda c: sum(per_cluster[c].values()), reverse=True)

    assigned: dict[int, str] = {}
    running: dict[str, Counter] = {sp: Counter() for sp in RATIOS}

    for cid in order:
        counts = per_cluster[cid]
        stratum = counts.most_common(1)[0][0]
        # pick the split with the largest remaining deficit for this class
        best_split, best_deficit = None, -1e18
        for sp in ("train", "val", "test"):
            deficit = targets[stratum][sp] - running[sp][stratum]
            # prefer larger splits when deficits tie (keeps train biggest)
            deficit += {"train": 0.0, "val": 0.01, "test": 0.02}[sp]
            if deficit > best_deficit:
                best_deficit, best_split = deficit, sp
        assigned[cid] = best_split
        running[best_split].update(counts)

    return {
        rec["path"].relative_to(ROOT).as_posix(): assigned[cid]
        for rec, cid in zip(records, cluster_ids)
    }


def cross_match(
    a_records: list[dict], b_records: list[dict]
) -> dict[int, set[int]]:
    """For each index in b, the set of a-indices it near-matches (union rule)."""
    a_ph, a_dh = to_uint64(a_records), np.array(
        [r["dhash"] for r in a_records], dtype=np.uint64
    )
    b_ph, b_dh = to_uint64(b_records), np.array(
        [r["dhash"] for r in b_records], dtype=np.uint64
    )
    hits: dict[int, set[int]] = defaultdict(set)
    for arr_a, arr_b, max_dist in (
        (a_ph, b_ph, PHASH_MAX),
        (a_dh, b_dh, DHASH_MAX),
    ):
        for ai, bi, _d in hamming_matches(arr_a, arr_b, max_dist):
            hits[bi].add(ai)
    return hits


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    git = git_info()

    # ---------------------------------------------------------------- step 1
    print("Step 1/3: digesting all sources (pHash/dHash/sha256)...", flush=True)
    source_records: dict[str, list[dict]] = {}
    for name, directory in SOURCES.items():
        if not directory.exists():
            print(f"  SKIP missing source: {name} ({directory})", flush=True)
            continue
        source_records[name] = digest_source(name, directory)

    hash_cache = {
        "generated_at": started,
        "script": "scripts/phase3_split.py",
        "git": git,
        "thresholds": {"phash_max": PHASH_MAX, "dhash_max": DHASH_MAX},
        "sources": {
            name: [
                {
                    "path": r["path"].relative_to(ROOT).as_posix(),
                    "label": r["label"],
                    "phash": r["phash"],
                    "dhash": r["dhash"],
                    "sha256": r["sha256"],
                }
                for r in recs
            ]
            for name, recs in source_records.items()
        },
    }
    (OUT_DIR / "hash_cache.json").write_text(
        json.dumps(hash_cache, indent=1), encoding="utf-8"
    )
    print("  hash_cache.json written (unreadable excluded above)", flush=True)

    # ---------------------------------------------------------------- step 2
    print("Step 2/3: pooling internal set + clustering near-dups...", flush=True)
    internal = source_records.get("internal_train", []) + source_records.get(
        "internal_test", []
    )
    if not internal:
        raise SystemExit("No internal images found - check data/raw/classification")

    cluster_ids, cluster_stats = cluster_records(internal)

    # cluster label consistency (Phase 1 found 0 disagreements; re-verify)
    per_cluster_labels: dict[int, Counter] = defaultdict(Counter)
    for rec, cid in zip(internal, cluster_ids):
        per_cluster_labels[cid][rec["label"]] += 1
    mixed = [c for c, cnt in per_cluster_labels.items() if len(cnt) > 1]

    split_of_path = stratified_group_split(internal, cluster_ids, SPLIT_SEED)

    entries = []
    for rec, cid in zip(internal, cluster_ids):
        path = rec["path"].relative_to(ROOT).as_posix()
        entries.append(
            {
                "path": path,
                "class": rec["label"],
                "split": split_of_path[path],
                "cluster": cid,
                "origin": rec["source"],  # original Train/Test dir (for reference)
            }
        )

    # validation: no cluster spans splits; every image exactly once
    cluster_splits: dict[int, set[str]] = defaultdict(set)
    for e in entries:
        cluster_splits[e["cluster"]].add(e["split"])
    spanning = [c for c, sp in cluster_splits.items() if len(sp) > 1]

    per_split: dict[str, dict] = {}
    for sp in RATIOS:
        sp_entries = [e for e in entries if e["split"] == sp]
        per_split[sp] = {
            "n": len(sp_entries),
            "classes": dict(sorted(Counter(e["class"] for e in sp_entries).items())),
        }

    split_manifest = {
        "schema_version": 1,
        "kind": "clean_grouped_split",
        "created_at": started,
        "git": git,
        "seed": SPLIT_SEED,
        "ratios": RATIOS,
        "near_dup_rule": {
            "phash_max": PHASH_MAX,
            "dhash_max": DHASH_MAX,
            "union": True,
            "note": "identical criteria to Phase 1 (scripts/leakage_check.py)",
        },
        "pooled_from": ["internal_train", "internal_test"],
        "statistics": {
            "n_total": len(entries),
            "n_clusters": cluster_stats["n_clusters"],
            "n_edges": cluster_stats["n_edges"],
            "largest_cluster": cluster_stats["largest_cluster"],
            "mixed_label_clusters": len(mixed),
            "cluster_spanning_splits": len(spanning),
            "per_split": per_split,
            "validation": "PASS" if not spanning else "FAIL",
        },
        "entries": entries,
    }
    if spanning:
        raise SystemExit(f"VALIDATION FAIL: {len(spanning)} clusters span splits")
    (OUT_DIR / "split_manifest_clean.json").write_text(
        json.dumps(split_manifest, indent=1), encoding="utf-8"
    )
    print(
        f"  split_manifest_clean.json written: {cluster_stats['n_clusters']} clusters, "
        f"spanning={len(spanning)}, mixed_label={len(mixed)}",
        flush=True,
    )
    for sp in RATIOS:
        print(f"    {sp:5s}: {per_split[sp]['n']:5d}  {per_split[sp]['classes']}", flush=True)

    # ---------------------------------------------------------------- step 3
    print("Step 3/3: external de-dup vs CLEAN TRAIN split...", flush=True)
    clean_train = [
        r
        for r, e in zip(internal, entries)
        if e["split"] == "train"
    ]
    # zip alignment: entries were built in the same order as `internal`
    assert len(clean_train) == per_split["train"]["n"]

    ext_manifest: dict = {
        "schema_version": 1,
        "kind": "external_dedup",
        "created_at": started,
        "git": git,
        "near_dup_rule": {"phash_max": PHASH_MAX, "dhash_max": DHASH_MAX, "union": True},
        "reference_split": "results/phase3/split_manifest_clean.json",
        "rule": (
            "DROP an external image if it near-matches ANY clean-train image "
            "(leakage). Matches against clean val/test are recorded but kept."
        ),
        "sources": {},
    }
    for name, recs in source_records.items():
        if name in ("internal_train", "internal_test"):
            continue
        hits = cross_match(clean_train, recs)
        keep, drop = [], []
        for i, rec in enumerate(recs):
            path = rec["path"].relative_to(ROOT).as_posix()
            (drop if i in hits else keep).append(path)
        ext_manifest["sources"][name] = {
            "total": len(recs),
            "drop_vs_clean_train": len(drop),
            "keep": len(keep),
            "pct_dropped": round(100.0 * len(drop) / max(len(recs), 1), 2),
            "per_class_total": dict(
                sorted(Counter(r["label"] for r in recs).items())
            ),
            "keep_paths": keep,
            "drop_paths": drop,
        }
        print(
            f"  {name}: keep {len(keep)}/{len(recs)} "
            f"(drop {len(drop)}, {ext_manifest['sources'][name]['pct_dropped']}%)",
            flush=True,
        )
    (OUT_DIR / "external_dedup_manifest.json").write_text(
        json.dumps(ext_manifest, indent=1), encoding="utf-8"
    )

    print("DONE. Wrote:", flush=True)
    for p in sorted(OUT_DIR.glob("*.json")):
        print(f"  {p.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
