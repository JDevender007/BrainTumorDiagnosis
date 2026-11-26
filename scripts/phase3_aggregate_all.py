"""Group-by-config aggregation for the Phase 3 TEST evaluations.

Reads ``results/phase3/runs/<run>/eval_internal_test/metrics.json`` for every run
and produces one summary per configuration group:

  - ``mean`` over seeds (never the best seed)
  - ``std_sample_ddof1`` (sample std), min, max
  - every per-seed value with its source path, so each paper number traces back
    to exactly one evaluation artifact

Writes:
  results/phase3/agg/<slug>.json            one file per group
  results/phase3/agg/combined_summary.json  all groups + provenance

Single-seed groups (resnet50/convnext/vit/leaky/ep50/λ-sweep configs) are kept
with ``n_seeds = 1`` and no std - they are reported as point estimates only.

Usage:
    python scripts/phase3_aggregate_all.py [--eval-dir eval_internal_test]
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "phase3" / "runs"
OUT = ROOT / "results" / "phase3" / "agg"
METRICS = ("accuracy", "macro_f1", "weighted_f1")

GROUPS: list[tuple[str, str, list[str]]] = [
    ("tamnet_clean", "TAMNet baseline (g+t fusion), clean protocol",
     ["tamnet_clean_s42", "tamnet_clean_s43", "tamnet_clean_s44", "tamnet_clean_s45", "tamnet_clean_s46"]),
    ("tamnet_clean_mc", "TAMNet + channel attention (MC)", 
     ["tamnet_clean_s42+mc", "tamnet_clean_s43+mc", "tamnet_clean_s44+mc", "tamnet_clean_s45+mc"]),
    ("tamnet_clean_concat", "TAMNet with concat fusion (no attention fusion)",
     ["tamnet_clean_s42+concat", "tamnet_clean_s43+concat", "tamnet_clean_s44+concat", "tamnet_clean_s45+concat"]),
    ("tamnet_clean_tonly", "TAMNet t-only fusion",
     ["tamnet_clean_s42-tonly", "tamnet_clean_s43-tonly", "tamnet_clean_s44-tonly", "tamnet_clean_s45-tonly"]),
    ("densenet121_clean", "DenseNet-121 classification-only baseline",
     ["densenet121_clean_s42", "densenet121_clean_s43", "densenet121_clean_s44", "densenet121_clean_s45"]),
    ("efficientnet_v2_s_clean", "EfficientNetV2-S baseline",
     ["efficientnet_v2_s_clean_s42", "efficientnet_v2_s_clean_s43", "efficientnet_v2_s_clean_s44"]),
    ("swin_t_clean", "Swin-T baseline",
     ["swin_t_clean_s42", "swin_t_clean_s43", "swin_t_clean_s44"]),
    ("shared_unet_brisc", "Shared-encoder U-Net multi-task baseline (BRISC trained)",
     ["shared_unet_brisc_s42+ls0.25+sc2", "shared_unet_brisc_s43+ls0.25+sc2",
      "shared_unet_brisc_s44+ls0.25+sc2", "shared_unet_brisc_s45+ls0.25+sc2"]),
    ("resnet50_clean", "ResNet-50 baseline (single seed)",
     ["resnet50_clean_s42"]),
    ("convnext_tiny_clean", "ConvNeXt-Tiny baseline (single seed)",
     ["convnext_tiny_clean_s42"]),
    ("vit_b_16_clean", "ViT-B/16 baseline (single seed)",
     ["vit_b_16_clean_s42"]),
    ("tamnet_leaky_control", "Leaky-split control (reproduces the historical split)",
     ["tamnet_leaky_s42"]),
    ("tamnet_clean_ep50", "TAMNet 50-epoch budget control",
     ["tamnet_clean_s42_ep50budget"]),
    ("tamnet_brisc_lambda0", "BRISC λ_seg = 0 (no segmentation loss)",
     ["tamnet_brisc_s42+sc2"]),
    ("tamnet_brisc_lambda0.1", "BRISC λ_seg = 0.1",
     ["tamnet_brisc_s42+ls0.1+sc2"]),
    ("tamnet_brisc_lambda0.25", "BRISC λ_seg = 0.25",
     ["tamnet_brisc_s42+ls0.25+sc2"]),
    ("tamnet_brisc_lambda1", "BRISC λ_seg = 1.0",
     ["tamnet_brisc_s42+ls1+sc2"]),
]


def git_info() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=ROOT, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "results/phase3/agg"],
            capture_output=True, text=True, cwd=ROOT, check=True,
        ).stdout.strip() != ""
        return {"commit": commit, "agg_dirty": dirty}
    except Exception:  # noqa: BLE001
        return {"commit": None, "agg_dirty": None}


def summarize(eval_dir: str) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    groups_out = []
    missing = []

    for slug, description, names in GROUPS:
        per_seed = []
        for name in names:
            path = RUNS / name / eval_dir / "metrics.json"
            if not path.exists():
                missing.append(name)
                continue
            m = json.loads(path.read_text(encoding="utf-8"))
            prov = m.get("provenance", {})
            per_seed.append({
                "run": name,
                "seed": prov.get("seed"),
                "checkpoint_best_epoch": m.get("checkpoint_best_epoch"),
                "n": m.get("n"),
                **{k: m[k] for k in METRICS},
                "source": path.relative_to(ROOT).as_posix(),
            })
        if not per_seed:
            continue

        summary = {}
        for metric in METRICS:
            vals = [s[metric] for s in per_seed]
            summary[metric] = {
                "mean": statistics.fmean(vals),
                "std_sample_ddof1": statistics.stdev(vals) if len(vals) > 1 else None,
                "min": min(vals),
                "max": max(vals),
            }

        payload = {
            "kind": "phase3_test_summary_group",
            "slug": slug,
            "description": description,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git": git_info(),
            "eval_dir": eval_dir,
            "n_seeds": len(per_seed),
            "headline_rule": "mean over seeds (never best seed); std = sample ddof=1; single seed = point estimate",
            "per_seed": per_seed,
            "summary": summary,
        }
        (OUT / f"{slug}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        groups_out.append({
            "slug": slug,
            "description": description,
            "n_seeds": len(per_seed),
            "summary": summary,
            "per_seed_acc": {s["run"]: s["accuracy"] for s in per_seed},
        })

    combined = {
        "kind": "phase3_test_summary_all",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_info(),
        "eval_dir": eval_dir,
        "headline_rule": "mean over seeds (never best seed); std = sample ddof=1",
        "groups": groups_out,
        "runs_missing_eval": missing,
    }
    (OUT / "combined_summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    return combined


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="eval_internal_test")
    args = ap.parse_args()

    combined = summarize(args.eval_dir)

    print("=" * 96)
    print(f" Phase 3 TEST summary ({args.eval_dir}, n=1672) - mean over seeds ± sample std")
    print("=" * 96)
    print(f" {'group':<26} {'n':>2}  {'accuracy':<16} {'macro-F1':<16} {'weighted-F1':<16}")
    print("-" * 96)
    rows = sorted(
        combined["groups"],
        key=lambda g: -g["summary"]["accuracy"]["mean"],
    )
    for g in rows:
        acc = g["summary"]["accuracy"]
        mf1 = g["summary"]["macro_f1"]
        wf1 = g["summary"]["weighted_f1"]

        def cell(d):
            if d["std_sample_ddof1"] is None:
                return f"{d['mean']:.4f}"
            return f"{d['mean']:.4f}±{d['std_sample_ddof1']:.4f}"

        print(f" {g['slug'][:26]:<26} {g['n_seeds']:>2}  {cell(acc):<16} {cell(mf1):<16} {cell(wf1):<16}")
    print("-" * 96)
    if combined["runs_missing_eval"]:
        print(" missing evals:", ", ".join(combined["runs_missing_eval"]))
    print(f" wrote {OUT.relative_to(ROOT)}/*.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
