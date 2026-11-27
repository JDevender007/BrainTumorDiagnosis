"""Phase 5 — build every paper table from the JSON artifacts, never by hand.

Reads only files under ``results/``; each table records the exact source files
it consumed, so ``scripts/check_paper_numbers.py`` can re-verify the paper text
against them.  Missing inputs are printed as ``MISSING`` and the table is
skipped — nothing is fabricated or back-filled from the old paper.

Outputs
-------
paper/tables/*.md          markdown tables (paste-ready, IEEE-ish)
paper/tables/sources.json  table -> {source files, row count, generated_at}

Usage:
    python -X utf8 scripts/make_tables.py
    python -X utf8 scripts/make_tables.py --only main_results external_dedup
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
OUT = ROOT / "paper" / "tables"

sys.path.insert(0, str(ROOT))


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def fmt(mean, std=None, digits=4) -> str:
    if mean is None:
        return "—"
    if std is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def note_missing(name: str, path: Path) -> None:
    print(f"  MISSING {name}: {path.relative_to(ROOT) if path.is_absolute() else path}")


# --------------------------------------------------------------------------- #
# 1. main results (internal TEST, mean over seeds)
# --------------------------------------------------------------------------- #
def table_main_results() -> tuple[str, dict] | None:
    src = R / "phase3" / "agg" / "combined_summary.json"
    data = read_json(src)
    if not data:
        note_missing("main_results", src)
        return None
    rows = sorted(data["groups"], key=lambda g: -g["summary"]["accuracy"]["mean"])
    lines = [
        "| Model / configuration | Seeds | Accuracy | Macro-F1 | Weighted-F1 |",
        "|---|---|---|---|---|",
    ]
    for g in rows:
        s = g["summary"]
        lines.append(
            f"| {g['description']} | {g['n_seeds']} | "
            f"{fmt(s['accuracy']['mean'], s['accuracy']['std_sample_ddof1'])} | "
            f"{fmt(s['macro_f1']['mean'], s['macro_f1']['std_sample_ddof1'])} | "
            f"{fmt(s['weighted_f1']['mean'], s['weighted_f1']['std_sample_ddof1'])} |"
        )
    lines += [
        "",
        "Values are mean over seeds; ± is the sample standard deviation (ddof = 1) and is "
        "omitted when only one seed exists. Internal TEST split, n = 1,672, evaluated once "
        "per final configuration.",
        "",
        f"Source: `{src.relative_to(ROOT).as_posix()}`",
    ]
    return "\n".join(lines), {"source": [src.as_posix()], "rows": len(rows)}


# --------------------------------------------------------------------------- #
# 2. confusion/noise: bootstrap CIs + paired tests
# --------------------------------------------------------------------------- #
def table_bootstrap() -> tuple[str, dict] | None:
    src = R / "phase3" / "bootstrap_ci.json"
    data = read_json(src)
    if not data:
        note_missing("bootstrap", src)
        return None
    lines = [
        "| Run | Accuracy | 95% CI (bootstrap) |",
        "|---|---|---|",
    ]
    for name in sorted(data["runs"]):
        a = data["runs"][name]["metrics"]["accuracy"]
        lines.append(f"| {name} | {a['point']:.4f} | [{a['ci95_low']:.4f}, {a['ci95_high']:.4f}] |")
    pairs = data.get("paired_tamnet_vs_baseline") or []
    if pairs:
        lines += [
            "",
            "| Comparison (same test indices) | Δ accuracy | 95% CI | Conclusion |",
            "|---|---|---|---|",
        ]
        for p in pairs:
            verdict = "different" if p["significant"] else "not distinguishable"
            lines.append(
                f"| {p['tamnet']} − {p['baseline']} | {p['delta']:+.4f} | "
                f"[{p['ci95_low']:+.4f}, {p['ci95_high']:+.4f}] | {verdict} |"
            )
    lines += [
        "",
        f"Percentile bootstrap, B = {data['splits']}, resampling test images (splits are "
        "image-level, so this is not a patient-level interval).",
        "",
        f"Source: `{src.relative_to(ROOT).as_posix()}`",
    ]
    return "\n".join(lines), {"source": [src.as_posix()], "rows": len(data["runs"])}


# --------------------------------------------------------------------------- #
# 3. external validation, before / after de-duplication
# --------------------------------------------------------------------------- #
def table_external() -> tuple[str, dict] | None:
    src = R / "phase3" / "external_eval_summary.json"
    data = read_json(src)
    if not data:
        note_missing("external", src)
        return None
    lines = [
        "| Evaluation set | Images | Accuracy (mean ± SD) | Macro-F1 |",
        "|---|---|---|---|",
    ]
    order = ["kaggle_testing_full", "kaggle_testing_dedup",
             "brisc_test_full", "brisc_test_dedup"]
    for key in order:
        a = (data.get("aggregate") or {}).get(key)
        if not a:
            continue
        label = {"kaggle_testing_full": "Kaggle Testing — all images (leaky)",
                 "kaggle_testing_dedup": "Kaggle Testing — de-duplicated",
                 "brisc_test_full": "BRISC test — all images (leaky)",
                 "brisc_test_dedup": "BRISC test — de-duplicated"}[key]
        lines.append(
            f"| {label} | {a['n_images_per_seed']} | "
            f"{fmt(a['accuracy_mean'], a['accuracy_std'])} | "
            f"{fmt(a['macro_f1_mean'], a['macro_f1_std'])} |"
        )
    lines += [
        "",
        "'De-duplicated' drops every external image whose perceptual hash near-matches a "
        "clean-training image (pHash ≤ 6 or dHash ≤ 4 union). The all-images row reproduces "
        "the leakage the legacy numbers reported; the de-duplicated row is the only number "
        "that may be called external generalisation.",
        "",
        f"Source: `{src.relative_to(ROOT).as_posix()}`",
    ]
    return "\n".join(lines), {"source": [src.as_posix()], "rows": len(order)}


# --------------------------------------------------------------------------- #
# 4. calibration / uncertainty
# --------------------------------------------------------------------------- #
def table_calibration() -> tuple[str, dict] | None:
    src = R / "phase3" / "uncertainty_summary.json"
    data = read_json(src)
    if not data or not data.get("groups"):
        note_missing("calibration", src)
        return None
    lines = [
        "| Configuration | Seeds | Accuracy | ECE (15 bins) | MCE | NLL | Brier | MI (epistemic) | Error AUROC |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    rows = sorted(data["groups"], key=lambda g: -g["summary"]["accuracy"]["mean"])
    for g in rows:
        s = g["summary"]

        def cell(k):
            d = s.get(k)
            return fmt(d["mean"], d["std_sample_ddof1"]) if d else "—"

        lines.append(
            f"| {g['description']} | {g['n_seeds']} | {cell('accuracy')} | {cell('ece_15')} | "
            f"{cell('mce_15')} | {cell('nll')} | {cell('brier')} | "
            f"{cell('mean_mutual_information')} | {cell('error_detection_auroc')} |"
        )
    lines += [
        "",
        f"MC-Dropout T = {data.get('passes', 10)} with BatchNorm frozen and only Dropout "
        "layers active (Phase 2 / D5 fix). Error AUROC = how well mutual information "
        "separates misclassified from correctly classified slices (0.5 = no signal).",
        "",
        f"Source: `{src.relative_to(ROOT).as_posix()}`",
    ]
    return "\n".join(lines), {"source": [src.as_posix()], "rows": len(rows)}


# --------------------------------------------------------------------------- #
# 5. robustness
# --------------------------------------------------------------------------- #
def table_robustness() -> tuple[str, dict] | None:
    src = R / "phase3" / "robustness.json"
    data = read_json(src)
    if not data or not data.get("runs"):
        note_missing("robustness", src)
        return None
    lines = [
        "| Configuration | Seeds | Clean accuracy | MRRS (mean ± SD) | Best-seed MRRS |",
        "|---|---|---|---|---|",
    ]
    for slug, g in (data.get("groups") or {}).items():
        if g.get("mrrs_mean") is None:
            continue
        cleans = [r["clean_accuracy"] for k, r in data["runs"].items() if k.startswith(slug)]
        clean = (sum(cleans) / len(cleans)) if cleans else None
        lines.append(
            f"| {slug} | {g['n_seeds']} | {fmt(clean)} | "
            f"{fmt(g['mrrs_mean'], g.get('mrrs_std_sample_ddof1'))} | "
            f"{g['mrrs_best_seed']:.4f} |"
        )
    lines += [
        "",
        "MRRS = mean over the 27 corruptions (9 types × 3 severities, listed in the source "
        "file) of accuracy(corruption) / accuracy(clean). Headline value is the mean over "
        "seeds, never the best seed.",
        "",
        f"Source: `{src.relative_to(ROOT).as_posix()}`",
    ]
    return "\n".join(lines), {"source": [src.as_posix()], "rows": len(data["runs"])}


# --------------------------------------------------------------------------- #
# 6. efficiency
# --------------------------------------------------------------------------- #
def table_efficiency() -> tuple[str, dict] | None:
    src = R / "phase3" / "efficiency.json"
    data = read_json(src)
    if not data or not data.get("models"):
        note_missing("efficiency", src)
        return None
    ap = data.get("protocol", {})
    mckey = f"mc_{ap.get('mc_passes', 10)}_pass"
    lines = [
        "| Model | Parameters | Weights (MiB) | Single pass (ms) | MC 10 passes (ms) | FPS (single) |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in data["models"].items():
        lines.append(
            f"| {name} | {m['params_total']:,} | {m['weights_mib']:.2f} | "
            f"{m['single_pass']['median_ms']:.2f} | {m[mckey]['median_ms']:.2f} | "
            f"{m['single_pass']['fps_from_median']:.1f} |"
        )
    lines += [
        "",
        f"Protocol: batch {ap.get('batch_size')}, {ap.get('warmup_iterations')} warm-up, "
        f"{ap.get('timed_iterations')} timed iterations, CUDA synchronised around each "
        f"iteration, device = {data.get('hardware', {}).get('device')}.",
        "",
        f"Source: `{src.relative_to(ROOT).as_posix()}`",
    ]
    return "\n".join(lines), {"source": [src.as_posix()], "rows": len(data["models"])}


# --------------------------------------------------------------------------- #
# 7. segmentation (aux branch)
# --------------------------------------------------------------------------- #
def table_segmentation() -> tuple[str, dict] | None:
    src = R / "phase3" / "seg_metrics_summary.json"
    data = read_json(src)
    if not data or not data.get("groups"):
        note_missing("segmentation", src)
        return None
    lines = [
        "| Configuration | Seeds | Dice (tumour slices) | IoU | Pixel accuracy | Empty→empty |",
        "|---|---|---|---|---|---|",
    ]
    for g in data["groups"]:
        lines.append(
            f"| {g['slug']} | {g['n_seeds']} | "
            f"{fmt(g['dice_tumor_mean'], g.get('dice_tumor_std_sample_ddof1'))} | "
            f"{g['iou_tumor_mean']:.4f} | {g['pixel_acc_mean']:.4f} | "
            f"{g['empty_gt_empty_pred_mean']:.3f} |"
        )
    lines += [
        "",
        "Binary tumour-vs-background Dice of the **auxiliary segmentation head** on the "
        "BRISC 2025 test split (2-D 224×224 slices). These are not the paper's WT/TC/ET "
        "numbers, which came from a standalone 3-D U-Net on BraTS volumes.",
        "",
        f"Source: `{src.relative_to(ROOT).as_posix()}`",
    ]
    return "\n".join(lines), {"source": [src.as_posix()], "rows": len(data["groups"])}


# --------------------------------------------------------------------------- #
# 8. attention alignment
# --------------------------------------------------------------------------- #
def table_attention() -> tuple[str, dict] | None:
    src = R / "phase3" / "attention_alignment.json"
    data = read_json(src)
    if not data:
        note_missing("attention", src)
        return None
    lines = [
        "| Map | Slices | Mass in tumour mask | Mask area (uniform baseline) | Lift | Pointing hit rate |",
        "|---|---|---|---|---|---|",
    ]
    for key, label in (("tamnet_attention", "TAMNet spatial attention"),
                       ("densenet121_gradcam", "DenseNet-121 Grad-CAM"),
                       ("uniform", "Uniform (reference)")):
        s = data.get(key)
        if not s:
            continue
        lines.append(
            f"| {label} | {s['n']} | {s['mass_in_mask_mean']:.4f} | "
            f"{s['mask_fraction_mean']:.4f} | {s['lift_over_mask_fraction']:.2f}× | "
            f"{s['pointing_hit_rate']:.3f} |"
        )
    p = data.get("paired_tamnet_minus_gradcam")
    fw = data.get("fusion_weights") or {}
    lines += [""]
    if p:
        lines.append(
            f"Paired TAMNet − Grad-CAM mass difference: {p['delta_mass_mean']:+.4f} "
            f"(95% CI [{p['ci95_low']:+.4f}, {p['ci95_high']:+.4f}], "
            f"{'significant' if p['significant'] else 'not distinguishable'})."
        )
    if fw.get("alpha_tumor_mean") is not None:
        lines.append(
            f"Fusion weights: α_global = {fw['alpha_global_mean']:.4f}, "
            f"α_tumour = {fw['alpha_tumor_mean']:.4f}."
        )
    lines += ["", f"Source: `{src.relative_to(ROOT).as_posix()}`"]
    return "\n".join(lines), {"source": [src.as_posix()], "rows": 3}


TABLES = {
    "main_results": table_main_results,
    "bootstrap": table_bootstrap,
    "external_dedup": table_external,
    "calibration": table_calibration,
    "robustness": table_robustness,
    "efficiency": table_efficiency,
    "segmentation": table_segmentation,
    "attention": table_attention,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    sources: dict[str, dict] = {}
    for name in (args.only or list(TABLES)):
        fn = TABLES.get(name)
        if not fn:
            print(f"unknown table: {name}")
            continue
        result = fn()
        if not result:
            continue
        text, meta = result
        (OUT / f"{name}.md").write_text(text + "\n", encoding="utf-8")
        sources[name] = {**meta, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        print(f"  wrote paper/tables/{name}.md ({meta['rows']} rows)")

    (OUT / "sources.json").write_text(
        json.dumps({"kind": "paper_tables_sources", "tables": sources}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote paper/tables/sources.json ({len(sources)} tables)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
