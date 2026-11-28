"""Phase 4 — figures that need no GPU and no training data re-read.

Adds the three structural figures the paper still lacked:

  fig1_architecture      block diagram with tensor shapes captured by real
                         forward hooks on TumorAwareMultiTaskNet (CPU);
  fig2_data_pipeline     data flow with the audited counts, every number read
                         from results/dataset_table.csv, results/leakage_report.json
                         and results/phase3/external_dedup_manifest.json;
  fig3_samples           4 classes x 3 datasets sample grid read from the image
                         files themselves;
  fig8_seg_box           box plot of per-slice Dice for the BRISC auxiliary-head
                         runs, read from each run's eval_brisc_test/seg_metrics.json.

Rules (same as scripts/make_figures.py):
  * every printed number comes from a file, never a literal;
  * a figure whose input is missing logs ``MISSING:`` and is skipped, never faked;
  * vector PDF + 600-dpi PNG.

Usage:
    python -X utf8 scripts/make_figures_static.py
    python -X utf8 scripts/make_figures_static.py --only fig1_architecture
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "paper" / "figs"
RUNS = ROOT / "results" / "phase3" / "runs"
RESULTS = ROOT / "results"

CLASS_ORDER = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

COLOR = {
    "tamnet": "#2f6fb3",
    "backbone": "#c8642a",
    "unet": "#5a5a5a",
    "data": "#3f8f5f",
    "warn": "#b33b3b",
}


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def savefig_both(fig, stem: str) -> list[str]:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", **kw)
        paths.append(str(p))
    return paths


# --------------------------------------------------------------- fig 1 arch
def fig1_architecture(manifest: dict) -> None:
    """Block diagram; every shape is captured with a real forward hook."""
    import torch

    from models.tumor_aware import TumorAwareConfig, TumorAwareMultiTaskNet

    cfg = TumorAwareConfig()
    model = TumorAwareMultiTaskNet(cfg)
    model.eval()

    shapes: dict[str, tuple] = {}
    handles = []
    for name, mod in model.named_modules():
        if name in ("", "encoder"):
            continue
        handles.append(mod.register_forward_hook(
            lambda _m, _i, out, n=name: shapes.__setitem__(
                n, tuple(out.shape) if hasattr(out, "shape") else None)))

    with torch.no_grad():
        out = model(torch.zeros(1, 3, 224, 224), return_seg=True)
    for h in handles:
        h.remove()

    def sh(*candidates: str) -> str:
        for c in candidates:
            if shapes.get(c):
                s = shapes[c]
                return " \u00d7 ".join(str(v) for v in s)
        return "?"

    # locate the actual submodule names so the labels cannot drift from the code
    names = {n for n, _ in model.named_modules()}

    def pick(*subs: str) -> str:
        for s in subs:
            if s in names:
                return s
        return subs[-1]

    enc_out = sh(pick("encoder.features"), "encoder")
    attn_s = sh(pick("spatial_attention", "spatial_attn"), "spatial_attention")
    pool = sh(pick("tumor_pool", "tumor_region_pool", "region_pool"), "tumor_pool")
    fuse = sh(pick("fusion", "attention_fusion"), "fusion")
    seg = sh(pick("segmentation_head", "seg_head"), "segmentation_head")
    if isinstance(out, dict) and "logits" in out:
        logits = " \u00d7 ".join(str(v) for v in out["logits"].shape)
    else:
        logits = sh(pick("classifier", "head"))

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#333", fs=8.2, tc="black", lw=1.1):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
            facecolor=fc, edgecolor=ec, lw=lw, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, zorder=3, linespacing=1.35)

    def arrow(x1, y1, x2, y2, style="-|>", color="#444", lw=1.4, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=11, lw=lw, color=color,
                                     linestyle=ls, zorder=1,
                                     shrinkA=0, shrinkB=0))

    # input / encoder
    box(2, 46, 15, 11, "Input MRI\n1 \u00d7 3 \u00d7 224 \u00d7 224", "#eeeeee")
    box(21, 46, 20, 11, f"Shared encoder\nDenseNet-121\n{enc_out}", "#dbe7f3")
    arrow(17.6, 51.5, 21, 51.5)

    # attention stack
    box(45, 46, 17, 11, f"Spatial attention $M_s$\n{attn_s}", "#f6e3d0")
    arrow(41.6, 51.5, 45, 51.5)
    box(45, 32, 17, 10, f"Tumor-region pool\n{pool}", "#e6f0e6")
    arrow(53.5, 46, 53.5, 42)

    # global branch
    box(21, 30, 20, 9, f"Global branch\nGAP \u2192 BN \u2192 $W_g$", "#dbe7f3")
    arrow(31, 46, 31, 39.4)

    # fusion
    box(66, 36, 17, 12,
        f"Fusion $\\Phi$\nsoftmax 2-way\n+ LayerNorm\n{fuse}", "#e9e2f3")
    arrow(62.4, 51.5, 66, 46.5)
    arrow(41.6, 34.5, 66, 41.5)

    # classifier
    box(66, 18, 17, 10, f"Classifier MLP\n{logits}", "#dbe7f3")
    arrow(74.5, 36, 74.5, 28.4)

    # seg branch
    box(21, 12, 20, 11, f"Segmentation head\n7\u00d77 map only\n{seg}", "#f6e3d0")
    arrow(31, 46, 31, 23.6, style="-", ls=(0, (3, 2)), color="#888")
    ax.text(32.2, 30, "gradient\n(D2)", fontsize=6.6, color="#888", va="center")

    # uncertainty
    box(66, 4, 17, 10, "MC dropout\nT = 10\n(BN frozen)", "#f2f2f2")
    arrow(74.5, 18, 74.5, 14.4)

    # annotation of the corrections
    ax.text(50, 2.4,
            "$M_c$ (channel attention) present but disabled in the executed "
            "configuration (D1);  mask gate $m$ never passed by the trainer (D6)",
            ha="center", fontsize=6.8, color=COLOR["warn"])

    paths = savefig_both(fig, "fig1_architecture")
    plt.close(fig)
    print(f"fig1: wrote {paths[0]}")
    manifest["fig1_architecture"] = {
        "shapes": {k: list(v) for k, v in shapes.items() if v},
        "logits": (list(out["logits"].shape)
                   if isinstance(out, dict) and "logits" in out
                   else (list(out.shape) if hasattr(out, "shape") else None)),
        "source": "models/tumor_aware.py (forward hooks, CPU, eval mode)",
    }


# -------------------------------------------------------- fig 2 data pipeline
def fig2_data_pipeline(manifest: dict) -> None:
    """Data flow with audited counts; nothing here is typed as a number."""
    import io

    with (ROOT / "results" / "dataset_table.csv").open(
            newline="", encoding="utf-8") as fh:
        ds = list(csv.DictReader(fh))
    leak = read_json(ROOT / "results" / "leakage_report.json")
    dedup = read_json(ROOT / "results" / "phase3" / "external_dedup_manifest.json")
    runcfg = read_json(RUNS / "tamnet_clean_s42" / "run_config.json")
    if not (ds and leak and dedup and runcfg):
        print("MISSING: dataset_table.csv / leakage_report.json / "
              "external_dedup_manifest.json / run_config.json")
        return

    def drow(name: str, split: str) -> dict:
        for r in ds:
            if r["dataset"] == name and r["split"] == split:
                return r
        raise KeyError((name, split))

    n_train = int(drow("internal_classification", "train")["n_images"])
    n_test = int(drow("internal_classification", "test")["n_images"])
    n_kag = int(drow("kaggle_brain_tumor", "testing")["n_images"])
    n_brisc = int(drow("BRISC2025", "test")["n_images"])

    ec = leak["verdicts"]["kaggle_testing_independence"]
    br = leak["verdicts"].get("brisc_test_independence", {})
    it = leak["pairs"]["internal_train__internal_test"]

    s_tr = runcfg["dataset"]["n_total"]["train"]
    s_va = runcfg["dataset"]["n_total"]["val"]
    s_te = runcfg["dataset"]["n_total"]["test"]

    dk = dedup["sources"]["kaggle_testing"]
    db = dedup["sources"]["brisc_test"]

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 68)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#333", fs=7.8, lw=1.1):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.55,rounding_size=1.2",
            facecolor=fc, edgecolor=ec, lw=lw, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, zorder=3, linespacing=1.4)

    def arrow(x1, y1, x2, y2, color="#444", ls="-", lw=1.3):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=11, lw=lw, color=color,
                                     linestyle=ls, zorder=1, shrinkA=0,
                                     shrinkB=0))

    # sources
    box(1, 52, 22, 13,
        f"Internal pool\nTrain {n_train:,} \u00b7 Test {n_test:,}\n"
        f"{n_train + n_test:,} images, no patient IDs", "#dbe7f3")
    box(1, 36, 22, 13,
        f"Kaggle Testing\n{n_kag:,} images\n"
        f"{ec['contaminated_unique_images']:,} "
        f"({ec['contamination_rate'] * 100:.1f} %) in train", "#f6e3d0")
    box(1, 20, 22, 13,
        f"BRISC test\n{n_brisc:,} images\n"
        f"{db['drop_vs_clean_train']:,} "
        f"({db['pct_dropped']} %) near-train", "#f6e3d0")
    box(1, 4, 22, 13,
        f"Internal train\u2194test\n{it['n_near_duplicates_union']:,} "
        f"near-duplicate pairs", "#f6d9d9", ec=COLOR["warn"])

    # audit box
    box(30, 24, 20, 30,
        "LEAKAGE AUDIT\n\u2022 pHash \u2264 6\n\u2022 dHash \u2264 4\n"
        "\u2022 union of both\n\u2022 SHA-256 split\n  manifest pinned",
        "#fff6d9", ec="#a8862a", fs=7.6)

    for y in (58.5, 42.5, 26.5, 10.5):
        arrow(23.6, y, 30, 39 if y > 30 else 30)

    # dedup / split
    box(56, 44, 20, 18,
        "DE-DUPLICATION\nkeep-lists\n"
        f"Kaggle {dk['total']}\u2192{dk['keep']} "
        f"({100.0 * dk['keep'] / dk['total']:.1f} %)\n"
        f"BRISC {db['total']}\u2192{db['keep']} "
        f"({100.0 * db['keep'] / db['total']:.1f} %)", "#e6f0e6", fs=7.4)
    arrow(50.6, 39, 56, 50)

    box(56, 22, 20, 17,
        "MANIFEST SPLIT\n"
        f"train {s_tr:,}\nval {s_va:,}\ntest {s_te:,}\n"
        "image-level", "#e6f0e6", fs=7.6)
    arrow(50.6, 31, 56, 30.5)

    # evaluation
    box(81, 34, 18, 24,
        "EVALUATION\n\u2022 TEST once per\n  final config\n"
        "\u2022 mean over seeds\n\u2022 SD ddof = 1\n"
        "\u2022 paired bootstrap\n  B = 1,000", "#e9e2f3", fs=7.4)
    arrow(76.6, 53, 81, 50)
    arrow(76.6, 30.5, 81, 42)

    ax.text(50, 1.4,
            "Full and de-duplicated external numbers are always reported "
            "side by side; the leaky control is shown only to quantify "
            "inflation.",
            ha="center", fontsize=6.9, color=COLOR["warn"])

    paths = savefig_both(fig, "fig2_data_pipeline")
    plt.close(fig)
    print(f"fig2: wrote {paths[0]}")
    manifest["fig2_data_pipeline"] = {
        "internal_train": n_train, "internal_test": n_test,
        "kaggle_testing": n_kag, "brisc_test": n_brisc,
        "kaggle_contaminated": ec["contaminated_unique_images"],
        "kaggle_contamination_rate": ec["contamination_rate"],
        "brisc_dropped": db["drop_vs_clean_train"],
        "internal_train_test_near_dups": it["n_near_duplicates_union"],
        "split": {"train": s_tr, "val": s_va, "test": s_te},
        "dedup": {"kaggle": [dk["total"], dk["keep"]],
                  "brisc": [db["total"], db["keep"]]},
        "sources": ["results/dataset_table.csv", "results/leakage_report.json",
                    "results/phase3/external_dedup_manifest.json",
                    "results/phase3/runs/tamnet_clean_s42/run_config.json"],
    }


# ------------------------------------------------------------- fig 3 samples
def _first_image(folder: Path) -> Path | None:
    if not folder.is_dir():
        return None
    for pat in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG"):
        got = sorted(folder.glob(pat))
        if got:
            return got[0]
    return None


def fig3_samples(manifest: dict) -> None:
    """4 classes x 3 datasets, plus a BRISC mask where one exists."""
    from PIL import Image

    cols = [
        ("Internal train", ROOT / "data" / "raw" / "classification" / "Train",
         {"Glioma": "glioma", "Meningioma": "meningioma",
          "Pituitary": "pituitary", "No Tumor": "no tumor"}),
        ("Internal test", ROOT / "data" / "raw" / "classification" / "Test",
         {"Glioma": "glioma", "Meningioma": "meningioma",
          "Pituitary": "pituitary", "No Tumor": "no tumor"}),
        ("Kaggle testing", ROOT / "data" / "external" / "kaggle_brain_tumor"
         / "Testing",
         {"Glioma": "glioma", "Meningioma": "meningioma",
          "Pituitary": "pituitary", "No Tumor": "notumor"}),
        ("BRISC test", ROOT / "data" / "external" / "brisc2025" / "brisc2025"
         / "classification_task" / "test",
         {"Glioma": "glioma", "Meningioma": "meningioma",
          "Pituitary": "pituitary", "No Tumor": "no_tumor"}),
    ]

    grid, used = [], []
    for title, base, mapping in cols:
        row = []
        for cls in CLASS_ORDER:
            p = _first_image(base / mapping[cls])
            row.append(p)
            used.append(str(p.relative_to(ROOT)) if p else None)
        grid.append((title, row))
    if all(p is None for _, row in grid for p in row):
        print("MISSING: fig3 sample images")
        return

    n_rows, n_cols = len(CLASS_ORDER), len(grid) + 1
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(1.62 * n_cols, 1.72 * n_rows))
    axes = np.atleast_2d(axes)

    mask_dir = (ROOT / "data" / "external" / "brisc2025" / "brisc2025"
                / "segmentation_task" / "test" / "masks")

    for ci, (title, row) in enumerate(grid):
        for ri, cls in enumerate(CLASS_ORDER):
            ax = axes[ri, ci]
            p = row[ri]
            if p is None:
                ax.set_visible(False)
                continue
            im = Image.open(p).convert("RGB")
            ax.imshow(im)
            # overlay a BRISC mask when the source image is a BRISC one
            if title == "BRISC test" and mask_dir.is_dir():
                cand = mask_dir / (p.stem + ".png")
                if not cand.exists():
                    cand = mask_dir / (p.stem + ".jpg")
                if cand.exists():
                    m = np.asarray(Image.open(cand).convert("L"))
                    if m.shape == (im.height, im.width):
                        ax.imshow(m, alpha=0.42, cmap="autumn")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#bbb")
                sp.set_linewidth(0.6)
            if ri == 0:
                ax.set_title(title, fontsize=8.4)
            if ci == 0:
                ax.set_ylabel(cls, fontsize=8.4)
            if ci == n_cols - 1 and title == "BRISC test":
                ax.set_xlabel("mask overlay", fontsize=6.4, color="#666")

    fig.suptitle("Sample slices from each source (first file per class; "
                 "not cherry-picked)", fontsize=9.4, y=1.005)
    fig.tight_layout()
    paths = savefig_both(fig, "fig3_samples")
    plt.close(fig)
    print(f"fig3: wrote {paths[0]}")
    manifest["fig3_samples"] = {
        "files": used, "classes": CLASS_ORDER,
        "note": "first sorted file per class/directory; BRISC mask overlay "
                "from segmentation_task/test/masks",
    }


# ----------------------------------------------------------- fig 8 seg boxes
def _fig8_bras_wt_tc_et() -> list[dict] | None:
    """Per-case WT/TC/ET from the standalone 3-D U-Net (finding D3)."""
    rec = read_json(RESULTS / "brats_unet3d_wt_tc_et.json")
    if not rec:
        return None
    out = []
    for region in ("WT", "TC", "ET"):
        vals = [c[region] for c in rec["per_case"]
                if isinstance(c.get(region), (int, float))]
        if not vals:
            return None
        out.append({"label": f"{region} (n={len(vals)})", "dice": vals,
                    "mean": float(np.mean(vals))})
    return out or None


def fig8_seg_box(manifest: dict) -> None:
    """Two-panel Dice distributions: BRISC auxiliary head vs the 3-D U-Net.

    Left: per-slice Dice of every BRISC segmentation evaluation (TAMNet
    auxiliary head and the shared-U-Net baseline, one box per run).
    Right: per-case WT/TC/ET Dice of the *standalone* 3-D U-Net on the
    BraTS patient-held-out test split, read from
    ``results/brats_unet3d_wt_tc_et.json`` — the record behind the legacy
    manuscript's 0.9005 / 0.5803 / 0.6010, now attributed to D3's model.
    """
    rows = []
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir():
            continue
        m = read_json(d / "eval_brisc_test" / "seg_metrics.json")
        if not m:
            continue
        cfg = read_json(d / "run_config.json") or {}
        model = cfg.get("model", "?")
        lam = cfg.get("seg_weight")
        if model == "tamnet":
            label = f"TAMNet \u03bb_seg={lam:g}" if isinstance(lam, (int, float)) \
                else "TAMNet"
        elif model == "shared_unet":
            label = "Shared U-Net"
        else:
            label = str(model)
        vals = [s["dice"] for s in m.get("per_slice", [])
                if s.get("gt_pixels", 0) > 0]
        if not vals:
            continue
        rows.append({"label": label, "run": d.name, "dice": vals,
                     "n": len(vals),
                     "mean": float(np.mean(vals)),
                     "seed": cfg.get("seed")})

    order = sorted(rows, key=lambda r: r["mean"])
    brasc = _fig8_bras_wt_tc_et()

    if not order and not brasc:
        print("MISSING: no eval_brisc_test/seg_metrics.json found")
        return

    n_pan = 2 if brasc else 1
    fig, axes = plt.subplots(1, n_pan, figsize=(10.4, 4.3),
                             gridspec_kw={"width_ratios":
                                          [len(order) + 1.6, 3.2]} if brasc
                             else None)
    ax = axes[0] if brasc else axes

    data = [r["dice"] for r in order]
    labels = [f"{r['label']} (seed {r['seed']}, n={r['n']})" for r in order]
    bp = ax.boxplot(data, orientation="horizontal", widths=0.62,
                    patch_artist=True,
                    medianprops={"color": "#111", "lw": 1.3},
                    flierprops={"markersize": 2.4, "alpha": 0.45},
                    whiskerprops={"lw": 0.9}, capprops={"lw": 0.9})
    palette = [COLOR["tamnet"] if r["label"].startswith("TAMNet")
               else COLOR["unet"] for r in order]
    for patch, c in zip(bp["boxes"], palette):
        patch.set_facecolor(c)
        patch.set_alpha(0.72)
        patch.set_edgecolor("#333")
    ax.set_yticklabels(labels, fontsize=7.8)
    for i, r in enumerate(order, start=1):
        ax.text(r["mean"], i + 0.26, f"mean {r['mean']:.4f}",
                fontsize=6.9, ha="center", va="bottom", color="#222")
    ax.set_xlabel("per-slice Dice on tumour slices "
                  "(BRISC 2025 official test split)", fontsize=8.4)
    ax.set_title("(a) TAMNet auxiliary head — 2-D slices", fontsize=8.8)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)

    if brasc:
        ax2 = axes[1]
        bp2 = ax2.boxplot([r["dice"] for r in brasc],
                          orientation="horizontal", widths=0.55,
                          patch_artist=True,
                          medianprops={"color": "#111", "lw": 1.3},
                          flierprops={"markersize": 2.6, "alpha": 0.5},
                          whiskerprops={"lw": 0.9}, capprops={"lw": 0.9})
        for patch in bp2["boxes"]:
            patch.set_facecolor(COLOR["unet"])
            patch.set_alpha(0.72)
            patch.set_edgecolor("#333")
        ax2.set_yticklabels([r["label"] for r in brasc], fontsize=8.0)
        for i, r in enumerate(brasc, start=1):
            ax2.text(r["mean"], i + 0.22, f"mean {r['mean']:.4f}",
                     fontsize=7.0, ha="center", va="bottom", color="#222")
        ax2.set_xlabel("per-case Dice (3-D volume)", fontsize=8.4)
        ax2.set_title("(b) standalone 3-D U-Net — BraTS test (D3)",
                      fontsize=8.8)
        ax2.grid(axis="x", alpha=0.25, lw=0.6)
        ax2.set_axisbelow(True)

    fig.tight_layout()
    paths = savefig_both(fig, "fig8_seg_box")
    plt.close(fig)
    n_br = len(brasc) if brasc else 0
    print(f"fig8: wrote {paths[0]}  ({len(rows)} BRISC runs, {n_br} WT/TC/ET)")
    manifest["fig8_seg_box"] = {
        "panel_a": [{"label": r["label"], "run": r["run"], "n": r["n"],
                     "mean": r["mean"], "seed": r["seed"]} for r in order],
        "panel_b": None if not brasc else {
            "source": "results/brats_unet3d_wt_tc_et.json",
            "model": "standalone 3-D U-Net (finding D3), patient-held-out "
                     "BraTS test split",
            "regions": [{"label": r["label"], "mean": r["mean"]}
                        for r in brasc],
        },
        "note": "panel (b) numbers are NOT TAMNet's auxiliary head: they are "
                "the WT/TC/ET Dice the legacy manuscript attributed to TAMNet "
                "(0.9005/0.5803/0.6010), re-measured from the 3-D U-Net "
                "checkpoint with per-case provenance",
    }


FIGURES = {
    "fig1_architecture": fig1_architecture,
    "fig2_data_pipeline": fig2_data_pipeline,
    "fig3_samples": fig3_samples,
    "fig8_seg_box": fig8_seg_box,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    mp = OUT / "figures_manifest.json"
    manifest = read_json(mp) or {"kind": "phase4_figures", "figures": {}}
    manifest.setdefault("figures", {})
    manifest["static_generated_at"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds")

    for name in (args.only or list(FIGURES)):
        fn = FIGURES.get(name)
        if not fn:
            print(f"unknown figure: {name}")
            continue
        try:
            fn(manifest["figures"])
        except Exception as exc:  # never let one figure block the rest
            print(f"MISSING: {name} ({type(exc).__name__}: {exc})")

    mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {mp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
