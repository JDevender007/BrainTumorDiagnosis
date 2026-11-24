"""Phase 4 figure generation.

Reads only files that already exist under results/phase3/runs/ and writes PNGs
plus a provenance manifest into paper/figs/.

Rules enforced here (audit requirement: no invented numbers):
  * every bar/cell comes from a saved JSON (training_result.json,
    eval_internal_test/metrics.json);
  * a config with n=1 seed is annotated "n=1" and gets no error bar - a single
    seed has no standard deviation;
  * std is the sample std (ddof=1) over seeds, only when n >= 2;
  * val and TEST numbers are never mixed inside one panel.

Usage:
    python -X utf8 scripts/make_figures.py            # all figures with data
    python -X utf8 scripts/make_figures.py --only fig4_ablation
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = Path("results/phase3/runs")
OUT = Path("paper/figs")

CLASS_ORDER_DEFAULT = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]


# --------------------------------------------------------------------------- io
def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def sha256_short(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def collect_runs() -> list[dict]:
    """One record per completed run, with whatever metrics exist on disk."""
    out = []
    if not RUNS.is_dir():
        return out
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir():
            continue
        cfg = read_json(d / "run_config.json")
        res = read_json(d / "training_result.json")
        if not cfg or not res:
            continue
        ev = read_json(d / "eval_internal_test" / "metrics.json")
        out.append({
            "dir": d,
            "name": d.name,
            "model": cfg.get("model"),
            "protocol": cfg.get("protocol"),
            "seed": cfg.get("seed"),
            "seg_weight": cfg.get("seg_weight"),
            "seg_classes": cfg.get("segmentation_classes"),
            "flags": cfg.get("flags") or {},
            "val_f1": res.get("best_val_f1_weighted"),
            "val_epoch": res.get("best_epoch"),
            "test": ev,
            "test_source": d / "eval_internal_test" / "metrics.json" if ev else None,
        })
    return out


def config_label(r: dict) -> str:
    """Human label for the configuration (seed-independent).

    On the BRISC split the sweep variable is the segmentation loss weight, so
    those runs are labelled by lambda - they are NOT replicates of one config.
    """
    m, f, p = r["model"], r["flags"], r["protocol"]
    if m == "tamnet" and p == "brisc":
        lam = r.get("seg_weight")
        return f"TAMNet λ={lam:g}" if isinstance(lam, (int, float)) else "TAMNet (BRISC)"
    if m == "tamnet":
        if f.get("fusion_mode") == "t":
            return "TAMNet t-only"
        if f.get("use_channel_attention"):
            return "TAMNet +MC"
        if not f.get("use_attention_fusion", True):
            return "TAMNet +concat"
        return "TAMNet full"
    if m == "shared_unet":
        return "shared U-Net"
    if m == "efficientnet_v2_s":
        return "EfficientNetV2-S"
    if m == "swin_t":
        return "Swin-T"
    if m == "resnet50":
        return "ResNet-50"
    if m == "convnext_tiny":
        return "ConvNeXt-T"
    if m == "densenet121":
        return "DenseNet-121"
    if m == "vit_b_16":
        return "ViT-B/16"
    return str(m)


def protocol_group(r: dict) -> str:
    return r["protocol"] or "clean"


# ----------------------------------------------------------------- fig 4 ablation
def variant_of(r: dict) -> str:
    """Runs that used a different training budget are not part of the ablation."""
    if "ep50budget" in r["name"]:
        return "ep50"
    return ""


def fig4_ablation(runs: list[dict], manifest: dict) -> None:
    """Mean +- sample std of the selection metric (val F1w), split by protocol.

    Clean and BRISC are different splits, and leaky shares images across
    train/test, so they are never plotted on the same axis.
    """
    excluded = []
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in runs:
        if r["val_f1"] is None:
            continue
        if variant_of(r):
            excluded.append({"run": r["name"], "reason": f"variant={variant_of(r)}"})
            continue
        if protocol_group(r) == "leaky":
            excluded.append({"run": r["name"], "reason": "leaky split (leaks train->test)"})
            continue
        groups[(config_label(r), protocol_group(r))].append(float(r["val_f1"]))

    if not groups:
        print("fig4: no runs with val_f1, skipped")
        return

    rows = []
    for (label, proto), vals in groups.items():
        arr = np.asarray(vals, dtype=float)
        rows.append({
            "label": label,
            "protocol": proto,
            "n": len(arr),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        })

    panels = [("clean", "Internal split (clean protocol)"),
              ("brisc", "BRISC segmentation split")]
    panels = [(p, t) for p, t in panels if any(r["protocol"] == p for r in rows)]

    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.2),
                             squeeze=False)
    for ax, (proto, title) in zip(axes.ravel(), panels):
        sub = sorted([r for r in rows if r["protocol"] == proto],
                     key=lambda x: x["mean"])
        y = np.arange(len(sub))
        palette = {"TAMNet": "#2f6fb3", "backbone": "#c8642a", "other": "#5a5a5a"}

        def kind(label: str) -> str:
            if label.startswith("TAMNet"):
                return "TAMNet"
            if label == "shared U-Net":
                return "other"
            return "backbone"

        ax.barh(y, [r["mean"] for r in sub], xerr=[r["std"] for r in sub],
                color=[palette[kind(r["label"])] for r in sub],
                error_kw={"ecolor": "#222", "capsize": 3, "lw": 1}, height=0.62)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r['label']} (n={r['n']})" for r in sub], fontsize=8)
        lo = min(r["mean"] for r in sub)
        ax.set_xlim(lo - 0.03, 1.0)
        for yi, r in zip(y, sub):
            ax.text(r["mean"] + r["std"] + 0.003, yi, f"{r['mean']:.4f}",
                    va="center", fontsize=7.5)
        ax.set_xlabel("val weighted F1 at best epoch, mean ± sample SD", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="x", alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)

    fig.suptitle("Ablation / baseline comparison (validation metric; "
                 "n=1 entries have no SD)", fontsize=10)
    fig.tight_layout()
    p = OUT / "fig4_ablation_val.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"fig4: wrote {p} ({len(rows)} configs, {len(excluded)} excluded)")
    manifest["fig4_ablation_val.png"] = {
        "metric": "best_val_f1_weighted from training_result.json",
        "configs": rows,
        "excluded_runs": excluded,
        "note": "TEST-set version pending; only configs with eval_internal_test are eligible",
    }


# ------------------------------------------------------------------ fig 6 confusion
def fig6_confusion(runs: list[dict], manifest: dict) -> None:
    evs = [r for r in runs if r["test"] and r["test"].get("confusion_matrix")]
    if not evs:
        print("fig6: no eval metrics, skipped")
        return

    n = len(evs)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.2 * rows), squeeze=False)
    used_order = evs[0]["test"].get("confusion_matrix_order") or CLASS_ORDER_DEFAULT
    for ax, r in zip(axes.ravel(), evs):
        cm = np.asarray(r["test"]["confusion_matrix"], dtype=float)
        order = r["test"].get("confusion_matrix_order") or used_order
        norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                        fontsize=7, color="white" if norm[i, j] > 0.55 else "#222")
        short = [c.replace("No Tumor", "None") for c in order]
        ax.set_xticks(range(len(short)))
        ax.set_yticks(range(len(short)))
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(short, fontsize=7)
        ax.set_title(f"{r['name']}\nacc {r['test'].get('accuracy'):.4f}", fontsize=8)
        if ax is axes.ravel()[0]:
            ax.set_ylabel("true", fontsize=8)
            ax.set_xlabel("predicted", fontsize=8)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02,
                 label="row-normalized rate")
    fig.suptitle("Internal test confusion matrices (counts annotated)", fontsize=10)
    fig.tight_layout()
    p = OUT / "fig6_confusion_internal.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"fig6: wrote {p} ({n} runs)")
    manifest["fig6_confusion_internal.png"] = {
        "runs": [{"run": r["name"], "n": r["test"].get("n"),
                  "accuracy": r["test"].get("accuracy"),
                  "source": str(r["test_source"])} for r in evs],
    }


# ------------------------------------------------------- fig 6b per-class F1 spread
def fig6b_per_class(runs: list[dict], manifest: dict) -> None:
    evs = [r for r in runs if r["test"] and r["test"].get("per_class")]
    if len(evs) < 2:
        print("fig6b: needs >=2 evaluated runs, skipped")
        return
    per: dict[str, list[float]] = defaultdict(list)
    for r in evs:
        pc = r["test"]["per_class"]
        items = pc.items() if isinstance(pc, dict) else []
        for cls, m in items:
            f1 = m.get("f1") if isinstance(m, dict) else None
            if isinstance(f1, (int, float)):
                per[cls].append(float(f1))
    if not per:
        print("fig6b: no per-class f1 found, skipped")
        return

    classes = list(per.keys())
    means = [float(np.mean(per[c])) for c in classes]
    stds = [float(np.std(per[c], ddof=1)) if len(per[c]) > 1 else 0.0 for c in classes]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(classes))
    ax.bar(x, means, yerr=stds, color="#2f6fb3", width=0.6,
           error_kw={"ecolor": "#222", "capsize": 3, "lw": 1})
    for xi, m in zip(x, means):
        ax.text(xi, m + 0.01, f"{m:.4f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=8)
    ax.set_ylim(min(means) - 0.08, 1.0)
    ax.set_ylabel("test F1")
    ax.set_title(f"Per-class test F1 over {len(evs)} seeds (mean ± sample SD)", fontsize=9)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = OUT / "fig6b_per_class_f1.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"fig6b: wrote {p}")
    manifest["fig6b_per_class_f1.png"] = {
        "runs": [r["name"] for r in evs],
        "values": {c: per[c] for c in classes},
    }


FIGURES = {
    "fig4_ablation": fig4_ablation,
    "fig6_confusion": fig6_confusion,
    "fig6b_per_class": fig6b_per_class,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="subset of figure names")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    runs = collect_runs()
    print(f"collect: {len(runs)} completed runs, "
          f"{sum(1 for r in runs if r['test'])} with TEST eval")

    manifest: dict = {
        "kind": "phase4_figures",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_runs": len(runs),
        "n_runs_with_test_eval": sum(1 for r in runs if r["test"]),
        "input_files_sha256_16": {
            str(r["test_source"]): sha256_short(r["test_source"])
            for r in runs if r["test_source"]
        },
        "figures": {},
    }

    names = args.only or list(FIGURES)
    for name in names:
        fn = FIGURES.get(name)
        if not fn:
            print(f"unknown figure: {name}")
            continue
        fn(runs, manifest["figures"])

    mp = OUT / "figures_manifest.json"
    mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {mp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
