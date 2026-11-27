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
    fig.savefig(OUT / "fig4_ablation_val.pdf", bbox_inches="tight")
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
    # one panel per configuration, not one per seed: 41 seeds would be unreadable
    curated = (
        "tamnet_clean_s42",
        "densenet121_clean_s42",
        "efficientnet_v2_s_clean_s42",
        "swin_t_clean_s42",
    )
    evs = [r for r in runs if r["test"] and r["test"].get("confusion_matrix")
           and r["name"] in curated]
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
    fig.savefig(OUT / "fig6_confusion_internal.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"fig6: wrote {p} ({n} runs)")
    manifest["fig6_confusion_internal.png"] = {
        "runs": [{"run": r["name"], "n": r["test"].get("n"),
                  "accuracy": r["test"].get("accuracy"),
                  "source": str(r["test_source"])} for r in evs],
    }


# ------------------------------------------------------- fig 6b per-class F1 spread
def fig6b_per_class(runs: list[dict], manifest: dict) -> None:
    # per-class F1 of ONE configuration over its seeds (never a mix of protocols)
    evs = [r for r in runs if r["test"] and r["test"].get("per_class")
           and r["name"].startswith("tamnet_clean_s")
           and "+" not in r["name"] and "-" not in r["name"]
           and not r["name"].endswith("budget")]
    if len(evs) < 2:
        print("fig6b: needs >=2 evaluated TAMNet seeds, skipped")
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
    fig.savefig(OUT / "fig6b_per_class_f1.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"fig6b: wrote {p}")
    manifest["fig6b_per_class_f1.png"] = {
        "runs": [r["name"] for r in evs],
        "values": {c: per[c] for c in classes},
    }


# --------------------------------------------------------------- shared helpers
AGG = Path("results/phase3/agg/combined_summary.json")
PARAMS = Path("results/phase3/params_table.json")
EXTERNAL = Path("results/phase3/external_eval_summary.json")
UNCERTAINTY = Path("results/phase3/uncertainty_summary.json")
ATTENTION = Path("results/phase3/attention_alignment.json")
ROBUSTNESS = Path("results/phase3/robustness.json")
RELIABILITY = Path("results/phase3/reliability.json")

PAPER_CLAIMS = {
    "internal_accuracy": 0.9645,
    "kaggle_external": 0.9916,
    "brisc_external": 0.9770,
    "params": 9034416,
    "latency_ms": 48.63,
}

COLOR = {
    "tamnet": "#2f6fb3",
    "backbone": "#c8642a",
    "unet": "#5a5a5a",
    "reference": "#6b6b6b",
}


def savefig_both(fig, stem: str) -> list[str]:
    """Vector PDF for the paper + 600-dpi PNG for review."""
    OUT.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", **kw)
        paths.append(str(p))
    return paths


def agg_groups() -> list[dict]:
    data = read_json(AGG)
    return (data or {}).get("groups") or []


def kind_of(slug: str) -> str:
    if slug.startswith("tamnet"):
        return "tamnet"
    if "shared_unet" in slug:
        return "unet"
    return "backbone"


# ------------------------------------------------- fig 4b TEST accuracy by config
def fig4b_test_accuracy(runs: list[dict], manifest: dict) -> None:
    """Headline figure: TEST accuracy per configuration, mean over seeds ± SD.

    Source: results/phase3/agg/combined_summary.json (one row per seed derived
    from that run's eval_internal_test/metrics.json).
    """
    groups = agg_groups()
    if not groups:
        print("fig4b: no aggregation file, skipped")
        return
    groups = sorted(groups, key=lambda g: g["summary"]["accuracy"]["mean"])
    y = np.arange(len(groups))
    means = [g["summary"]["accuracy"]["mean"] for g in groups]
    stds = [g["summary"]["accuracy"]["std_sample_ddof1"] or 0.0 for g in groups]
    labels = [f"{g['slug']} (n={g['n_seeds']})" for g in groups]

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    ax.barh(y, means, xerr=stds,
            color=[COLOR[kind_of(g["slug"])] for g in groups],
            error_kw={"ecolor": "#222", "capsize": 2.5, "lw": 0.9}, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    lo = min(m - s for m, s in zip(means, stds))
    ax.set_xlim(lo - 0.02, 1.0)
    for yi, g, m, s in zip(y, groups, means, stds):
        ax.text(m + s + 0.003, yi, f"{m:.4f}", va="center", fontsize=7)
    ax.axvline(PAPER_CLAIMS["internal_accuracy"], ls="--", lw=1,
               color=COLOR["reference"])
    ax.text(PAPER_CLAIMS["internal_accuracy"], len(groups) - 0.4,
            " paper: 0.9645", fontsize=7, color=COLOR["reference"], rotation=90,
            va="top")
    ax.set_xlabel("internal TEST accuracy (mean over seeds; error bar = sample SD, n≥2)", fontsize=8)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    paths = savefig_both(fig, "fig4b_test_accuracy")
    plt.close(fig)
    print(f"fig4b: wrote {paths[0]} ({len(groups)} configs)")
    manifest["fig4b_test_accuracy"] = {
        "source": str(AGG),
        "rule": "mean over seeds, sample SD ddof=1; n=1 has no SD",
        "values": {g["slug"]: g["summary"]["accuracy"] for g in groups},
        "paper_claim_line": PAPER_CLAIMS["internal_accuracy"],
    }


# ------------------------------------------------------------ fig 5 params/accuracy
def fig5_pareto(runs: list[dict], manifest: dict) -> None:
    """Accuracy vs parameter count — the paper's 'efficiency' claim, measured."""
    params = read_json(PARAMS)
    groups = agg_groups()
    if not params or not groups:
        print("fig5: missing params_table or aggregation, skipped")
        return
    key_map = {
        "tamnet_clean": "tamnet_full",
        "tamnet_clean_mc": "tamnet_mc",
        "tamnet_clean_concat": "tamnet_concat",
        "tamnet_clean_tonly": "tamnet_tonly",
        "densenet121_clean": "densenet121",
        "resnet50_clean": "resnet50",
        "efficientnet_v2_s_clean": "efficientnet_v2_s",
        "convnext_tiny_clean": "convnext_tiny",
        "swin_t_clean": "swin_t",
        "vit_b_16_clean": "vit_b_16",
    }
    pts = []
    for g in groups:
        key = key_map.get(g["slug"])
        if not key or key not in params["params"]:
            continue
        info = params["params"][key]
        pts.append({
            "slug": g["slug"],
            "params_m": info.get("params_millions") or info.get("total", 0) / 1e6,
            "accuracy": g["summary"]["accuracy"]["mean"],
            "std": g["summary"]["accuracy"]["std_sample_ddof1"] or 0.0,
            "kind": kind_of(g["slug"]),
        })
    if not pts:
        print("fig5: no joinable configs, skipped")
        return

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for p in pts:
        ax.errorbar(p["params_m"], p["accuracy"], yerr=p["std"], fmt="o",
                    ms=6, color=COLOR[p["kind"]], capsize=2.5, lw=0.9)
        ax.annotate(p["slug"].replace("_clean", ""), (p["params_m"], p["accuracy"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=6.5)
    ax.axhline(PAPER_CLAIMS["internal_accuracy"], ls="--", lw=0.9, color=COLOR["reference"])
    ax.set_xscale("log")
    ax.set_xlabel("parameters (millions, log scale)", fontsize=8)
    ax.set_ylabel("internal TEST accuracy (mean over seeds)", fontsize=8)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    paths = savefig_both(fig, "fig5_pareto_params_accuracy")
    plt.close(fig)
    print(f"fig5: wrote {paths[0]} ({len(pts)} points)")
    manifest["fig5_pareto_params_accuracy"] = {
        "sources": [str(PARAMS), str(AGG)],
        "points": pts,
    }


# ------------------------------------------------------- fig 9 reliability diagrams
def fig9_reliability(runs: list[dict], manifest: dict) -> None:
    """Reliability diagrams from scripts/phase3_calibration_curves.py output.

    Each entry provides equal-width confidence bins computed from the MC-mean
    predictive probabilities (T=10, BN frozen).  If the file is absent the
    figure is reported MISSING rather than drawn from the paper's old numbers.
    """
    data = read_json(RELIABILITY)
    if not data or not data.get("curves"):
        print("fig9: MISSING results/phase3/reliability.json "
              "(run scripts/phase3_calibration_curves.py)")
        manifest["fig9_reliability"] = {"status": "MISSING", "source": str(RELIABILITY)}
        return
    curves = data["curves"]
    n = len(curves)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], ls=":", color="#888", lw=1, label="perfect")
    for name, c in curves.items():
        xs = [b["mean_confidence"] for b in c["bins"] if b["n"] > 0]
        ys = [b["accuracy"] for b in c["bins"] if b["n"] > 0]
        ax.plot(xs, ys, marker="o", ms=3.5, lw=1.1,
                label=f"{name} (ECE {c['ece']:.4f})")
    ax.set_xlabel("predicted confidence (MC mean, T=10)", fontsize=8)
    ax.set_ylabel("empirical accuracy", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6.5, frameon=False)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    paths = savefig_both(fig, "fig9_reliability")
    plt.close(fig)
    print(f"fig9: wrote {paths[0]} ({n} curves)")
    manifest["fig9_reliability"] = {"source": str(RELIABILITY),
                                    "ece": {k: v["ece"] for k, v in curves.items()}}


# ------------------------------------------------ fig 10 uncertainty / risk-coverage
def fig10_uncertainty(runs: list[dict], manifest: dict) -> None:
    data = read_json(UNCERTAINTY)
    if not data or not data.get("groups"):
        print("fig10: MISSING uncertainty_summary.json, skipped")
        manifest["fig10_uncertainty"] = {"status": "MISSING"}
        return
    groups = data["groups"]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.4))

    # (a) ECE and error-detection AUROC per group
    ax = axes[0]
    names = [g["slug"] for g in groups]
    ece = [g["summary"].get("ece_15", {}).get("mean") for g in groups]
    ax.barh(np.arange(len(names)), ece, color=[COLOR[kind_of(n)] for n in names], height=0.6)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=6.5)
    ax.set_xlabel("ECE (15 bins, MC T=10)", fontsize=8)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)

    # (b) AUROC of MI for separating errors
    ax = axes[1]
    auroc = [g["summary"].get("error_detection_auroc", {}).get("mean") for g in groups]
    ax.barh(np.arange(len(names)), auroc, color=[COLOR[kind_of(n)] for n in names], height=0.6)
    ax.axvline(0.5, ls="--", lw=0.9, color="#888")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels([])
    ax.set_xlim(0.4, 0.8)
    ax.set_xlabel("AUROC: MI separates errors", fontsize=8)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)

    # (c) selective accuracy (risk-coverage) for the best-tested groups
    ax = axes[2]
    curves = read_json(RELIABILITY)
    sel = (curves or {}).get("selective") or {}
    if sel:
        for name, c in sel.items():
            xs = [1.0 - float(k) for k in c["coverage"]]
            ax.plot(xs, c["accuracy"], marker="o", ms=3.5, lw=1.1, label=name)
        ax.set_xlabel("fraction of samples rejected (1 − coverage)", fontsize=8)
        ax.set_ylabel("accuracy on the retained set", fontsize=8)
        ax.legend(fontsize=6.5, frameon=False)
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "risk–coverage: MISSING\n(run phase3_calibration_curves.py)",
                ha="center", va="center", fontsize=8, color="#666")

    fig.tight_layout()
    paths = savefig_both(fig, "fig10_uncertainty")
    plt.close(fig)
    print(f"fig10: wrote {paths[0]}")
    manifest["fig10_uncertainty"] = {"source": str(UNCERTAINTY), "groups": names}


# ----------------------------------------------------- fig 11/12 attention alignment
def fig11_attention(runs: list[dict], manifest: dict) -> None:
    data = read_json(ATTENTION)
    if not data:
        print("fig11: MISSING attention_alignment.json, skipped")
        manifest["fig11_attention"] = {"status": "MISSING"}
        return
    keys = [k for k in ("tamnet_attention", "densenet121_gradcam", "uniform")
            if data.get(k)]
    if not keys:
        print("fig11: no map summaries, skipped")
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    ax = axes[0]
    mass = [data[k]["mass_in_mask_mean"] for k in keys]
    err = [data[k]["mass_in_mask_std"] or 0.0 for k in keys]
    ax.bar(np.arange(len(keys)), mass, yerr=err, width=0.55,
           color=["#2f6fb3", "#c8642a", "#9a9a9a"][:len(keys)],
           error_kw={"ecolor": "#222", "capsize": 3, "lw": 0.9})
    for i, (k, m) in enumerate(zip(keys, mass)):
        ax.text(i, m + 0.01, f"{m:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels([k.replace("_", " ") for k in keys], fontsize=7.5)
    ax.set_ylabel("attention mass inside tumour mask", fontsize=8)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)

    ax = axes[1]
    hit = [data[k]["pointing_hit_rate"] for k in keys]
    ax.bar(np.arange(len(keys)), hit, width=0.55,
           color=["#2f6fb3", "#c8642a", "#9a9a9a"][:len(keys)])
    for i, h in enumerate(hit):
        ax.text(i, h + 0.01, f"{h:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels([k.replace("_", " ") for k in keys], fontsize=7.5)
    ax.set_ylabel("pointing-game hit rate", fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    paths = savefig_both(fig, "fig11_attention_alignment")
    plt.close(fig)
    fw = data.get("fusion_weights") or {}
    print(f"fig11: wrote {paths[0]} (alpha_tumour={fw.get('alpha_tumor_mean')})")
    manifest["fig11_attention_alignment"] = {
        "source": str(ATTENTION),
        "summaries": {k: data[k] for k in keys},
        "paired_tamnet_minus_gradcam": data.get("paired_tamnet_minus_gradcam"),
        "fusion_weights": fw,
    }


# ------------------------------------------------------ fig 13 corruption heatmap
def fig13_robustness(runs: list[dict], manifest: dict) -> None:
    data = read_json(ROBUSTNESS)
    if not data or not data.get("runs"):
        print("fig13: MISSING robustness.json, skipped")
        manifest["fig13_robustness"] = {"status": "MISSING"}
        return
    per_type: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in data["runs"].values():
        for c in r["corruptions"]:
            per_type[c["corruption"]][c["severity"]].append(c["accuracy"])
    types = sorted(per_type)
    sev = sorted({s for v in per_type.values() for s in v})
    mat = np.array([[np.mean(per_type[t][s]) if per_type[t].get(s) else np.nan for s in sev]
                    for t in types])

    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isnan(mat[i, j]):
                continue
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if mat[i, j] < 0.6 else "#111")
    ax.set_xticks(range(len(sev)))
    ax.set_xticklabels([f"s{s}" for s in sev], fontsize=8)
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels(types, fontsize=7.5)
    ax.set_xlabel("severity", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, label="accuracy")
    fig.tight_layout()
    paths = savefig_both(fig, "fig13_robustness_heatmap")
    plt.close(fig)
    print(f"fig13: wrote {paths[0]} ({len(types)}x{len(sev)})")
    manifest["fig13_robustness_heatmap"] = {
        "source": str(ROBUSTNESS),
        "mrrs_definition": data.get("mrrs_definition"),
        "groups": data.get("groups"),
        "types": types,
    }


# ------------------------------------------------------- fig 14 intensity curves
def fig14_intensity(runs: list[dict], manifest: dict) -> None:
    data = read_json(ROBUSTNESS)
    curves = (data or {}).get("intensity_curves") or {}
    if not curves:
        print("fig14: MISSING intensity curves, skipped")
        manifest["fig14_intensity"] = {"status": "MISSING"}
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3))
    for ax, key, xlabel in ((axes[0], "contrast", "contrast multiplier"),
                            (axes[1], "brightness", "brightness multiplier")):
        for i, (name, c) in enumerate(curves.items()):
            xs = [p["multiplier"] for p in c[key]]
            ys = [p["accuracy"] for p in c[key]]
            ax.plot(xs, ys, marker="o", ms=3.5, lw=1.1,
                    color=["#2f6fb3", "#c8642a", "#5a5a5a"][i % 3], label=name)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("TEST accuracy", fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].axvline(0.5, ls="--", lw=0.9, color="#888")
    axes[0].text(0.51, 0.05, "paper: contrast 0.5\n→ 22.40%", fontsize=6.5, color="#555")
    axes[0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    paths = savefig_both(fig, "fig14_intensity_curves")
    plt.close(fig)
    print(f"fig14: wrote {paths[0]}")
    manifest["fig14_intensity_curves"] = {
        "source": str(ROBUSTNESS),
        "runs": list(curves),
        "paper_claim": "contrast 0.5 -> 22.40% accuracy",
    }


# --------------------------------------------------- fig 15 train x test matrix
def fig15_generalization(runs: list[dict], manifest: dict) -> None:
    """Train-source × test-set accuracy matrix from on-disk eval directories."""
    eval_dirs = ["eval_internal_test", "eval_kaggle_testing_external",
                 "eval_kaggle_testing_external_dedup", "eval_brisc_test_external",
                 "eval_brisc_test_external_dedup"]
    col_names = ["internal TEST", "Kaggle full", "Kaggle de-dup",
                 "BRISC full", "BRISC de-dup"]

    def row_value(names: list[str]) -> list[float | None]:
        out = []
        for ed in eval_dirs:
            vals = []
            for n in names:
                m = read_json(RUNS / n / ed / "metrics.json")
                if m and m.get("accuracy") is not None:
                    vals.append(float(m["accuracy"]))
            out.append(float(np.mean(vals)) if vals else None)
        return out

    rows = {
        "TAMNet (internal train)": [f"tamnet_clean_s{s}" for s in (42, 43, 44, 45, 46)],
        "BRISC-trained TAMNet": ["tamnet_brisc_s42+ls0.25+sc2"],
        "BRISC-trained shared U-Net": [f"shared_unet_brisc_s{s}+ls0.25+sc2" for s in (42, 43, 44, 45)],
        "Leaky control (historical)": ["tamnet_leaky_s42"],
    }
    labels, matrix = [], []
    for label, names in rows.items():
        vals = row_value(names)
        if any(v is not None for v in vals):
            labels.append(label)
            matrix.append(vals)
    if not matrix:
        print("fig15: no eval directories found, skipped")
        manifest["fig15_generalization"] = {"status": "MISSING"}
        return

    mat = np.array([[v if v is not None else np.nan for v in row] for row in matrix])
    fig, ax = plt.subplots(figsize=(6.6, 0.8 + 0.55 * len(labels)))
    im = ax.imshow(mat, cmap="Blues", vmin=0.85, vmax=1.0, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isnan(mat[i, j]):
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="#999")
            else:
                ax.text(j, i, f"{mat[i, j]:.4f}", ha="center", va="center", fontsize=8,
                        color="white" if mat[i, j] > 0.95 else "#111")
    ax.set_xticks(range(len(col_names)))
    ax.set_xticklabels(col_names, fontsize=7.5, rotation=20, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.5)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="accuracy")
    fig.tight_layout()
    paths = savefig_both(fig, "fig15_generalization_matrix")
    plt.close(fig)
    print(f"fig15: wrote {paths[0]} ({len(labels)} train sources)")
    manifest["fig15_generalization_matrix"] = {
        "columns": col_names,
        "rows": {lab: {c: (None if v is None else v) for c, v in zip(col_names, vals)}
                 for lab, vals in zip(labels, matrix)},
    }


FIGURES = {
    "fig4_ablation": fig4_ablation,
    "fig4b_test_accuracy": fig4b_test_accuracy,
    "fig5_pareto": fig5_pareto,
    "fig6_confusion": fig6_confusion,
    "fig6b_per_class": fig6b_per_class,
    "fig9_reliability": fig9_reliability,
    "fig10_uncertainty": fig10_uncertainty,
    "fig11_attention": fig11_attention,
    "fig13_robustness": fig13_robustness,
    "fig14_intensity": fig14_intensity,
    "fig15_generalization": fig15_generalization,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="subset of figure names")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    runs = collect_runs()
    print(f"collect: {len(runs)} completed runs, "
          f"{sum(1 for r in runs if r['test'])} with TEST eval")

    mp = OUT / "figures_manifest.json"
    manifest: dict = read_json(mp) or {}
    manifest.update({
        "kind": "phase4_figures",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_runs": len(runs),
        "n_runs_with_test_eval": sum(1 for r in runs if r["test"]),
        "input_files_sha256_16": {
            str(r["test_source"]): sha256_short(r["test_source"])
            for r in runs if r["test_source"]
        },
        "figures": manifest.get("figures") or {},
    })

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
