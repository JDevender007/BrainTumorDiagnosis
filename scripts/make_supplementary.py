"""Phase 4 — supplementary figures S1–S10 (``paper/figs/supp/``).

Every panel is drawn from an artifact under ``results/``. Panels whose
ingredient data was never saved are **not** drawn: they are recorded as
``MISSING`` in ``paper/figs/supp/manifest.json`` with the reason, per the
no-fabrication rule.

    python -X utf8 scripts/make_supplementary.py

| Supp | Content                              | Source |
|------|--------------------------------------|--------|
| S1   | validation curves per run            | ``results/phase3/runs/*/training_history.json`` |
| S2   | leakage / duplicate audit            | ``results/leakage_report.json`` |
| S3   | per-class ROC / PR                    | MISSING (per-sample probabilities not retained) |
| S4   | per-class precision/recall/F1        | ``eval_internal_test/metrics.json`` |
| S5   | ECE, NLL, error-detection AUROC       | ``uncertainty_summary.json`` |
| S6   | t-SNE / UMAP of fused features       | MISSING (needs an inference pass) |
| S7   | failure gallery (12 worst)            | MISSING (per-sample predictions not retained) |
| S8   | class counts per split                | ``split_manifest_clean.json`` |
| S9   | per-seed dots + mean ± SD            | ``agg/combined_summary.json`` |
| S10  | latency summary, single vs T=10 MC    | ``efficiency.json`` (summary statistics only) |
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))

from datasets.classification import CLASS_NAMES  # noqa: E402
RESULTS = ROOT / "results"
P3 = RESULTS / "phase3"
RUNS = P3 / "runs"
OUT = ROOT / "paper" / "figs" / "supp"
OUT.mkdir(parents=True, exist_ok=True)

MISSING: dict[str, str] = {}
WRITTEN: dict[str, list[str]] = {}

MODEL_ORDER = ("tamnet", "densenet121", "efficientnet_v2_s", "convnext_tiny",
                "swin_t", "resnet50", "vit_b_16", "shared_unet")
COLOR = {
    "tamnet": "#d62728", "densenet121": "#1f77b4",
    "efficientnet_v2_s": "#2ca02c", "convnext_tiny": "#9467bd",
    "swin_t": "#8c564b", "resnet50": "#e377c2", "vit_b_16": "#7f7f7f",
    "shared_unet": "#17becf",
}


def rd(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def save(fig, stem: str) -> list[str]:
    paths = []
    for ext in ("png", "pdf"):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        paths.append(str(p.name))
    plt.close(fig)
    WRITTEN.setdefault(stem, []).extend(paths)
    print(f"{stem}: wrote {paths[0]}")
    return paths


def runs_with(stem: str):
    for d in sorted(RUNS.iterdir()):
        f = d / stem
        if d.is_dir() and f.is_file():
            yield d, f


# ---------------------------------------------------------------- S1 curves
def s1_training_curves() -> None:
    series = {}
    for d, f in runs_with("training_history.json"):
        h = rd(f)
        cfg = rd(d / "run_config.json") or {}
        if not h:
            continue
        series[d.name] = (cfg.get("model", "?"),
                          [e.get("val_f1") for e in h],
                          [e.get("val_accuracy") for e in h],
                          cfg.get("seed"))
    if not series:
        MISSING["S1_training_curves"] = "no training_history.json found"
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
    for name, (model, f1, acc, seed) in series.items():
        x = np.arange(1, len(f1) + 1)
        c = COLOR.get(model, "#333")
        for ax, ys, lab in ((axes[0], f1, "val macro-F1"),
                            (axes[1], acc, "val accuracy")):
            ax.plot(x, ys, color=c, lw=0.8, alpha=0.55)
            ax.set_xlabel("epoch")
            ax.set_ylabel(lab)
            ax.grid(alpha=0.25, lw=0.5)
            ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], color=COLOR[m], lw=2, label=m)
               for m in MODEL_ORDER if any(m == v[0] for v in series.values())]
    axes[1].legend(handles=handles, fontsize=7, loc="lower right")
    fig.suptitle(f"S1 — validation curves, all {len(series)} training runs "
                 "(30 epochs each; colour = model)", fontsize=9.5)
    fig.tight_layout()
    save(fig, "S1_training_curves")


# ------------------------------------------------------------- S2 leakage
def s2_leakage() -> None:
    lr = rd(RESULTS / "leakage_report.json")
    if not lr:
        MISSING["S2_leakage"] = "results/leakage_report.json missing"
        return
    pairs = lr.get("pairs") or {}
    names, union = [], []
    for k in sorted(pairs, key=lambda k: -(pairs[k].get("n_near_duplicates_union")
                                            or 0)):
        names.append(k.replace("__", "\n"))
        union.append(pairs[k].get("n_near_duplicates_union") or 0)
    verdicts = (lr.get("verdicts") or {})
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2),
                             gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    ax.barh(range(len(names)), union, color="#d62728", alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlabel("perceptual near-duplicate pairs (pHash ∪ dHash)")
    ax.set_title("cross-split image overlap", fontsize=9)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)

    ax2 = axes[1]
    labels, vals = [], []
    for key, v in verdicts.items():
        if isinstance(v, dict) and "contamination_rate" in v:
            labels.append(key.replace("_", " "))
            vals.append(100.0 * v["contamination_rate"])
    if not vals:
        labels, vals = ["(no contamination verdicts)"], [0.0]
    ax2.bar(range(len(labels)), vals, color="#ff7f0e", alpha=0.85)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=18, ha="right", fontsize=7)
    ax2.set_ylabel("% of the external set inside internal train")
    ax2.set_ylim(0, 100)
    ax2.set_title("external-set contamination", fontsize=9)
    ax2.grid(axis="y", alpha=0.25, lw=0.5)
    ax2.set_axisbelow(True)
    fig.suptitle("S2 — leakage audit: duplicates and contamination before "
                 "de-duplication", fontsize=9.5)
    fig.tight_layout()
    save(fig, "S2_leakage_audit")


# ------------------------------------------------------- S4 per-class PRF
def s4_per_class() -> None:
    rows = {}
    for d, f in runs_with(Path("eval_internal_test") / "metrics.json"):
        m = rd(f)
        cfg = rd(d / "run_config.json") or {}
        if not m or "per_class" not in m:
            continue
        rows[d.name] = (cfg.get("model", "?"), m["per_class"])
    if not rows:
        MISSING["S4_per_class_prf"] = "no eval_internal_test/metrics.json"
        return
    classes = list(next(iter(rows.values()))[1].keys())
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9),
                             sharey=True)
    metrics = (("precision", "precision"), ("recall", "recall"),
               ("f1", "F1"))
    x = np.arange(len(classes))
    seen_models = []
    for ax, (key, title) in zip(axes, metrics):
        for name, (model, per_cls) in rows.items():
            if model not in seen_models:
                seen_models.append(model)
            vals = [per_cls[c][key] for c in classes if c in per_cls]
            ax.scatter(x, vals, s=14, color=COLOR.get(model, "#333"),
                       alpha=0.6, label=model)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=20, ha="right", fontsize=7.5)
        ax.set_title(f"test {title}", fontsize=9)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("score")
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLOR[m],
                          label=m) for m in MODEL_ORDER if m in seen_models]
    axes[-1].legend(handles=handles, fontsize=6.6, loc="lower right")
    fig.suptitle(f"S4 — per-class precision / recall / F1 on the clean test "
                 f"split ({len(rows)} runs)", fontsize=9.5)
    fig.tight_layout()
    save(fig, "S4_per_class_prf")


# ------------------------------------------------------- S5 calibration
def s5_calibration() -> None:
    u = rd(P3 / "uncertainty_summary.json")
    if not u or not u.get("groups"):
        MISSING["S5_calibration"] = "uncertainty_summary.json empty"
        return
    groups = u["groups"]
    keys = [("ece_15", "ECE (15 bins)"), ("nll", "NLL"),
            ("error_detection_auroc", "error-detection AUROC"),
            ("mean_predictive_entropy", "mean predictive entropy")]
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.2))
    x = np.arange(len(groups))
    for ax, (k, title) in zip(axes.ravel(), keys):
        vals, errs = [], []
        for g in groups:
            s = g["summary"].get(k)
            if isinstance(s, dict) and "mean" in s:
                vals.append(s["mean"])
                errs.append(s.get("std_sample_ddof1") or 0.0)
            else:
                vals.append(np.nan)
                errs.append(0.0)
        ax.bar(x, vals, yerr=errs, capsize=2.5, alpha=0.85,
               color=["#d62728" if "tamnet" in g["slug"] else "#1f77b4"
                      for g in groups])
        ax.set_xticks(x)
        ax.set_xticklabels([g["slug"].replace("_", "\n") for g in groups],
                           fontsize=6.0)
        ax.set_title(f"{title}  (mean ± SD over seeds)", fontsize=8.6)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.set_axisbelow(True)
    fig.suptitle(f"S5 — calibration and uncertainty per configuration "
                 f"(MC T={u.get('passes')}, BN eval, D5 fix)", fontsize=9.6)
    fig.tight_layout()
    save(fig, "S5_calibration_mc")


# ------------------------------------------------------- S8 class counts
def s8_class_counts() -> None:
    m = rd(P3 / "split_manifest_clean.json")
    stats = (m or {}).get("statistics", {}).get("per_split")
    if not stats:
        MISSING["S8_class_counts"] = "split_manifest_clean.json statistics missing"
        return
    splits = list(stats)
    classes = list(stats[splits[0]]["classes"])
    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    width = 0.27
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, sp in enumerate(splits):
        vals = [stats[sp]["classes"].get(c, 0) for c in classes]
        ax.bar(x + (i - 1) * width, vals, width, label=f"{sp} (n={stats[sp]['n']})",
               color=palette[i % len(palette)], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=8)
    ax.set_ylabel("images")
    ax.legend(fontsize=7.5)
    ax.set_title("S8 — class counts per split (manifest-pinned, "
                 "image-level)", fontsize=9.5)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "S8_class_counts")


# ------------------------------------------------------- S9 per-seed dots
def s9_seed_dots() -> None:
    agg = rd(P3 / "agg" / "combined_summary.json")
    groups = (agg or {}).get("groups") or (agg or {}).get("configs")
    if not groups:
        MISSING["S9_seed_dots"] = "agg/combined_summary.json has no groups"
        return
    if isinstance(groups, dict):
        groups = [dict(v, slug=k) for k, v in groups.items()]
    rows = []
    for g in groups:
        acc = g.get("summary", {}).get("accuracy") or g.get("test_accuracy") \
            or g.get("accuracy")
        seeds = g.get("per_seed_acc") or g.get("per_seed_test_accuracy") \
            or g.get("per_seed_accuracy")
        if not isinstance(acc, dict) or "mean" not in acc:
            continue
        rows.append((g.get("slug", "?"), acc, seeds))
    if not rows:
        MISSING["S9_seed_dots"] = "no test_accuracy block in aggregate"
        return
    rows.sort(key=lambda r: r[1]["mean"])
    fig, ax = plt.subplots(figsize=(7.4, 0.36 * len(rows) + 1.6))
    y = np.arange(len(rows))
    for i, (slug, acc, seeds) in enumerate(rows):
        ax.hlines(i, acc["mean"] - (acc.get("std_sample_ddof1") or 0),
                  acc["mean"] + (acc.get("std_sample_ddof1") or 0),
                  color="#999", lw=3, alpha=0.55)
        ax.plot([acc["mean"]], [i], "D", ms=5, color="#111")
        if isinstance(seeds, dict):
            for v in seeds.values():
                ax.plot([v], [i], "o", ms=3.0, color="#d62728", alpha=0.65)
        elif isinstance(seeds, list):
            for v in seeds:
                ax.plot([v], [i], "o", ms=3.0, color="#d62728", alpha=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.set_xlabel("test accuracy")
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title("S9 — per-seed dots (red) and mean ± SD (diamond, grey "
                 "band); n=1 groups have no band", fontsize=9.2)
    fig.tight_layout()
    save(fig, "S9_seed_dots")


# ------------------------------------------------------- S10 latency
def s10_latency() -> None:
    eff = rd(P3 / "efficiency.json")
    if not eff or not eff.get("models"):
        MISSING["S10_latency"] = "efficiency.json empty"
        return
    names = list(eff["models"])
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    x = np.arange(len(names))
    w = 0.38
    for i, key in enumerate(("single_pass", "mc_10_pass")):
        med = [eff["models"][n][key]["median_ms"] for n in names]
        p95 = [eff["models"][n][key]["p95_ms"] for n in names]
        hi = [max(0.0, p - m) for m, p in zip(med, p95)]
        ax.bar(x + (i - 0.5) * w, med, w, yerr=[[0] * len(names), hi],
               capsize=2.5, alpha=0.85,
               label=("single pass" if key == "single_pass" else "T=10 MC"),
               color=("#1f77b4" if key == "single_pass" else "#d62728"))
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=22, ha="right", fontsize=7.5)
    ax.set_ylabel("latency (ms), bar = median, whisker = p95")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_yscale("log")
    ax.set_title("S10 — measured inference latency (batch 1, 20 warm-up, "
                 "200 timed, log scale)", fontsize=9.4)
    fig.tight_layout()
    save(fig, "S10_latency_summary")


def _load_predictions() -> tuple[dict, np.ndarray | None] | None:
    """Saved probabilities for the headline run (phase4_save_predictions.py)."""
    base = P3 / "predictions" / "tamnet_clean_s42_test"
    if not (base.with_suffix(".npz").is_file() and base.with_suffix(".json").is_file()):
        return None
    d = np.load(base.with_suffix(".npz"), allow_pickle=False)
    return (json.loads(base.with_suffix(".json").read_text(encoding="utf-8")),
            d)


# ------------------------------------------------- S3 per-class ROC and PR
def s3_roc_pr() -> None:
    from sklearn.metrics import (auc, precision_recall_curve, roc_curve)

    loaded = _load_predictions()
    if loaded is None:
        MISSING["S3_per_class_roc_pr"] = (
            "predictions/tamnet_clean_s42_test.npz missing - run "
            "scripts/phase4_save_predictions.py")
        return
    meta, d = loaded
    y_true, probs = d["y_true"], d["probs"].astype(np.float64)
    classes = list(CLASS_NAMES)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, name in enumerate(classes):
        yb = (y_true == i).astype(int)
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        fpr, tpr, _ = roc_curve(yb, probs[:, i])
        prec, rec, _ = precision_recall_curve(yb, probs[:, i])
        axes[0].plot(fpr, tpr, color=palette[i % 4], lw=1.4,
                     label=f"{name} (AUC {auc(fpr, tpr):.3f})")
        axes[1].plot(rec, prec, color=palette[i % 4], lw=1.4,
                     label=f"{name} (AP {auc(rec, prec):.3f})")
    for ax, t, x, y in ((axes[0], "ROC", "false positive rate", "true positive rate"),
                        (axes[1], "precision–recall", "recall", "precision")):
        ax.plot([0, 1], [0, 1], ls="--", color="#999", lw=0.9)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"S3 — {t}, one-vs-rest", fontsize=9.4)
        ax.legend(fontsize=7, loc="lower right" if ax is axes[0] else "upper right")
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
    fig.suptitle(f"S4 view of the scores — {meta['run']} on the clean test "
                 f"split (n={meta['n']})", fontsize=9.5)
    fig.tight_layout()
    save(fig, "S3_per_class_roc_pr")


# ------------------------------------------------- S6 embedding / t-SNE
def s6_embedding() -> None:
    loaded = _load_predictions()
    if loaded is None or loaded[1]["features"].size == 0:
        MISSING["S6_tsne_umap"] = (
            "no fused-feature embeddings on disk - run "
            "scripts/phase4_save_predictions.py")
        return
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    meta, d = loaded
    feats = d["features"].astype(np.float64)
    y = d["y_true"]
    n = min(len(feats), 1500)          # t-SNE cost cap, deterministic head
    feats, y = feats[:n], y[:n]
    pca = PCA(n_components=min(30, feats.shape[1], feats.shape[0] - 1),
              random_state=42).fit_transform(feats)
    emb = TSNE(n_components=2, perplexity=30, init="pca",
               random_state=42).fit_transform(pca)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, name in enumerate(CLASS_NAMES):
        m = y == i
        ax.scatter(emb[m, 0], emb[m, 1], s=9, alpha=0.72,
                   color=palette[i % 4], label=f"{name} (n={int(m.sum())})")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=7.5, loc="best")
    ax.set_title(f"S6 — t-SNE of the fused embedding, first {n} test slices "
                 f"({meta['run']})", fontsize=9.4)
    fig.tight_layout()
    save(fig, "S6_tsne_fused_embedding")


# ------------------------------------------------- S7 failure gallery
def s7_failures() -> None:
    from PIL import Image

    loaded = _load_predictions()
    if loaded is None:
        MISSING["S7_failure_gallery"] = (
            "predictions not on disk - run scripts/phase4_save_predictions.py")
        return
    meta, d = loaded
    y_true, probs, paths = d["y_true"], d["probs"], d["paths"]
    conf_true = probs[np.arange(len(y_true)), y_true]
    order = np.argsort(conf_true)[:12]
    fig, axes = plt.subplots(3, 4, figsize=(8.6, 7.2))
    wrong = 0
    for ax, idx in zip(axes.ravel(), order):
        p = paths[idx]
        try:
            im = Image.open(p).convert("L").resize((160, 160))
        except OSError:
            ax.axis("off")
            continue
        pred = int(probs[idx].argmax())
        wrong += int(pred != int(y_true[idx]))
        ax.imshow(im, cmap="gray")
        c = "#d62728" if pred != y_true[idx] else "#2ca02c"
        ax.set_title(f"true {CLASS_NAMES[y_true[idx]]}\n"
                     f"pred {CLASS_NAMES[pred]} · p={conf_true[idx]:.2f}",
                     fontsize=6.4, color=c)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"S7 — 12 lowest-confidence test slices by true-class "
                 f"probability ({wrong} of them misclassified)", fontsize=9.4)
    fig.tight_layout()
    save(fig, "S7_failure_gallery")


def main() -> int:
    MISSING.clear()

    s1_training_curves()
    s2_leakage()
    s3_roc_pr()
    s4_per_class()
    s5_calibration()
    s6_embedding()
    s7_failures()
    s8_class_counts()
    s9_seed_dots()
    s10_latency()

    manifest = {
        "kind": "phase4_supplementary",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "directory": "paper/figs/supp",
        "written": WRITTEN,
        "missing": MISSING,
        "note": "Supplementary panels are supporting material; no citation in "
                "TAMNet_Paper.docx depends on them. MISSING entries are "
                "recorded rather than approximated.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                       encoding="utf-8")
    print(f"\nsupp: {len(WRITTEN)} panels written, {len(MISSING)} recorded MISSING")
    for k, v in MISSING.items():
        print(f"  MISSING {k}: {v}")
    print(f"manifest -> {(OUT / 'manifest.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
