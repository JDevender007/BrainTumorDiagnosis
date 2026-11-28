"""Phase 4 — Fig 7, qualitative segmentation panel (best / median / worst).

Selection is *rule-based*, never hand-picked:

  * runs are ranked by their recorded tumour-slice mean Dice in
    ``results/phase3/runs/<run>/eval_brisc_test/seg_metrics.json``;
  * the best-ranked TAMNet run and the best-ranked shared-U-Net run are drawn;
  * inside each run the three shown slices are the argmax, the value closest to
    the median, and the argmin of the per-slice Dice among slices that have a
    non-empty ground truth.

The predictions are re-computed with one forward pass per shown slice from the
recorded ``best.pt`` (checkpoint SHA-256 written into the artifact), so the
panel cannot drift from the numbers in Table VII.

    python -X utf8 scripts/phase4_seg_qualitative.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.phase3_evaluate import build_model, sha256_file  # noqa: E402
from scripts.phase3_train import load_brisc_split  # noqa: E402

RUNS = ROOT / "results" / "phase3" / "runs"
FIGS = ROOT / "paper" / "figs"
MANIFEST = FIGS / "figures_manifest.json"
RECORD = ROOT / "results" / "phase4_seg_qualitative.json"


def candidates(run: Path) -> list[dict]:
    """Per-slice records of a run, only slices with a non-empty ground truth."""
    f = run / "eval_brisc_test" / "seg_metrics.json"
    if not f.is_file():
        return []
    payload = json.loads(f.read_text(encoding="utf-8-sig"))
    return [r for r in payload.get("per_slice", [])
            if not r.get("empty_gt") and r.get("gt_pixels", 0) > 0]


def pick_three(rows: list[dict]) -> list[dict]:
    """best (argmax Dice), median-nearest, worst (argmin) — deterministic."""
    if not rows:
        return []
    ordered = sorted(rows, key=lambda r: r["dice"])
    vals = np.array([r["dice"] for r in ordered])
    med = float(np.median(vals))
    med_i = int(np.argmin(np.abs(vals - med)))
    picks = [ordered[-1], ordered[med_i], ordered[0]]
    # deduplicate if the run is tiny
    seen, out = set(), []
    for p in picks:
        if p["index"] not in seen:
            seen.add(p["index"])
            out.append(p)
    return out


def best_run(model: str) -> Path | None:
    best, best_mean = None, -1.0
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir():
            continue
        cfg = json.loads((d / "run_config.json").read_text(encoding="utf-8")) \
            if (d / "run_config.json").is_file() else {}
        if cfg.get("model") != model or not (d / "best.pt").is_file():
            continue
        rows = candidates(d)
        if not rows:
            continue
        mean = float(np.mean([r["dice"] for r in rows]))
        if mean > best_mean:
            best, best_mean = d, mean
    return best


def load_model(run: Path, device: torch.device):
    cfg = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    model = build_model(cfg)
    ckpt = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), cfg


def predict(model, sample_path: Path, device: torch.device) -> np.ndarray:
    from datasets.classification import BrainTumorClassificationDataset
    from datasets.normalization import build_classification_transform

    # rebuild the single sample through the same transform pipeline
    splits = load_brisc_split()
    match = [s for s in splits["test"]
             if Path(s.image_path).as_posix().endswith(sample_path.as_posix())]
    if not match:
        raise RuntimeError(f"sample not found in BRISC test split: {sample_path}")
    ds = BrainTumorClassificationDataset(
        match, transform=build_classification_transform((224, 224)))
    img = ds[0][0].unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img, return_seg=True)
    if "seg_logits" not in out:
        raise RuntimeError("model returned no seg_logits")
    return out["seg_logits"].argmax(dim=1).cpu().numpy()[0] > 0


def load_image(path: Path) -> np.ndarray:
    from PIL import Image
    arr = np.array(Image.open(path).convert("L").resize((224, 224)))
    return arr.astype(np.float32) / 256.0


def load_mask(path: Path) -> np.ndarray:
    from PIL import Image
    arr = np.array(Image.open(path).convert("L").resize((224, 224)))
    return arr > 127


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None,
                    help="explicit run names (else best TAMNet + best shared)")
    ap.add_argument("--force", action="store_true",
                    help="recompute even if fig7 + record already exist")
    args = ap.parse_args()

    if not args.force and (FIGS / "fig7_seg_qualitative.png").is_file() \
            and RECORD.is_file():
        print("fig7_seg_qualitative.png exists - use --force to recompute")
        return 0

    if args.runs:
        run_dirs = [RUNS / r for r in args.runs]
    else:
        run_dirs = [d for d in (best_run("tamnet"), best_run("shared_unet"))
                    if d is not None]
    if not run_dirs:
        print("MISSING: no BRISC run with eval_brisc_test/seg_metrics.json")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FIGS.mkdir(parents=True, exist_ok=True)

    picks: list[dict] = []
    for d in run_dirs:
        rows = candidates(d)
        if not rows:
            print(f"MISSING: {d.name} has no tumour-slice records")
            continue
        model, cfg = load_model(d, device)
        for label, row in zip(("best", "median", "worst"), pick_three(rows)):
            rel = Path(row["path"])
            pred = predict(model, rel, device)
            gt = load_mask(ROOT / rel)
            picks.append({
                "run": d.name,
                "model": cfg.get("model"),
                "seed": cfg.get("seed"),
                "seg_weight": cfg.get("seg_weight"),
                "role": label,
                "path": row["path"],
                "dice_recorded": row["dice"],
                "gt_pixels": row["gt_pixels"],
                "pred_pixels": int(pred.sum()),
                "image": load_image(ROOT / rel),
                "gt": gt,
                "pred": pred,
            })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not picks:
        print("MISSING: nothing to plot")
        return 1

    # ---------------- figure: rows = slices, cols = image / GT / pred / overlay
    n_rows, n_cols = len(picks), 4
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7.4, 1.72 * n_rows),
                             squeeze=False)
    for r, p in enumerate(picks):
        ax = axes[r][0]
        ax.imshow(p["image"], cmap="gray")
        ax.set_title(f"{p['run'].replace('_', ' ')}\n{p['role']} "
                     f"(Dice {p['dice_recorded']:.3f})",
                     fontsize=6.6, loc="left")
        axes[r][1].imshow(p["gt"], cmap="gray")
        axes[r][2].imshow(p["pred"], cmap="gray")
        ov = axes[r][3]
        ov.imshow(p["image"], cmap="gray")
        ov.imshow(np.ma.masked_where(~p["gt"], p["gt"]),
                  cmap="autumn", alpha=0.42)
        ov.imshow(np.ma.masked_where(~p["pred"], p["pred"]),
                  cmap="winter", alpha=0.42)
        if r == 0:
            for c, t in enumerate(("input slice", "ground truth",
                                   "prediction", "overlay: GT red / "
                                   "pred cyan")):
                axes[r][c].set_title(t, fontsize=7.2)
        for c in range(n_cols):
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])
            axes[r][c].grid(False)
    fig.suptitle("Qualitative segmentation on the BRISC 2025 test split — "
                 "best / median / worst slice by recorded Dice",
                 fontsize=8.6, y=1.004)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"fig7_seg_qualitative.{ext}", dpi=600,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"fig7: wrote {FIGS / 'fig7_seg_qualitative.png'}  "
          f"({len(picks)} slices)")

    # ---------------- manifest + machine-readable record
    if MANIFEST.is_file():
        man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    else:
        man = {"kind": "phase4_figures"}
    man.setdefault("figures", {})["fig7_seg_qualitative.png"] = {
        "runs": sorted({p["run"] for p in picks}),
        "selection": "argmax / median-nearest / argmin of per-slice Dice "
                     "among non-empty-GT slices; runs chosen by group mean",
        "checkpoints": {p["run"]: sha256_file(RUNS / p["run"] / "best.pt")
                        for p in picks},
        "slices": [{k: p[k] for k in ("run", "role", "path", "dice_recorded",
                                      "gt_pixels", "pred_pixels", "seed",
                                      "model", "seg_weight")} for p in picks],
    }
    MANIFEST.write_text(json.dumps(man, indent=2), encoding="utf-8")

    RECORD.write_text(json.dumps({
        "kind": "phase4_seg_qualitative",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": str(device),
        "source": "best.pt forward pass, return_seg=True, argmax over "
                  "seg_logits (same transform pipeline as phase3_seg_eval)",
        "slices": man["figures"]["fig7_seg_qualitative.png"]["slices"],
    }, indent=2), encoding="utf-8")
    print(f"record -> {RECORD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
