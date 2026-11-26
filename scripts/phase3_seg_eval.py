"""Phase 3E — segmentation quality of the *auxiliary* branch actually trained here.

Two different segmentation numbers live in this repo and the paper conflates
them (finding D3):

* ``0.9005 / 0.5803 / 0.6010`` (WT / TC / ET) are from a **standalone 3-D
  U-Net** trained on BraTS volumes (``checkpoints/segmentation/unet3d``), not
  from TAMNet's 2-D auxiliary head.
* TAMNet's own head had ``L_seg ≡ 0`` in every historical run (finding D2),
  because no runner ever passed ``seg_targets``.

The clean protocol (Kaggle-style 2-D slices) has no masks, so the only runs
where the aux head actually received gradient are the BRISC λ_seg sweep and the
shared-UNet baseline (``--protocol brisc``, aligned masks from
``segmentation_task/<split>/masks``).  This script measures that head honestly:

    python -X utf8 scripts/phase3_seg_eval.py --dry-run
    python -X utf8 scripts/phase3_seg_eval.py

Writes ``results/phase3/runs/<run>/eval_brisc_test/seg_metrics.json`` and
``results/phase3/seg_metrics_summary.json`` (mean ± sample SD over seeds).

Metric definitions are stated explicitly in the artifact: foreground Dice is
the tumour-vs-background Dice; ``empty_gt_empty_pred`` is the fraction of
no-tumour slices for which the head correctly predicts an empty mask (Dice is
undefined there and conventionally scored as 1, which is why the tumour-only
number is reported separately).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets.classification import CLASS_NAMES  # noqa: E402
from scripts.phase3_evaluate import build_model, sha256_file  # noqa: E402
from scripts.phase3_train import load_brisc_split  # noqa: E402

RUNS = ROOT / "results" / "phase3" / "runs"
SUMMARY = ROOT / "results" / "phase3" / "seg_metrics_summary.json"

DEFAULT_RUNS = (
    "tamnet_brisc_s42+sc2",
    "tamnet_brisc_s42+ls0.1+sc2",
    "tamnet_brisc_s42+ls0.25+sc2",
    "tamnet_brisc_s42+ls1+sc2",
    "shared_unet_brisc_s42+ls0.25+sc2",
    "shared_unet_brisc_s43+ls0.25+sc2",
    "shared_unet_brisc_s44+ls0.25+sc2",
    "shared_unet_brisc_s45+ls0.25+sc2",
)

GROUPS = (
    ("tamnet_brisc_lambda0", ("tamnet_brisc_s42+sc2",)),
    ("tamnet_brisc_lambda0.1", ("tamnet_brisc_s42+ls0.1+sc2",)),
    ("tamnet_brisc_lambda0.25", ("tamnet_brisc_s42+ls0.25+sc2",)),
    ("tamnet_brisc_lambda1", ("tamnet_brisc_s42+ls1+sc2",)),
    ("shared_unet_brisc",
     ("shared_unet_brisc_s42+ls0.25+sc2", "shared_unet_brisc_s43+ls0.25+sc2",
      "shared_unet_brisc_s44+ls0.25+sc2", "shared_unet_brisc_s45+ls0.25+sc2")),
)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.sum(), gt.sum()
    if p == 0 and g == 0:
        return 1.0
    if p + g == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / (p + g))


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


@torch.no_grad()
def evaluate(run_dir: Path, samples, device: torch.device, batch: int) -> dict:
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    ckpt_path = run_dir / "best.pt"
    model = build_model(run_config)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    if not hasattr(model, "seg_head") or model.seg_head is None:
        raise RuntimeError("run has no segmentation head")

    from datasets.classification import BrainTumorClassificationDataset
    from datasets.normalization import build_classification_transform

    ds = BrainTumorClassificationDataset(
        samples, transform=build_classification_transform((224, 224))
    )
    rows = []
    for i in range(0, len(ds), batch):
        items = [ds[j] for j in range(i, min(i + batch, len(ds)))]
        images = torch.stack([it[0] for it in items]).to(device)
        masks = torch.stack([it[2] for it in items]).numpy()
        out = model(images, return_seg=True)
        if "seg_logits" not in out:
            raise RuntimeError("model returned no seg_logits")
        seg = out["seg_logits"]
        pred = seg.argmax(dim=1).cpu().numpy()
        for k in range(pred.shape[0]):
            gt = masks[k] > 0
            pr = pred[k] > 0
            cls = samples[i + k].class_name
            rows.append({
                "index": i + k,
                "path": samples[i + k].image_path.relative_to(ROOT).as_posix(),
                "class": cls,
                "gt_pixels": int(gt.sum()),
                "pred_pixels": int(pr.sum()),
                "dice": dice(pr, gt),
                "iou": iou(pr, gt),
                "pixel_acc": float((pr == gt).mean()),
                "empty_gt": bool(gt.sum() == 0),
                "pred_empty": bool(pr.sum() == 0),
            })

    def agg(subset):
        if not subset:
            return None
        return {
            "n": len(subset),
            "dice_mean": float(np.mean([r["dice"] for r in subset])),
            "dice_median": float(np.median([r["dice"] for r in subset])),
            "iou_mean": float(np.mean([r["iou"] for r in subset])),
            "pixel_acc_mean": float(np.mean([r["pixel_acc"] for r in subset])),
        }

    tumor = [r for r in rows if not r["empty_gt"]]
    empty = [r for r in rows if r["empty_gt"]]
    per_class = {}
    for name in CLASS_NAMES:
        sub = [r for r in tumor if r["class"] == name]
        if sub:
            per_class[name] = agg(sub)

    payload = {
        "kind": "phase3_seg_aux_branch",
        "run": run_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "BRISC 2025 official test split (classification_task/test + segmentation_task/test/masks)",
        "task": "binary tumour vs background on 2-D 224x224 slices (aux head, num_seg_classes=2)",
        "definition": {
            "dice": "2|P∩G|/(|P|+|G|); empty GT and empty prediction scored 1.0",
            "note": "tumour-only rows exclude no_tumour slices; those are reported as empty_gt_empty_pred",
        },
        "n_total": len(rows),
        "all_slices": agg(rows),
        "tumor_slices": agg(tumor),
        "empty_gt_slices": agg(empty),
        "empty_gt_empty_pred": (float(np.mean([r["pred_empty"] for r in empty])) if empty else None),
        "per_class_tumor_slices": per_class,
        "checkpoint": str(ckpt_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "run_config_sha256": sha256_file(run_dir / "run_config.json"),
        "run_git": run_config.get("git"),
        "seed": run_config.get("seed"),
        "script": "scripts/phase3_seg_eval.py",
        "per_slice": rows,
    }
    out_dir = run_dir / "eval_brisc_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "seg_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=list(DEFAULT_RUNS))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run_dirs = [RUNS / r for r in args.runs]
    missing = [d for d in run_dirs if not (d / "best.pt").is_file()]
    if missing:
        for d in missing:
            print(f"MISSING best.pt: {d.name}")
        return 2
    jobs = [d for d in run_dirs
            if args.force or not (d / "eval_brisc_test" / "seg_metrics.json").is_file()]
    print(f"seg_eval: {len(run_dirs)} runs, {len(jobs)} to compute")
    for d in jobs:
        print(f"  {d.name}")
    if args.dry_run:
        print("\n(dry run - nothing executed)")
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = load_brisc_split()
    samples = splits["test"]
    n_tumor = sum(1 for s in samples if s.mask_path is not None)
    print(f"BRISC test slices: {len(samples)} (all carry a mask; no_tumour uses the all-zero mask)")

    payloads, failed = [], []
    for i, d in enumerate(jobs, 1):
        t = time.time()
        print(f"[{i}/{len(jobs)}] {d.name} ...", flush=True)
        try:
            p = evaluate(d, samples, device, args.batch_size)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            failed.append(d.name)
            continue
        payloads.append(p)
        ts = p["tumor_slices"]
        print(f"  dice(tumour) {ts['dice_mean']:.4f} | iou {ts['iou_mean']:.4f} | "
              f"empty->empty {p['empty_gt_empty_pred']:.3f} ({time.time() - t:.0f}s)", flush=True)

    # merge on-disk artifacts
    on_disk = {}
    for d in RUNS.iterdir():
        f = d / "eval_brisc_test" / "seg_metrics.json"
        if f.is_file():
            try:
                on_disk[d.name] = json.loads(f.read_text(encoding="utf-8"))
            except ValueError:
                pass

    groups_out = []
    for slug, names in GROUPS:
        present = [on_disk[n] for n in names if n in on_disk]
        if not present:
            continue
        dices = [p["tumor_slices"]["dice_mean"] for p in present]
        groups_out.append({
            "slug": slug,
            "n_seeds": len(present),
            "runs": [p["run"] for p in present],
            "dice_tumor_mean": float(statistics.fmean(dices)),
            "dice_tumor_std_sample_ddof1": statistics.stdev(dices) if len(dices) > 1 else None,
            "iou_tumor_mean": float(statistics.fmean([p["tumor_slices"]["iou_mean"] for p in present])),
            "pixel_acc_mean": float(statistics.fmean([p["tumor_slices"]["pixel_acc_mean"] for p in present])),
            "empty_gt_empty_pred_mean": float(
                statistics.fmean([p["empty_gt_empty_pred"] for p in present
                                  if p["empty_gt_empty_pred"] is not None])
            ),
            "per_seed_dice": {p["run"]: p["tumor_slices"]["dice_mean"] for p in present},
        })

    summary = {
        "kind": "phase3_seg_summary",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "headline_rule": "mean over seeds (never best seed); std = sample ddof=1",
        "note": "These are the ONLY segmentation numbers this repo can defend for the "
                "multi-task net. The paper's 0.9005/0.5803/0.6010 WT/TC/ET come from a "
                "standalone 3-D U-Net on BraTS (D3) and must be relabelled, not reused.",
        "groups": groups_out,
        "failed_this_batch": failed,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 92)
    print(" Phase 3E segmentation — aux head on BRISC test (tumour slices)")
    print("=" * 92)
    for g in groups_out:
        sd = "" if g["dice_tumor_std_sample_ddof1"] is None else f" ± {g['dice_tumor_std_sample_ddof1']:.4f}"
        print(f" {g['slug']:<26} n={g['n_seeds']}  dice {g['dice_tumor_mean']:.4f}{sd}  "
              f"iou {g['iou_tumor_mean']:.4f}  pixacc {g['pixel_acc_mean']:.4f}")
    print(f" summary -> {SUMMARY.relative_to(ROOT)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
