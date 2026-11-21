"""Phase 3 evaluator: explicit, once-per-config test/external evaluation.

Loads a trained run (``run_config.json`` + ``best.pt``), rebuilds the model,
runs inference on ONE requested split/dataset, and writes:

    predictions.csv   per-image: path, y_true, y_pred, p_<class>... (the source
                      of truth for every later metric - F1 audit, calibration,
                      robustness, confusion matrices all re-read this file)
    metrics.json      accuracy, macro-F1 (headline), weighted-F1, per-class
                      P/R/F1/support, 4x4 confusion matrix + full provenance

Test-split usage is deliberate and logged: the test set is evaluated once per
final configuration. External datasets optionally pass ``--dedup`` to keep only
the leakage-free images listed in ``results/phase3/external_dedup_manifest.json``.

Usage:
    python scripts/phase3_evaluate.py --run-dir results/phase3/runs/tamnet_clean_s42 --split test
    python scripts/phase3_evaluate.py --run-dir ... --split test --dataset kaggle_testing --dedup
    python scripts/phase3_evaluate.py --run-dir ... --split val          # sanity, no test touch
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets.classification import (  # noqa: E402
    CLASS_NAMES,
    BrainTumorClassificationDataset,
    ClassificationSample,
)
from datasets.dataloaders import make_dataloader  # noqa: E402
from datasets.normalization import build_classification_transform  # noqa: E402
from models.factory import create_backbone_classifier  # noqa: E402
from models.tumor_aware import TumorAwareConfig, TumorAwareMultiTaskNet  # noqa: E402
from scripts.leakage_check import SOURCES, collect_files, label_of  # noqa: E402
from scripts.phase3_train import DictWrapper  # noqa: E402

LABEL_MAP = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "pituitary": "Pituitary",
    "no_tumor": "No Tumor",
}
DEFAULT_SPLIT = ROOT / "results" / "phase3" / "split_manifest_clean.json"
DEDUP_MANIFEST = ROOT / "results" / "phase3" / "external_dedup_manifest.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3 evaluator")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument(
        "--dataset",
        default="internal",
        help="internal | " + " | ".join(k for k in SOURCES if not k.startswith("internal")),
    )
    p.add_argument("--dedup", action="store_true", help="apply external keep-list")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_model(run_config: dict) -> torch.nn.Module:
    if run_config["model"] == "tamnet":
        flags = run_config.get("flags", {})
        cfg = TumorAwareConfig(
            backbone=run_config.get("backbone", "densenet121"),
            pretrained=False,  # weights come from the checkpoint
            num_classes=len(CLASS_NAMES),
            dropout=run_config.get("dropout", 0.3),
            use_channel_attention=flags.get("use_channel_attention", False),
            seg_decoder=flags.get("seg_decoder", "lightweight"),
            use_attention_fusion=flags.get("use_attention_fusion", True),
            use_segmentation_head=flags.get("use_segmentation_head", True),
            use_uncertainty_head=flags.get("use_uncertainty_head", True),
            fusion_mode=flags.get("fusion_mode", "g+t"),
            segmentation_classes=run_config.get("segmentation_classes", 4),
        )
        return TumorAwareMultiTaskNet(cfg)
    return DictWrapper(
        create_backbone_classifier(
            run_config["model"],
            num_classes=len(CLASS_NAMES),
            pretrained=False,
            dropout=0.3,
        )
    )


def resolve_samples(args: argparse.Namespace) -> tuple[list[ClassificationSample], dict]:
    """Return (samples, provenance-dict) for the requested evaluation set."""
    if args.dataset == "internal":
        manifest_path = DEFAULT_SPLIT
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        samples = [
            ClassificationSample(
                image_path=ROOT / e["path"],
                class_name=e["class"],
                class_index=CLASS_NAMES.index(e["class"]),
                split=e["split"],
            )
            for e in manifest["entries"]
            if e["split"] == args.split
        ]
        prov = {
            "dataset": "internal",
            "split": args.split,
            "manifest": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": sha256_file(manifest_path),
            "dedup_applied": False,
        }
        return samples, prov

    if args.dataset not in SOURCES:
        raise SystemExit(f"unknown --dataset {args.dataset!r}")
    files = collect_files(SOURCES[args.dataset])
    if args.dedup:
        if not DEDUP_MANIFEST.exists():
            raise SystemExit("external_dedup_manifest.json missing - run phase3_split.py")
        dedup = json.loads(DEDUP_MANIFEST.read_text(encoding="utf-8"))
        keep = set(dedup["sources"][args.dataset]["keep_paths"])
        files = [f for f in files if f.relative_to(ROOT).as_posix() in keep]
    samples = []
    for f in files:
        raw = label_of(f)
        cls = LABEL_MAP.get(raw, raw)
        if cls not in CLASS_NAMES:
            continue
        samples.append(
            ClassificationSample(
                image_path=f, class_name=cls, class_index=CLASS_NAMES.index(cls), split="test"
            )
        )
    prov = {
        "dataset": args.dataset,
        "split": "external",
        "manifest": str(DEDUP_MANIFEST.relative_to(ROOT)) if args.dedup else None,
        "manifest_sha256": sha256_file(DEDUP_MANIFEST) if args.dedup else None,
        "dedup_applied": bool(args.dedup),
    }
    return samples, prov


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"missing checkpoint: {ckpt_path}")

    samples, data_prov = resolve_samples(args)
    if not samples:
        raise SystemExit("no samples resolved - check dataset/dedup/split")
    print(
        f"EVAL {run_config['run_name']} | {data_prov['dataset']}/{data_prov['split']} "
        f"| n={len(samples)} | dedup={data_prov['dedup_applied']}",
        flush=True,
    )

    # ---- model + weights ------------------------------------------------
    model = build_model(run_config)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # ---- inference ------------------------------------------------------
    dataset = BrainTumorClassificationDataset(
        samples, transform=build_classification_transform((224, 224))
    )
    loader = make_dataloader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    y_true: list[int] = []
    y_pred: list[int] = []
    probs: list[np.ndarray] = []
    paths: list[str] = []
    with torch.no_grad():
        for batch in loader:
            # index access: mask-carrying batches are 3-tuples (Phase 3)
            images, labels = batch[0], batch[1]
            logits = model(images.to(device))
            if isinstance(logits, dict):
                logits = logits["logits"]
            p = torch.softmax(logits, dim=1).cpu().numpy()
            probs.extend(p)
            y_pred.extend(p.argmax(axis=1).tolist())
            y_true.extend(labels.numpy().tolist())
    for s in samples:
        paths.append(s.image_path.relative_to(ROOT).as_posix())

    # ---- metrics --------------------------------------------------------
    y_true_a, y_pred_a = np.asarray(y_true), np.asarray(y_pred)
    report = classification_report(
        y_true_a,
        y_pred_a,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true_a, y_pred_a)),
        "macro_f1": float(f1_score(y_true_a, y_pred_a, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true_a, y_pred_a, average="weighted", zero_division=0)
        ),
        "per_class": {
            cls: {
                "precision": report[cls]["precision"],
                "recall": report[cls]["recall"],
                "f1": report[cls]["f1-score"],
                "support": int(report[cls]["support"]),
            }
            for cls in CLASS_NAMES
        },
        "confusion_matrix": confusion_matrix(
            y_true_a, y_pred_a, labels=list(range(len(CLASS_NAMES)))
        ).tolist(),
        "confusion_matrix_order": list(CLASS_NAMES),
        "checkpoint_best_epoch": checkpoint.get("epoch"),
        "checkpoint_val_f1": checkpoint.get("val_f1"),
    }

    # ---- save -----------------------------------------------------------
    out_dir = run_dir / f"eval_{data_prov['dataset']}_{data_prov['split']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["path", "y_true", "y_pred"] + [f"p_{c.replace(' ', '_')}" for c in CLASS_NAMES]
        )
        for path, yt, yp, p in zip(paths, y_true, y_pred, probs):
            writer.writerow([path, yt, yp] + [f"{v:.6f}" for v in p])

    payload = {
        **metrics,
        "provenance": {
            "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_name": run_config["run_name"],
            "run_git": run_config.get("git"),
            "seed": run_config.get("seed"),
            "checkpoint": str(ckpt_path.relative_to(ROOT)),
            "checkpoint_sha256": sha256_file(ckpt_path),
            "run_config_sha256": sha256_file(run_dir / "run_config.json"),
            "evaluator": "scripts/phase3_evaluate.py",
            **data_prov,
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        f"  n={metrics['n']} acc={metrics['accuracy']:.4f} "
        f"macroF1={metrics['macro_f1']:.4f} weightedF1={metrics['weighted_f1']:.4f}",
        flush=True,
    )
    print(f"  wrote {out_dir.relative_to(ROOT)}/predictions.csv + metrics.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
