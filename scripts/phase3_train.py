"""Phase 3 single-run trainer (one model x one seed x one protocol).

Trains under the repo's canonical protocol (identical for every config, so
baselines are fair):

    optimizer AdamW(lr=1e-4, wd=1e-4) -> CosineAnnealingLR(T_max=epochs)
    loss     CE(label_smoothing=0.1) for cls (+ L_seg only if seg targets are
             supplied - Phase 3D finding: historical L_seg was inert, D2)
    input    Resize(224) -> ToTensor -> ImageNet normalize (deterministic)
    early stop  patience=7 on weighted val F1 (selection metric; the reported
             headline metric is macro-F1, computed by phase3_evaluate.py)

Provenance: writes ``run_config.json`` BEFORE training (seed, git commit,
split-manifest sha256, full flags, dataset counts, torch/device) and relies on
the trainer's per-epoch ``training_history.json`` + ``progress.json``.

The TEST split is NOT touched here - evaluation is a separate explicit step
(``scripts/phase3_evaluate.py``) so the test set is used once per final config.

Usage:
    python scripts/phase3_train.py --model tamnet --seed 42
    python scripts/phase3_train.py --model densenet121 --seed 42 --epochs 30
    python scripts/phase3_train.py --model tamnet --seed 43 --channel-attention
    python scripts/phase3_train.py --model tamnet --protocol leaky --seed 42
    python scripts/phase3_train.py --model tamnet --smoke        # 1-epoch pipeline test
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets.classification import (  # noqa: E402
    CLASS_NAMES,
    BrainTumorClassificationDataset,
    ClassificationSample,
    discover_classification_samples,
    split_classification_samples,
)
from datasets.dataloaders import make_dataloader  # noqa: E402
from datasets.normalization import build_classification_transform  # noqa: E402
from models.factory import create_backbone_classifier  # noqa: E402
from training.multitask_trainer import (  # noqa: E402
    MultiTaskTrainingConfig,
    train_multitask_model,
)

BACKBONES = (
    "densenet121",
    "resnet50",
    "efficientnet_v2_s",
    "convnext_tiny",
    "swin_t",
    "vit_b_16",
)
DEFAULT_SPLIT = ROOT / "results" / "phase3" / "split_manifest_clean.json"
RUNS_DIR = ROOT / "results" / "phase3" / "runs"


class DictWrapper(nn.Module):
    """Adapts a plain classifier to the dict contract of train_multitask_model.

    ``forward(x) -> {"logits": ..., "probabilities": ...}``; accepts and
    ignores ``return_seg``/``seg_mask`` so the trainer's call signature works
    unchanged (fair baselines = identical training loop).
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(
        self,
        x: torch.Tensor,
        return_seg: bool = False,  # noqa: ARG002 - contract compat
        seg_mask: torch.Tensor | None = None,  # noqa: ARG002
    ) -> dict[str, torch.Tensor]:
        logits = self.backbone(x)
        return {"logits": logits, "probabilities": torch.softmax(logits, dim=1)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3 single-run trainer")
    p.add_argument("--model", required=True, help="tamnet | " + " | ".join(BACKBONES))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=None, help="default: tamnet 50, backbones 30")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--protocol", choices=("clean", "leaky"), default="clean")
    p.add_argument("--split", type=str, default=str(DEFAULT_SPLIT))
    p.add_argument("--out", type=str, default=None, help="run directory")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--smoke", action="store_true", help="1 epoch, tiny subsets (pipeline test)")
    # TAMNet ablation flags (defaults = historically executed model)
    p.add_argument("--channel-attention", action="store_true", help="enable M_c (D1 ablation)")
    p.add_argument("--no-attention-fusion", action="store_true", help="concat fusion (D4 ablation)")
    p.add_argument("--no-seg-head", action="store_true")
    p.add_argument("--no-uncertainty-head", action="store_true")
    p.add_argument("--seg-decoder", choices=("lightweight", "multiscale"), default="lightweight")
    return p.parse_args()


def git_info() -> dict:
    def _run(args: list[str]) -> str:
        import subprocess

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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_split(protocol: str, manifest_path: Path) -> dict[str, list[ClassificationSample]]:
    if protocol == "clean":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        buckets: dict[str, list[ClassificationSample]] = {
            "train": [],
            "val": [],
            "test": [],
        }
        for e in manifest["entries"]:
            cls_idx = CLASS_NAMES.index(e["class"])
            buckets[e["split"]].append(
                ClassificationSample(
                    image_path=ROOT / e["path"],
                    class_name=e["class"],
                    class_index=cls_idx,
                    split=e["split"],
                )
            )
        return buckets

    # leaky = the historical protocol (reproduction run, 1 seed)
    samples = discover_classification_samples(ROOT / "data" / "raw" / "classification")
    train_all = [s for s in samples if s.split.lower() == "train"]
    test = [s for s in samples if s.split.lower() == "test"]
    train, val = split_classification_samples(
        train_all, validation_split=0.2, seed=42
    )
    return {"train": train, "val": val, "test": test}


def main() -> int:
    args = parse_args()
    if args.model != "tamnet" and args.model not in BACKBONES:
        raise SystemExit(f"--model must be tamnet or one of {BACKBONES}")

    if args.epochs is None:
        args.epochs = 1 if args.smoke else (50 if args.model == "tamnet" else 30)

    # ---- reproducibility ------------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- run naming / output dir ---------------------------------------
    suffix = ""
    if args.model == "tamnet":
        if args.channel_attention:
            suffix += "+mc"
        if args.no_attention_fusion:
            suffix += "+concat"
        if args.no_seg_head:
            suffix += "-seg"
        if args.no_uncertainty_head:
            suffix += "-unc"
        if args.seg_decoder == "multiscale":
            suffix += "+msdec"
    run_name = f"{args.model}_{args.protocol}_s{args.seed}{suffix}"
    out_dir = Path(args.out) if args.out else RUNS_DIR / run_name
    if out_dir.exists() and (out_dir / "run_config.json").exists():
        raise SystemExit(f"run config already exists (refusing to overwrite): {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data -----------------------------------------------------------
    split_path = Path(args.split)
    splits = load_split(args.protocol, split_path)
    if args.smoke:
        splits["train"] = splits["train"][:128]
        splits["val"] = splits["val"][:64]
    for key in ("train", "val"):
        if not splits[key]:
            raise SystemExit(f"empty {key} split - check manifest/protocol")

    transform = build_classification_transform((224, 224))
    train_ds = BrainTumorClassificationDataset(splits["train"], transform=transform)
    val_ds = BrainTumorClassificationDataset(splits["val"], transform=transform)
    train_loader = make_dataloader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
        persistent_workers=not args.smoke,
    )
    val_loader = make_dataloader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    class_counts = {
        "train": dict(sorted(Counter(s.class_name for s in splits["train"]).items())),
        "val": dict(sorted(Counter(s.class_name for s in splits["val"]).items())),
        "test": dict(sorted(Counter(s.class_name for s in splits["test"]).items())),
    }

    # ---- model ----------------------------------------------------------
    flags = {
        "use_channel_attention": args.channel_attention,
        "seg_decoder": args.seg_decoder,
        "use_attention_fusion": not args.no_attention_fusion,
        "use_segmentation_head": not args.no_seg_head,
        "use_uncertainty_head": not args.no_uncertainty_head,
    }
    model: nn.Module | None = None
    if args.model != "tamnet":
        model = DictWrapper(
            create_backbone_classifier(
                args.model,
                num_classes=len(CLASS_NAMES),
                pretrained=not args.no_pretrained,
                dropout=0.3,
            )
        )

    config = MultiTaskTrainingConfig(
        backbone="densenet121" if args.model == "tamnet" else args.model,
        pretrained=not args.no_pretrained,
        num_classes=len(CLASS_NAMES),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        output_dir=str(out_dir),
        **flags,
    )

    # ---- provenance (written BEFORE training) ---------------------------
    run_config = {
        "run_name": run_name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_info(),
        "model": args.model,
        "protocol": args.protocol,
        "flags": flags if args.model == "tamnet" else {},
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "label_smoothing": config.label_smoothing,
        "optimizer": "AdamW+CosineAnnealingLR",
        "selection_metric": "val_f1_weighted",
        "image_size": [224, 224],
        "transform": "Resize->ToTensor->Normalize(ImageNet)",
        "num_workers": args.num_workers,
        "pretrained": not args.no_pretrained,
        "smoke": args.smoke,
        "split": {
            "protocol": args.protocol,
            "manifest": str(split_path.relative_to(ROOT))
            if split_path.is_relative_to(ROOT)
            else str(split_path),
            "sha256": sha256_file(split_path) if split_path.exists() else None,
        },
        "dataset": {"n_total": {k: len(v) for k, v in splits.items()}, "classes": class_counts},
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "command": " ".join(sys.argv),
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    print(f"RUN {run_name}", flush=True)
    print(f"  out      : {out_dir}", flush=True)
    print(f"  protocol : {args.protocol} | seed {args.seed} | epochs {args.epochs}", flush=True)
    print(f"  data     : train {len(train_ds)} | val {len(val_ds)} | test {len(splits['test'])} (untouched)", flush=True)
    print(f"  model    : {args.model} {flags if args.model == 'tamnet' else ''}", flush=True)
    print(f"  device   : {run_config['device']}", flush=True)

    # ---- train ----------------------------------------------------------
    started = time.time()
    result = train_multitask_model(config, train_loader, val_loader, device, model=model)
    duration = time.time() - started

    ckpt = Path(result.checkpoint_path)
    try:
        ckpt_str = ckpt.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        ckpt_str = str(ckpt)
    summary = {
        "run_name": run_name,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_s": round(duration, 1),
        "best_val_f1_weighted": result.best_val_f1,
        "best_epoch": result.best_epoch,
        "epochs_run": len(result.training_history),
        "checkpoint": ckpt_str,
        "git": run_config["git"],
        "seed": args.seed,
    }
    (out_dir / "training_result.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"DONE {run_name}: best val F1(w)={result.best_val_f1:.4f} "
        f"@ epoch {result.best_epoch} in {duration/3600:.2f} h",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
