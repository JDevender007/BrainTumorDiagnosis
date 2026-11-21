"""
Multi-task training framework for TumorAwareMultiTaskNet.

Handles:
- Combined segmentation + classification loss
- Multi-task gradient balancing
- Checkpointing
- Mixed precision training
- Gradient accumulation
"""

from __future__ import annotations

import json
import random
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from models.tumor_aware import TumorAwareMultiTaskNet, TumorAwareConfig
from losses.segmentation import DiceLoss2D, DiceLoss3D, SegmentationLoss


@dataclass
class MultiTaskTrainingConfig:
    backbone: str = "densenet121"
    pretrained: bool = True
    num_classes: int = 4
    segmentation_classes: int = 4
    epochs: int = 50
    batch_size: int = 8
    image_size: tuple[int, int] = (224, 224)
    lr: float = 0.0001
    weight_decay: float = 0.0001
    patience: int = 7
    label_smoothing: float = 0.1
    seg_loss_weight: float = 0.5
    cls_loss_weight: float = 1.0
    uncertainty_loss_weight: float = 0.1
    dropout: float = 0.3
    fusion_dim: int = 256
    use_attention_fusion: bool = True
    use_segmentation_head: bool = True
    use_uncertainty_head: bool = True
    # Phase 2/3 ablation flags (defaults = historically executed model)
    use_channel_attention: bool = False
    seg_decoder: str = "lightweight"
    # Phase 3A t-only ablation: "g+t" (default) | "t"
    fusion_mode: str = "g+t"
    seed: int = 42
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 2
    output_dir: str = "outputs/research_experiments/proposed_model"


@dataclass
class TrainingResult:
    best_val_f1: float
    best_epoch: int
    training_history: list[dict]
    checkpoint_path: str
    config: dict


class MultiTaskLoss(nn.Module):
    """Combined loss for multi-task learning."""

    def __init__(
        self,
        num_classes: int = 4,
        cls_weight: float = 1.0,
        seg_weight: float = 0.5,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.seg_loss = DiceLoss3D()
        # 2D variant for TAMNet's (B, C, H, W) aux-branch logits (Phase 2):
        # DiceLoss3D would reject them with a shape error.
        self.seg_loss_2d = DiceLoss2D()
        self.cls_weight = cls_weight
        self.seg_weight = seg_weight

    def forward(
        self,
        outputs: dict,
        labels: torch.Tensor,
        seg_targets: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        cls_loss = self.cls_loss(outputs["logits"], labels)

        total_loss = self.cls_weight * cls_loss
        seg_loss_val = torch.tensor(0.0, device=labels.device)

        if "seg_logits" in outputs and seg_targets is None and not getattr(self, "_seg_skipped", False):
            # Phase 2 audit: every training path in this repo calls
            # criterion(outputs, labels) — or passes seg_targets=None
            # explicitly — so L_seg was 0 in ALL historical runs and the
            # auxiliary segmentation branch received no gradient.
            # Warn once instead of failing silently (loss value unchanged,
            # so historical behaviour is preserved). See docs/ARCHITECTURE.md.
            warnings.warn(
                "MultiTaskLoss: model returned 'seg_logits' but no "
                "seg_targets were provided — segmentation loss is 0 for "
                "this step and the auxiliary branch receives no gradient. "
                "Supply seg_targets (masks) to activate L_seg, or disable "
                "return_seg. See docs/ARCHITECTURE.md finding D2.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._seg_skipped = True

        if "seg_logits" in outputs and seg_targets is not None:
            seg_pred = outputs["seg_logits"]
            # dispatch: 2D aux-branch logits -> DiceLoss2D, 5D volumes -> DiceLoss3D
            seg_loss_val = (
                self.seg_loss_2d(seg_pred, seg_targets)
                if seg_pred.ndim == 4
                else self.seg_loss(seg_pred, seg_targets)
            )
            total_loss = total_loss + self.seg_weight * seg_loss_val

        return {
            "total_loss": total_loss,
            "cls_loss": cls_loss,
            "seg_loss": seg_loss_val,
        }


def train_multitask_model(
    config: MultiTaskTrainingConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    external_loader: Optional[DataLoader] = None,
    model: Optional[nn.Module] = None,
    resume: bool = False,
) -> TrainingResult:
    """Train the multi-task tumor-aware model.

    ``model``: optional pre-built network (Phase 3) for fair baselines under
    the *identical* protocol (same optimizer/schedule/loss/early-stopping).
    It must accept ``forward(x, return_seg=...)`` and return a dict with
    ``"logits"``. When None (default) TAMNet is built from ``config``.

    ``resume``: when True and ``<output_dir>/checkpoint_last.pt`` exists,
    training continues from the last COMPLETED epoch (model/optimizer/
    scheduler/scaler/best-state/history/RNG/shuffle order all restored) —
    a shutdown mid-run loses at most the in-flight epoch, never the run.
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if model is None:
        model_config = TumorAwareConfig(
            backbone=config.backbone,
            pretrained=config.pretrained,
            num_classes=config.num_classes,
            segmentation_classes=config.segmentation_classes,
            dropout=config.dropout,
            fusion_dim=config.fusion_dim,
            use_attention_fusion=config.use_attention_fusion,
            use_segmentation_head=config.use_segmentation_head,
            use_uncertainty_head=config.use_uncertainty_head,
            use_channel_attention=config.use_channel_attention,
            seg_decoder=config.seg_decoder,
            fusion_mode=config.fusion_mode,
        )
        model = TumorAwareMultiTaskNet(model_config)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    criterion = MultiTaskLoss(
        num_classes=config.num_classes,
        cls_weight=config.cls_loss_weight,
        seg_weight=config.seg_loss_weight,
        label_smoothing=config.label_smoothing,
    )

    scaler = GradScaler("cuda", enabled=config.mixed_precision)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []
    run_start = time.time()
    progress_path = output_dir / "progress.json"
    last_ckpt_path = output_dir / "checkpoint_last.pt"
    start_epoch = 0

    if resume and last_ckpt_path.exists():
        # weights_only=False: the checkpoint carries numpy/python RNG state
        # tuples (trusted, written by this trainer).
        ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        best_val_f1 = float(ckpt["best_val_f1"])
        best_epoch = int(ckpt["best_epoch"])
        patience_counter = int(ckpt["patience_counter"])
        history = list(ckpt["history"])
        start_epoch = int(ckpt["epoch"])  # completed epochs -> next epoch index
        run_start = time.time() - float(ckpt.get("elapsed_s", 0.0))
        rng = ckpt.get("rng_state") or {}
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"].cpu())
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])
        if rng.get("python") is not None:
            random.setstate(rng["python"])
        gen = getattr(train_loader, "generator", None)
        if gen is not None and ckpt.get("train_generator_state") is not None:
            gen.set_state(ckpt["train_generator_state"].cpu())
        print(
            f"RESUME: checkpoint_last.pt found -> continuing from epoch "
            f"{start_epoch + 1}/{config.epochs} (best val F1 {best_val_f1:.4f} "
            f"@ epoch {best_epoch}, {len(history)} epochs in history)",
            flush=True,
        )
    elif resume:
        print("RESUME: no checkpoint_last.pt yet - starting from epoch 1", flush=True)

    def _write_progress(status: str, epoch_time_s: float) -> None:
        """Write machine-readable per-epoch status (Phase 3 progress protocol).

        An external watcher polls ``progress.json`` every few minutes and can
        report each epoch completion without waiting for the run to finish.
        """
        elapsed = time.time() - run_start
        epochs_done = len(history)
        last = history[-1] if history else {}
        eta = (elapsed / max(epochs_done, 1)) * (config.epochs - epochs_done)
        payload = {
            "status": status,
            "epoch": epochs_done,
            "epochs_total": config.epochs,
            "epoch_time_s": round(epoch_time_s, 1),
            "elapsed_s": round(elapsed, 1),
            "eta_s": round(eta, 1),
            "train_loss": last.get("train_loss"),
            "val_f1": last.get("val_f1"),
            "val_accuracy": last.get("val_accuracy"),
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        tmp = progress_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(progress_path)

    def _save_last_checkpoint(epoch: int) -> None:
        """Atomically persist full resumable state after every epoch.

        Written BEFORE progress.json so the watcher never claims an epoch
        that is not checkpointed. A shutdown mid-run therefore loses at most
        the current (unfinished) epoch; restarting phase3_train auto-resumes
        from ``epoch + 1`` with model/optimizer/scheduler/scaler, best-state,
        patience, history, wall-clock and ALL RNG + shuffle-generator state.
        """
        gen = getattr(train_loader, "generator", None)
        payload = {
            "epoch": epoch + 1,  # completed epochs; resume starts at this index
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
            "patience_counter": patience_counter,
            "history": history,
            "elapsed_s": time.time() - run_start,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": (
                    torch.cuda.get_rng_state_all()
                    if device.type == "cuda"
                    else None
                ),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            "train_generator_state": gen.get_state() if gen is not None else None,
        }
        tmp = last_ckpt_path.parent / (last_ckpt_path.name + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(last_ckpt_path)

    # ``epoch`` pre-set so a resume at start_epoch == config.epochs (crash
    # after the final epoch, before training_result.json) still completes.
    epoch = start_epoch - 1
    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()
        model.train()
        epoch_losses = []
        epoch_cls_losses = []
        epoch_seg_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", leave=False)
        for batch_idx, batch in enumerate(pbar):
            # Phase 3 (D2/λ_seg): datasets that carry masks yield
            # (image, label, mask); everything else stays (image, label)
            # so historical behaviour is bit-identical.
            if len(batch) == 3:
                images, labels, seg_targets = batch
                seg_targets = seg_targets.to(device, non_blocking=True)
            else:
                images, labels = batch
                seg_targets = None
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast("cuda", enabled=config.mixed_precision):
                outputs = model(images, return_seg=True)
                losses = criterion(outputs, labels, seg_targets=seg_targets)
                loss = losses["total_loss"] / config.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            epoch_losses.append(losses["total_loss"].item())
            epoch_cls_losses.append(losses["cls_loss"].item())
            epoch_seg_losses.append(losses["seg_loss"].item())

            pbar.set_postfix({
                "loss": f"{losses['total_loss'].item():.4f}",
                "cls": f"{losses['cls_loss'].item():.4f}",
            })

            if (batch_idx + 1) % 5 == 0:
                n = len(epoch_losses)
                print(
                    f"  [Batch {batch_idx+1}/{len(train_loader)}] "
                    f"loss={sum(epoch_losses[-5:])/min(5,n):.4f} "
                    f"cls={sum(epoch_cls_losses[-5:])/min(5,n):.4f} "
                    f"seg={sum(epoch_seg_losses[-5:])/min(5,n):.4f}",
                    flush=True,
                )

        scheduler.step()

        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                # index access: mask-carrying batches are 3-tuples (Phase 3)
                images, labels = batch[0], batch[1]
                images = images.to(device, non_blocking=True)
                outputs = model(images)
                preds = outputs["logits"].argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.numpy())

        val_f1 = f1_score(val_labels, val_preds, average="weighted", zero_division=0)
        val_acc = accuracy_score(val_labels, val_preds)

        epoch_info = {
            "epoch": epoch + 1,
            "epoch_time_s": round(time.time() - epoch_start, 1),
            "train_loss": float(np.mean(epoch_losses)),
            "train_cls_loss": float(np.mean(epoch_cls_losses)),
            "train_seg_loss": float(np.mean(epoch_seg_losses)),
            "val_f1": val_f1,
            "val_accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_info)

        print(
            f"  Epoch {epoch+1}/{config.epochs} | "
            f"train_loss={epoch_info['train_loss']:.4f} "
            f"cls={epoch_info['train_cls_loss']:.4f} "
            f"seg={epoch_info['train_seg_loss']:.4f} | "
            f"val_f1={val_f1:.4f} val_acc={val_acc:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f}",
            flush=True,
        )

        # Phase 3 progress protocol: persist training_history after EVERY
        # epoch so watchers (and crash recovery) see per-epoch state.
        # NOTE: progress.json itself is written AFTER the best/patience block
        # below so it never reports a best_val_f1 that lags one epoch behind
        # the checkpoint on disk (best.pt).
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_accuracy": val_acc,
                "config": {
                    "backbone": config.backbone,
                    "num_classes": config.num_classes,
                    "dropout": config.dropout,
                    "fusion_dim": config.fusion_dim,
                },
            }, output_dir / "best.pt")
        else:
            patience_counter += 1

        # Checkpoint FIRST (contains the just-updated best/patience), then
        # progress.json - so both disk states always agree, even if the
        # system shuts down between the two writes.
        _save_last_checkpoint(epoch)

        # Write progress AFTER best/patience are updated so the watcher's
        # best_val_f1/best_epoch always match best.pt on disk.
        _write_progress("running", time.time() - epoch_start)

        if patience_counter >= config.patience:
            print(f"  Early stopping at epoch {epoch+1}")
            _write_progress("early_stopped", time.time() - epoch_start)
            break

    torch.save({
        "epoch": history[-1]["epoch"] if history else start_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
    }, output_dir / "last.pt")

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    _write_progress("completed", history[-1].get("epoch_time_s", 0.0) if history else 0.0)

    print(f"\nBest val F1: {best_val_f1:.4f} at epoch {best_epoch}")

    return TrainingResult(
        best_val_f1=best_val_f1,
        best_epoch=best_epoch,
        training_history=history,
        checkpoint_path=str(output_dir / "best.pt"),
        config={
            "backbone": config.backbone,
            "epochs": config.epochs,
            "lr": config.lr,
            "batch_size": config.batch_size,
        },
    )
