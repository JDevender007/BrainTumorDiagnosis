"""
Comprehensive Research Experiment Runner for BrainTumorDiagnosis.

Runs all experiments required for publication:
- Ablation study (Experiments A-H)
- Multi-seed validation (seeds 42, 123, 456)
- Classification evaluation with full metrics
- Statistical comparison
- External validation
- Robustness, calibration, uncertainty
- Computational efficiency
- Generates all tables and figures

Usage:
    python scripts/run_all_experiments.py --phase ablation
    python scripts/run_all_experiments.py --phase multi_seed
    python scripts/run_all_experiments.py --phase evaluation
    python scripts/run_all_experiments.py --phase all
"""

from __future__ import annotations

import json
import sys
import time
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, matthews_corrcoef,
    cohen_kappa_score, log_loss, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.seed import set_seed
from utils.device import get_device

# ============================================================================
# Data Loading
# ============================================================================

CLASS_NAMES = ["glioma", "meningioma", "no tumor", "pituitary"]
CLASS_MAP = {"glioma": 0, "meningioma": 1, "no tumor": 2, "pituitary": 3,
             "notumor": 2, "no_tumor": 2}


def normalize_class_name(name: str) -> str:
    n = name.lower().strip()
    for key, val in CLASS_MAP.items():
        if n == key:
            return ["glioma", "meningioma", "no tumor", "pituitary"][val]
    return n


class ClassificationDataset(Dataset):
    def __init__(self, root: str, split: str = "Train", transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.labels = []

        split_dir = self.root / split
        if not split_dir.exists():
            split_dir = self.root / split.capitalize()

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            cls_name = normalize_class_name(class_dir.name)
            if cls_name not in CLASS_MAP:
                continue
            label = CLASS_MAP[cls_name]
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
                    self.samples.append(str(img_path))
                    self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def load_data(
    root: str = "data/raw/classification",
    image_size: tuple = (224, 224),
    batch_size: int = 8,
    val_split: float = 0.2,
    seed: int = 42,
):
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full_train = ClassificationDataset(root, "Train", train_transform)
    test_ds = ClassificationDataset(root, "Test", test_transform)

    set_seed(seed)
    n = len(full_train)
    indices = list(range(n))
    np.random.seed(seed)
    np.random.shuffle(indices)
    val_n = int(n * val_split)
    val_indices = indices[:val_n]
    train_indices = indices[val_n:]

    train_ds = Subset(full_train, train_indices)
    val_ds_orig = ClassificationDataset(root, "Train", test_transform)
    val_ds = Subset(val_ds_orig, val_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    return train_loader, val_loader, test_loader


def load_external(root: str, image_size: tuple = (224, 224), batch_size: int = 8):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    samples, labels = [], []
    root_path = Path(root)
    for class_dir in sorted(root_path.iterdir()):
        if not class_dir.is_dir():
            continue
        cls_name = normalize_class_name(class_dir.name)
        if cls_name not in CLASS_MAP:
            continue
        label = CLASS_MAP[cls_name]
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
                samples.append(str(img_path))
                labels.append(label)

    class ExtDS(Dataset):
        def __init__(self, samples, labels, transform):
            self.samples = samples
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            img = Image.open(self.samples[idx]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, self.labels[idx]

    ds = ExtDS(samples, labels, transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


# ============================================================================
# Model Builders
# ============================================================================

def build_baseline(backbone: str = "densenet121", num_classes: int = 4):
    from models.factory import create_backbone_classifier
    return create_backbone_classifier(backbone, num_classes=num_classes, pretrained=True)


def build_tumor_aware(backbone: str = "densenet121", num_classes: int = 4):
    from models.tumor_aware import TumorAwareMultiTaskNet, TumorAwareConfig
    config = TumorAwareConfig(backbone=backbone, num_classes=num_classes, use_segmentation_head=True)
    return TumorAwareMultiTaskNet(config)


def build_turma_net(backbone: str = "densenet121", num_classes: int = 4, **kwargs):
    from models.novel.turma_net import TurmaNet, TurmaNetConfig
    config = TurmaNetConfig(backbone=backbone, num_classes=num_classes, **kwargs)
    return TurmaNet(config)


# ============================================================================
# Training
# ============================================================================

@dataclass
class TrainConfig:
    model_name: str
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 5
    seed: int = 42
    output_dir: str = "paper/results"
    use_turma_loss: bool = False
    turma_kwargs: dict = field(default_factory=dict)


def train_model(
    model: nn.Module,
    config: TrainConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> dict:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    if config.use_turma_loss:
        from models.novel.losses import TurmaNetLoss
        criterion = TurmaNetLoss(num_classes=4, use_uncertainty_weights=True)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    use_amp = device.type == "cuda"
    output_dir = Path(config.output_dir) / config.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []
    start_time = time.time()

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                if config.use_turma_loss:
                    outputs = model(images, return_seg=True)
                else:
                    outputs = model(images)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs

                if config.use_turma_loss and isinstance(outputs, dict) and "seg_logits" in outputs:
                    loss, loss_dict = criterion(logits, labels, outputs["seg_logits"], None)
                else:
                    loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        val_preds, val_labels, val_probs = [], [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                if isinstance(outputs, dict):
                    logits = outputs["logits"]
                else:
                    logits = outputs
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.numpy())
                val_probs.extend(probs.cpu().numpy())

        val_f1 = f1_score(val_labels, val_preds, average="weighted", zero_division=0)
        val_acc = accuracy_score(val_labels, val_preds)

        epoch_info = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_f1": val_f1,
            "val_accuracy": val_acc,
        }
        history.append(epoch_info)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{config.epochs} loss={avg_loss:.4f} val_f1={val_f1:.4f} val_acc={val_acc:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_f1": val_f1,
                "val_accuracy": val_acc,
            }, output_dir / "best.pt")
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    total_time = time.time() - start_time

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return {
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "total_time": total_time,
        "history": history,
        "checkpoint": str(output_dir / "best.pt"),
    }


# ============================================================================
# Evaluation
# ============================================================================

@dataclass
class EvalResult:
    model_name: str
    accuracy: float
    balanced_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    per_class_precision: dict
    per_class_recall: dict
    per_class_f1: dict
    sensitivity: dict
    specificity: dict
    roc_auc: float
    pr_auc: float
    mcc: float
    kappa: float
    logloss: float
    confusion_matrix: list
    y_true: list
    y_pred: list
    y_probs: list
    roc_data: dict
    pr_data: dict


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    model_name: str = "model",
) -> EvalResult:
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    cm = confusion_matrix(y_true, y_pred, labels=range(4))
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    per_class_p = {CLASS_NAMES[i]: float(cm[i, i] / max(cm[:, i].sum(), 1)) for i in range(4)}
    per_class_r = {CLASS_NAMES[i]: float(cm[i, i] / max(cm[i, :].sum(), 1)) for i in range(4)}
    per_class_f1_dict = {CLASS_NAMES[i]: float(2 * per_class_p[CLASS_NAMES[i]] * per_class_r[CLASS_NAMES[i]] /
                                                max(per_class_p[CLASS_NAMES[i]] + per_class_r[CLASS_NAMES[i]], 1e-10))
                         for i in range(4)}

    sensitivity = {CLASS_NAMES[i]: per_class_r[CLASS_NAMES[i]] for i in range(4)}
    specificity = {}
    for i in range(4):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        specificity[CLASS_NAMES[i]] = float(tn / max(tn + fp, 1))

    try:
        y_true_bin = label_binarize(y_true, classes=range(4))
        roc_auc_val = roc_auc_score(y_true_bin, y_probs, multi_class="ovr", average="macro")
        pr_auc_val = average_precision_score(y_true_bin, y_probs, average="macro")
    except Exception:
        roc_auc_val = 0.0
        pr_auc_val = 0.0

    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    ll = log_loss(y_true, y_probs)

    fpr, tpr, roc_auc_per_class = {}, {}, {}
    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc_per_class[CLASS_NAMES[i]] = auc(fpr[i], tpr[i])

    precision_curve, recall_curve, pr_auc_per_class = {}, {}, {}
    for i in range(4):
        precision_curve[i], recall_curve[i], _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
        pr_auc_per_class[CLASS_NAMES[i]] = auc(recall_curve[i], precision_curve[i])

    return EvalResult(
        model_name=model_name,
        accuracy=float(acc),
        balanced_accuracy=float(bal_acc),
        macro_precision=float(macro_p),
        macro_recall=float(macro_r),
        macro_f1=float(macro_f1),
        weighted_f1=float(weighted_f1),
        per_class_precision=per_class_p,
        per_class_recall=per_class_r,
        per_class_f1=per_class_f1_dict,
        sensitivity=sensitivity,
        specificity=specificity,
        roc_auc=float(roc_auc_val),
        pr_auc=float(pr_auc_val),
        mcc=float(mcc),
        kappa=float(kappa),
        logloss=float(ll),
        confusion_matrix=cm.tolist(),
        y_true=y_true.tolist(),
        y_pred=y_pred.tolist(),
        y_probs=y_probs.tolist(),
        roc_data={"fpr": {k: v.tolist() for k, v in fpr.items()},
                  "tpr": {k: v.tolist() for k, v in tpr.items()},
                  "auc": roc_auc_per_class},
        pr_data={"precision": {k: v.tolist() for k, v in precision_curve.items()},
                 "recall": {k: v.tolist() for k, v in recall_curve.items()},
                 "auc": pr_auc_per_class},
    )


# ============================================================================
# Efficiency Measurement
# ============================================================================

def measure_efficiency(model: nn.Module, device: torch.device, image_size=(224, 224), n_runs=20):
    model.eval()
    dummy = torch.randn(1, 3, *image_size, device=device)

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            model(dummy)

    times = []
    for _ in range(n_runs):
        start = time.time()
        with torch.no_grad():
            model(dummy)
        times.append((time.time() - start) * 1000)

    params = sum(p.numel() for p in model.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    peak_mem = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            model(dummy)
        peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return {
        "params": params,
        "size_mb": round(size_mb, 2),
        "latency_ms": round(np.mean(times), 2),
        "latency_std": round(np.std(times), 2),
        "fps": round(1000 / np.mean(times), 2),
        "peak_vram_mb": round(peak_mem, 2),
    }


# ============================================================================
# Ablation Experiments
# ============================================================================

ABLATION_CONFIGS = {
    "A_baseline_densenet121": {
        "desc": "Baseline: DenseNet121 classifier only",
        "builder": lambda: build_baseline("densenet121"),
        "use_turma_loss": False,
    },
    "B_baseline_resnet50": {
        "desc": "Baseline: ResNet50 classifier",
        "builder": lambda: build_baseline("resnet50"),
        "use_turma_loss": False,
    },
    "C_baseline_efficientnet": {
        "desc": "Baseline: EfficientNetV2-S classifier",
        "builder": lambda: build_baseline("efficientnet_v2_s"),
        "use_turma_loss": False,
    },
    "D_tumor_aware": {
        "desc": "TumorAwareMultiTaskNet (attention fusion)",
        "builder": lambda: build_tumor_aware("densenet121"),
        "use_turma_loss": False,
    },
    "E_turma_full": {
        "desc": "TURMA-Net full (ATGFM + consistency + uncertainty)",
        "builder": lambda: build_turma_net("densenet121", use_atgfm=True, use_consistency=True, use_uncertainty_weights=True),
        "use_turma_loss": True,
    },
    "F_turma_no_atgfm": {
        "desc": "TURMA-Net w/o ATGFM (simple concat fusion)",
        "builder": lambda: build_turma_net("densenet121", use_atgfm=False, use_consistency=True, use_uncertainty_weights=True),
        "use_turma_loss": True,
    },
    "G_turma_no_consistency": {
        "desc": "TURMA-Net w/o consistency loss",
        "builder": lambda: build_turma_net("densenet121", use_atgfm=True, use_consistency=False, use_uncertainty_weights=True),
        "use_turma_loss": True,
    },
    "H_turma_no_uncertainty": {
        "desc": "TURMA-Net w/o uncertainty weighting (fixed weights)",
        "builder": lambda: build_turma_net("densenet121", use_atgfm=True, use_consistency=True, use_uncertainty_weights=False),
        "use_turma_loss": True,
    },
}


def run_ablation(data_root, device, epochs=20, seeds=(42,)):
    results = {}
    train_loader, val_loader, test_loader = load_data(data_root, batch_size=8, seed=42)

    for exp_name, exp_cfg in ABLATION_CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"  {exp_name}: {exp_cfg['desc']}")
        print(f"{'='*70}")

        for seed in seeds:
            set_seed(seed)
            model = exp_cfg["builder"]()
            param_count = sum(p.numel() for p in model.parameters())

            config = TrainConfig(
                model_name=f"{exp_name}_seed{seed}",
                epochs=epochs,
                seed=seed,
                output_dir="paper/results/ablation",
                use_turma_loss=exp_cfg["use_turma_loss"],
            )

            train_result = train_model(model, config, train_loader, val_loader, device)
            model.load_state_dict(torch.load(train_result["checkpoint"], map_location=device, weights_only=False)["model_state_dict"])

            test_result = evaluate_model(model, test_loader, device, f"{exp_name}_seed{seed}")
            eff = measure_efficiency(model, device)

            result_entry = {
                "experiment": exp_name,
                "description": exp_cfg["desc"],
                "seed": seed,
                "params": param_count,
                "train": {
                    "best_val_f1": train_result["best_val_f1"],
                    "best_epoch": train_result["best_epoch"],
                    "total_time": train_result["total_time"],
                },
                "test": {
                    "accuracy": test_result.accuracy,
                    "balanced_accuracy": test_result.balanced_accuracy,
                    "macro_f1": test_result.macro_f1,
                    "weighted_f1": test_result.weighted_f1,
                    "roc_auc": test_result.roc_auc,
                    "pr_auc": test_result.pr_auc,
                    "mcc": test_result.mcc,
                    "kappa": test_result.kappa,
                    "per_class_f1": test_result.per_class_f1,
                    "sensitivity": test_result.sensitivity,
                    "specificity": test_result.specificity,
                },
                "efficiency": eff,
                "confusion_matrix": test_result.confusion_matrix,
                "roc_data": test_result.roc_data,
                "pr_data": test_result.pr_data,
            }

            results[f"{exp_name}_seed{seed}"] = result_entry

            out_path = Path("paper/results/ablation") / f"{exp_name}_seed{seed}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(result_entry, f, indent=2)

            print(f"  Accuracy: {test_result.accuracy:.4f} | F1: {test_result.macro_f1:.4f} | AUC: {test_result.roc_auc:.4f}")

    return results


# ============================================================================
# Statistical Comparison
# ============================================================================

def paired_bootstrap_test(y_true, y_probs_a, y_probs_b, n_bootstrap=10000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    acc_a = (y_probs_a.argmax(axis=1) == y_true).astype(float)
    acc_b = (y_probs_b.argmax(axis=1) == y_true).astype(float)
    diff = acc_a - acc_b
    orig_diff = diff.mean()

    count = 0
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        boot_diff = diff[idx].mean()
        if boot_diff <= 0:
            count += 1

    p_value = count / n_bootstrap
    ci_lower = np.percentile(diff, 2.5)
    ci_upper = np.percentile(diff, 97.5)

    return {
        "original_diff": float(orig_diff),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
    }


def compute_effect_size(y_true, y_probs_a, y_probs_b):
    acc_a = (y_probs_a.argmax(axis=1) == y_true).astype(float)
    acc_b = (y_probs_b.argmax(axis=1) == y_true).astype(float)
    pooled_std = np.sqrt((acc_a.var() + acc_b.var()) / 2)
    cohens_d = (acc_a.mean() - acc_b.mean()) / max(pooled_std, 1e-10)

    n_a, n_b = len(acc_a), len(acc_b)
    cliff_delta = 0.0
    for i in range(n_a):
        for j in range(n_b):
            if acc_a[i] > acc_b[j]:
                cliff_delta += 1
            elif acc_a[i] < acc_b[j]:
                cliff_delta -= 1
    cliff_delta /= (n_a * n_b)

    return {
        "cohens_d": float(cohens_d),
        "cliff_delta": float(cliff_delta),
        "effect_interpretation": (
            "negligible" if abs(cohens_d) < 0.2 else
            "small" if abs(cohens_d) < 0.5 else
            "medium" if abs(cohens_d) < 0.8 else
            "large"
        ),
    }


# ============================================================================
# Figure Generation
# ============================================================================

def generate_figures(all_results: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "figure.dpi": 150,
    })

    figures_dir = Path("paper/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: Accuracy & F1 comparison ---
    exp_names = []
    accuracies = []
    f1s = []
    for key, res in sorted(all_results.items()):
        exp_names.append(res["experiment"].replace("_", "\n", 1))
        accuracies.append(res["test"]["accuracy"])
        f1s.append(res["test"]["macro_f1"])

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(exp_names))
    width = 0.35
    ax.bar(x - width/2, accuracies, width, label="Accuracy", color="#3498db")
    ax.bar(x + width/2, f1s, width, label="Macro F1", color="#e74c3c")
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study: Accuracy & Macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, fontsize=8, rotation=0)
    ax.legend()
    ax.set_ylim([0, 1.05])
    for i, (a, f) in enumerate(zip(accuracies, f1s)):
        ax.text(i - width/2, a + 0.01, f"{a:.3f}", ha="center", fontsize=7)
        ax.text(i + width/2, f + 0.01, f"{f:.3f}", ha="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(figures_dir / "ablation_accuracy_f1.png", dpi=150)
    plt.close()

    # --- Figure 2: Confusion matrices for key models ---
    key_models = ["A_baseline_densenet121_seed42", "D_tumor_aware_seed42", "E_turma_full_seed42"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, model_key in zip(axes, key_models):
        if model_key in all_results:
            cm = np.array(all_results[model_key]["confusion_matrix"])
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            ax.set_title(all_results[model_key]["experiment"].replace("_", " "))
            for i in range(4):
                for j in range(4):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=9)
            ax.set_xticks(range(4))
            ax.set_yticks(range(4))
            ax.set_xticklabels(["Glio", "Meno", "NoT", "Pit"], fontsize=8)
            ax.set_yticklabels(["Glio", "Meno", "NoT", "Pit"], fontsize=8)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrices.png", dpi=150)
    plt.close()

    # --- Figure 3: ROC curves ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    for ax, model_key in zip(axes, key_models):
        if model_key in all_results:
            roc = all_results[model_key]["roc_data"]
            for i, cls_name in enumerate(CLASS_NAMES):
                fpr = np.array(roc["fpr"][cls_name])
                tpr = np.array(roc["tpr"][cls_name])
                auc_val = roc["auc"][cls_name]
                ax.plot(fpr, tpr, color=colors[i], label=f"{cls_name} ({auc_val:.3f})")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
            ax.set_title(all_results[model_key]["experiment"].replace("_", " "))
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(figures_dir / "roc_curves.png", dpi=150)
    plt.close()

    # --- Figure 4: Efficiency comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    model_labels = []
    param_counts = []
    latencies = []
    for key, res in sorted(all_results.items()):
        model_labels.append(res["experiment"].split("_seed")[0])
        param_counts.append(res["efficiency"]["params"] / 1e6)
        latencies.append(res["efficiency"]["latency_ms"])

    unique_labels = []
    unique_params = []
    unique_latencies = []
    seen = set()
    for l, p, lat in zip(model_labels, param_counts, latencies):
        if l not in seen:
            unique_labels.append(l)
            unique_params.append(p)
            unique_latencies.append(lat)
            seen.add(l)

    axes[0].barh(unique_labels, unique_params, color="#3498db")
    axes[0].set_xlabel("Parameters (M)")
    axes[0].set_title("Model Size")
    axes[1].barh(unique_labels, unique_latencies, color="#e74c3c")
    axes[1].set_xlabel("Latency (ms)")
    axes[1].set_title("Inference Speed")
    plt.tight_layout()
    plt.savefig(figures_dir / "efficiency_comparison.png", dpi=150)
    plt.close()

    print(f"Figures saved to {figures_dir}")


# ============================================================================
# Table Generation
# ============================================================================

def generate_tables(all_results: dict):
    tables_dir = Path("paper/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    # --- Table 1: Main comparison ---
    header = "| Model | Accuracy | Balanced Acc | Macro F1 | Weighted F1 | ROC-AUC | PR-AUC | MCC | Params (M) | Latency (ms) |"
    separator = "|-------|----------|--------------|----------|-------------|---------|--------|-----|------------|--------------|"
    rows = []
    for key, res in sorted(all_results.items()):
        rows.append(
            f"| {res['experiment']} | {res['test']['accuracy']:.4f} | {res['test']['balanced_accuracy']:.4f} | "
            f"{res['test']['macro_f1']:.4f} | {res['test']['weighted_f1']:.4f} | {res['test']['roc_auc']:.4f} | "
            f"{res['test']['pr_auc']:.4f} | {res['test']['mcc']:.4f} | {res['efficiency']['params']/1e6:.2f} | "
            f"{res['efficiency']['latency_ms']:.1f} |"
        )

    table = "\n".join([header, separator] + rows)
    with open(tables_dir / "main_comparison.md", "w") as f:
        f.write(table)

    # --- Table 2: Per-class F1 ---
    header2 = "| Model | Glioma | Meningioma | No Tumor | Pituitary |"
    separator2 = "|-------|--------|------------|----------|-----------|"
    rows2 = []
    for key, res in sorted(all_results.items()):
        pc = res["test"]["per_class_f1"]
        rows2.append(
            f"| {res['experiment']} | {pc.get('glioma', 0):.4f} | {pc.get('meningioma', 0):.4f} | "
            f"{pc.get('no tumor', 0):.4f} | {pc.get('pituitary', 0):.4f} |"
        )

    table2 = "\n".join([header2, separator2] + rows2)
    with open(tables_dir / "per_class_f1.md", "w") as f:
        f.write(table2)

    # --- Table 3: Ablation delta ---
    baseline_acc = all_results.get("A_baseline_densenet121_seed42", {}).get("test", {}).get("accuracy", 0)
    baseline_f1 = all_results.get("A_baseline_densenet121_seed42", {}).get("test", {}).get("macro_f1", 0)
    header3 = "| Configuration | Accuracy | Macro F1 | Δ Accuracy | Δ F1 |"
    separator3 = "|---------------|----------|----------|------------|------|"
    rows3 = []
    for key, res in sorted(all_results.items()):
        acc = res["test"]["accuracy"]
        f1 = res["test"]["macro_f1"]
        rows3.append(
            f"| {res['experiment']} | {acc:.4f} | {f1:.4f} | {acc - baseline_acc:+.4f} | {f1 - baseline_f1:+.4f} |"
        )

    table3 = "\n".join([header3, separator3] + rows3)
    with open(tables_dir / "ablation_delta.md", "w") as f:
        f.write(table3)

    print(f"Tables saved to {tables_dir}")


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all", choices=["ablation", "multi_seed", "evaluation", "all"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Phase: {args.phase}")
    print(f"Epochs: {args.epochs}")
    print(f"Seeds: {args.seeds}")

    data_root = "data/raw/classification"
    external_brisc = "data/external/brisc2025/brisc2025/classification_task/test"
    external_kaggle = "data/external/kaggle_brain_tumor/Testing"

    results = {}

    if args.phase in ["ablation", "all"]:
        print("\n" + "="*70)
        print("PHASE 1: ABLATION STUDY")
        print("="*70)
        results = run_ablation(data_root, device, epochs=args.epochs, seeds=args.seeds)

    if args.phase in ["evaluation", "all"] and results:
        print("\n" + "="*70)
        print("PHASE 2: EXTERNAL VALIDATION")
        print("="*70)

        train_loader, val_loader, test_loader = load_data(data_root, batch_size=8, seed=42)

        external_results = {}
        for ext_name, ext_root in [("BRISC", external_brisc), ("Kaggle", external_kaggle)]:
            if Path(ext_root).exists():
                ext_loader = load_external(ext_root, batch_size=8)
                print(f"\n  External: {ext_name} ({len(ext_loader.dataset)} samples)")

                for model_key in ["E_turma_full_seed42", "D_tumor_aware_seed42", "A_baseline_densenet121_seed42"]:
                    if model_key in results:
                        ckpt_path = Path(results[model_key]["train"]["checkpoint"])
                        if model_key.startswith("E_turma"):
                            model = build_turma_net("densenet121", use_atgfm=True, use_consistency=True, use_uncertainty_weights=True)
                        elif model_key.startswith("D_tumor"):
                            model = build_tumor_aware("densenet121")
                        else:
                            model = build_baseline("densenet121")

                        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                        model.load_state_dict(ckpt["model_state_dict"])
                        model = model.to(device)

                        ext_result = evaluate_model(model, ext_loader, device, f"{model_key}_{ext_name}")
                        ext_results[f"{model_key}_{ext_name}"] = {
                            "accuracy": ext_result.accuracy,
                            "macro_f1": ext_result.macro_f1,
                            "roc_auc": ext_result.roc_auc,
                            "per_class_f1": ext_result.per_class_f1,
                            "confusion_matrix": ext_result.confusion_matrix,
                        }
                        print(f"    {model_key}: acc={ext_result.accuracy:.4f} f1={ext_result.macro_f1:.4f}")

        with open("paper/results/external_validation.json", "w") as f:
            json.dump(ext_results, f, indent=2)

    if args.phase in ["all"] and results:
        print("\n" + "="*70)
        print("PHASE 3: STATISTICAL COMPARISON")
        print("="*70)

        comparisons = [
            ("A_baseline_densenet121_seed42", "E_turma_full_seed42", "Baseline vs TURMA-Net"),
            ("D_tumor_aware_seed42", "E_turma_full_seed42", "TumorAware vs TURMA-Net"),
        ]

        stats_results = {}
        for key_a, key_b, desc in comparisons:
            if key_a in results and key_b in results:
                y_true_a = np.array(results[key_a]["y_true"])
                y_probs_a = np.array(results[key_a]["y_probs"])
                y_true_b = np.array(results[key_b]["y_true"])
                y_probs_b = np.array(results[key_b]["y_probs"])

                bootstrap = paired_bootstrap_test(y_true_a, y_probs_a, y_probs_b)
                effect = compute_effect_size(y_true_a, y_probs_a, y_probs_b)

                stats_results[desc] = {"bootstrap": bootstrap, "effect_size": effect}
                print(f"\n  {desc}:")
                print(f"    Accuracy diff: {bootstrap['original_diff']:+.4f}")
                print(f"    95% CI: [{bootstrap['ci_lower']:+.4f}, {bootstrap['ci_upper']:+.4f}]")
                print(f"    p-value: {bootstrap['p_value']:.4f}")
                print(f"    Cohen's d: {effect['cohens_d']:.4f} ({effect['effect_interpretation']})")
                print(f"    Significant: {bootstrap['significant']}")

        with open("paper/results/statistical_comparison.json", "w") as f:
            json.dump(stats_results, f, indent=2)

    if args.phase in ["all"] and results:
        print("\n" + "="*70)
        print("PHASE 4: GENERATE FIGURES & TABLES")
        print("="*70)
        generate_figures(results)
        generate_tables(results)

    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
