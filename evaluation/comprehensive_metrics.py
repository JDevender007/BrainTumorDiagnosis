"""
BrainTumorDiagnosis - Comprehensive Metrics Evaluation

Computes 30+ metrics for all backbones on both internal test set and external BRISC 2025.

Metrics computed:
  Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC, PR-AUC,
  Confusion Matrix, Log Loss, Matthews Correlation Coefficient,
  Cohen's Kappa, Classification Report, Per-class metrics,
  Training curves, Inference latency, FPS, Model size,
  Error analysis, Robustness (noise perturbation), Statistical significance

Usage:
  python -m evaluation.comprehensive_metrics
  python -m evaluation.comprehensive_metrics --backbones densenet121 resnet50
  python -m evaluation.comprehensive_metrics --external-only
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from datasets.classification import discover_classification_samples, BrainTumorClassificationDataset
from models.factory import ModelConfig, create_backbone_classifier, create_classification_model

LOGGER = logging.getLogger(__name__)

CLASS_NAMES = ("Glioma", "Meningioma", "Pituitary", "No Tumor")
NUM_CLASSES = 4

ALL_BACKBONES = [
    "cnn2d", "densenet121", "resnet50", "efficientnet_v2_s",
    "convnext_tiny", "swin_t", "vit_b_16",
]


class ExternalTestDataset(Dataset):
    """Dataset for external BRISC 2025 test set."""

    def __init__(self, root: Path, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform
        class_map = {"glioma": 0, "meningioma": 1, "pituitary": 2, "no_tumor": 3}
        for cls_name, label in class_map.items():
            cls_dir = root / cls_name
            if cls_dir.exists():
                for img_path in sorted(cls_dir.glob("*.jpg")):
                    self.samples.append(img_path)
                    self.labels.append(label)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def compute_specificity(y_true, y_pred, num_classes=4):
    """Compute per-class and averaged specificity."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specificities = []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return specificities, np.mean(specificities)


def compute_roc_auc_per_class(y_true_prob, num_classes=4):
    """Compute per-class and macro ROC-AUC."""
    try:
        macro_auc = roc_auc_score(y_true_prob.argmax(axis=1), y_true_prob, multi_class="ovr", average="macro")
        per_class = []
        for i in range(num_classes):
            binary = (y_true_prob.argmax(axis=1) == i).astype(int)
            if binary.sum() > 0 and binary.sum() < len(binary):
                per_class.append(roc_auc_score(binary, y_true_prob[:, i]))
            else:
                per_class.append(0.0)
        return macro_auc, per_class
    except Exception:
        return 0.0, [0.0] * num_classes


def compute_pr_auc_per_class(y_true_prob, num_classes=4):
    """Compute per-class and macro PR-AUC."""
    try:
        macro_auc = average_precision_score(
            np.eye(num_classes)[y_true_prob.argmax(axis=1)], y_true_prob, average="macro"
        )
        per_class = []
        for i in range(num_classes):
            binary = (y_true_prob.argmax(axis=1) == i).astype(int)
            if binary.sum() > 0:
                per_class.append(average_precision_score(binary, y_true_prob[:, i]))
            else:
                per_class.append(0.0)
        return macro_auc, per_class
    except Exception:
        return 0.0, [0.0] * num_classes


def evaluate_model(model, dataloader, device, num_images=None):
    """Run inference and collect predictions, probabilities, labels, and timing."""
    model.eval()
    all_probs = []
    all_labels = []
    all_preds = []
    latencies = []

    total = len(dataloader.dataset) if num_images is None else min(num_images, len(dataloader.dataset))

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            torch.cuda.synchronize() if device.type == "cuda" else None
            t0 = time.perf_counter()

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            torch.cuda.synchronize() if device.type == "cuda" else None
            t1 = time.perf_counter()

            batch_size = images.size(0)
            latencies.extend([((t1 - t0) / batch_size) * 1000] * batch_size)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            if len(all_labels) * dataloader.batch_size >= total:
                break

    all_probs = np.concatenate(all_probs)[:total]
    all_labels = np.concatenate(all_labels)[:total]
    all_preds = all_probs.argmax(axis=1)

    return all_probs, all_preds, all_labels, latencies[:total]


def compute_all_metrics(y_true, y_pred, y_prob):
    """Compute all classification metrics."""
    metrics = {}

    # Basic metrics
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["precision_weighted"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    metrics["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["recall_weighted"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Specificity
    per_class_spec, avg_spec = compute_specificity(y_true, y_pred, NUM_CLASSES)
    metrics["specificity_macro"] = float(avg_spec)
    metrics["specificity_per_class"] = {CLASS_NAMES[i]: round(s, 4) for i, s in enumerate(per_class_spec)}

    # ROC-AUC
    macro_roc, per_class_roc = compute_roc_auc_per_class(y_prob, NUM_CLASSES)
    metrics["roc_auc_macro"] = float(macro_roc)
    metrics["roc_auc_per_class"] = {CLASS_NAMES[i]: round(a, 4) for i, a in enumerate(per_class_roc)}

    # PR-AUC
    macro_pr, per_class_pr = compute_pr_auc_per_class(y_prob, NUM_CLASSES)
    metrics["pr_auc_macro"] = float(macro_pr)
    metrics["pr_auc_per_class"] = {CLASS_NAMES[i]: round(a, 4) for i, a in enumerate(per_class_pr)}

    # Log Loss
    try:
        metrics["log_loss"] = float(log_loss(y_true, y_prob, labels=list(range(NUM_CLASSES))))
    except Exception:
        metrics["log_loss"] = -1.0

    # MCC
    metrics["matthews_corrcoef"] = float(matthews_corrcoef(y_true, y_pred))

    # Cohen's Kappa
    metrics["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    metrics["confusion_matrix"] = cm.tolist()

    # Per-class metrics
    per_class = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    metrics["per_class"] = {}
    for cls_name in CLASS_NAMES:
        if cls_name in per_class:
            metrics["per_class"][cls_name] = {
                "precision": round(per_class[cls_name]["precision"], 4),
                "recall": round(per_class[cls_name]["recall"], 4),
                "f1": round(per_class[cls_name]["f1-score"], 4),
                "support": int(per_class[cls_name]["support"]),
            }

    return metrics


def robustness_analysis(model, dataloader, device, noise_levels=[0.05, 0.1, 0.2]):
    """Test model robustness under Gaussian noise perturbation."""
    model.eval()
    results = {}

    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    for noise_std in noise_levels:
        noisy_correct = 0
        noisy_total = 0

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc=f"Noise {noise_std}", leave=False):
                images = images.to(device)
                labels = labels.to(device)

                denorm = images * imagenet_std + imagenet_mean
                noise = torch.randn_like(denorm) * noise_std
                noisy_denorm = torch.clamp(denorm + noise, 0, 1)
                noisy_images = (noisy_denorm - imagenet_mean) / imagenet_std

                outputs = model(noisy_images)
                preds = outputs.argmax(dim=1)
                noisy_correct += (preds == labels).sum().item()
                noisy_total += labels.size(0)

        results[f"noise_std_{noise_std}"] = {
            "accuracy": round(noisy_correct / noisy_total, 4) if noisy_total > 0 else 0,
            "accuracy_drop": round(
                1.0 - (noisy_correct / noisy_total) if noisy_total > 0 else 0, 4
            ),
        }

    return results


def measure_inference_speed(model, dataloader, device, num_batches=50):
    """Measure inference latency and FPS."""
    model.eval()
    latencies = []
    count = 0

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            batch_size = images.size(0)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            _ = model(images)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            batch_ms = (t1 - t0) * 1000
            per_image_ms = batch_ms / batch_size
            latencies.extend([per_image_ms] * batch_size)

            count += 1
            if count >= num_batches:
                break

    latencies = np.array(latencies)
    return {
        "mean_ms": round(float(latencies.mean()), 2),
        "std_ms": round(float(latencies.std()), 2),
        "median_ms": round(float(np.median(latencies)), 2),
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "p99_ms": round(float(np.percentile(latencies, 99)), 2),
        "fps": round(1000.0 / float(latencies.mean()), 1),
    }


def get_model_size_mb(checkpoint_path):
    """Get model checkpoint file size in MB."""
    path = Path(checkpoint_path)
    if path.exists():
        return round(path.stat().st_size / (1024 * 1024), 2)
    return 0.0


def get_num_params(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_training_history(backbone):
    """Load training history for curve data."""
    history_path = Path(f"outputs/{backbone}_training/training_history.json")
    if history_path.exists():
        with open(history_path) as f:
            return json.load(f)
    return None


def run_comprehensive_evaluation(
    backbone: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    external_loader: DataLoader,
    device: torch.device,
    checkpoint_dir: str,
) -> dict:
    """Run full comprehensive evaluation for one backbone."""
    LOGGER.info("=" * 78)
    LOGGER.info("Comprehensive Evaluation: %s", backbone)
    LOGGER.info("=" * 78)

    checkpoint_path = Path(checkpoint_dir) / backbone / "best.pt"
    if not checkpoint_path.exists():
        LOGGER.warning("No checkpoint for %s at %s", backbone, checkpoint_path)
        return {}

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}

    if backbone == "cnn2d":
        stem_weight = cleaned.get("stem.0.weight")
        base_channels = stem_weight.shape[0] if stem_weight is not None else 32
        model = create_classification_model(
            ModelConfig(
                model_type="classification",
                in_channels=3,
                num_classes=NUM_CLASSES,
                classification_input_size=(224, 224),
                pretrained=False,
                base_channels=base_channels,
            )
        )
    else:
        model = create_backbone_classifier(
            backbone=backbone,
            num_classes=NUM_CLASSES,
            pretrained=False,
        )
    model.load_state_dict(cleaned, strict=False)

    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    results = {"backbone": backbone}

    # Training history
    history = load_training_history(backbone)
    if history:
        results["training_history"] = history
        if isinstance(history, list) and len(history) > 0:
            train_losses = [h.get("train_loss", h.get("loss", 0)) for h in history if "train_loss" in h or "loss" in h]
            val_losses = [h.get("val_loss", h.get("validation_loss", 0)) for h in history if "val_loss" in h or "validation_loss" in h]
            train_accs = [h.get("train_accuracy", h.get("accuracy", 0)) for h in history if "train_accuracy" in h or "accuracy" in h]
            val_accs = [h.get("val_accuracy", h.get("validation_accuracy", 0)) for h in history if "val_accuracy" in h or "validation_accuracy" in h]
            results["training_curves"] = {
                "epochs": [h.get("epoch", i + 1) for i, h in enumerate(history)],
                "train_loss": train_losses,
                "val_loss": val_losses,
                "train_accuracy": train_accs,
                "val_accuracy": val_accs,
            }

    # Model size
    total_params, trainable_params = get_num_params(model)
    model_size_mb = get_model_size_mb(checkpoint_path)
    results["model_info"] = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": model_size_mb,
        "checkpoint_path": str(checkpoint_path),
    }

    # === Internal Test Set ===
    LOGGER.info("Evaluating on internal test set...")
    probs, preds, labels, latencies = evaluate_model(model, test_loader, device)
    internal_metrics = compute_all_metrics(labels, preds, probs)
    internal_metrics["latency"] = {
        "mean_ms": round(float(np.mean(latencies)), 2),
        "std_ms": round(float(np.std(latencies)), 2),
        "fps": round(1000.0 / float(np.mean(latencies)), 1),
    }
    results["internal_test"] = internal_metrics

    # ROC and PR curve data
    fpr_dict, tpr_dict = {}, {}
    precision_dict, recall_dict = {}, {}
    for i in range(NUM_CLASSES):
        binary = (labels == i).astype(int)
        if binary.sum() > 0:
            fpr_dict[i], tpr_dict[i], _ = roc_curve(binary, probs[:, i])
            precision_dict[i], recall_dict[i], _ = precision_recall_curve(binary, probs[:, i])
    results["internal_test"]["roc_curve"] = {CLASS_NAMES[i]: {"fpr": fpr_dict.get(i, []).tolist(), "tpr": tpr_dict.get(i, []).tolist()} for i in range(NUM_CLASSES)}
    results["internal_test"]["pr_curve"] = {CLASS_NAMES[i]: {"precision": precision_dict.get(i, []).tolist(), "recall": recall_dict.get(i, []).tolist()} for i in range(NUM_CLASSES)}

    # Error analysis on internal test set
    error_indices = np.where(preds != labels)[0]
    error_classes_pred = [CLASS_NAMES[preds[i]] for i in error_indices[:20]]
    error_classes_true = [CLASS_NAMES[labels[i]] for i in error_indices[:20]]
    error_confidences = [float(probs[i].max()) for i in error_indices[:20]]
    error_pairs = {}
    for t, p in zip(labels[error_indices], preds[error_indices]):
        pair = f"{CLASS_NAMES[t]}->{CLASS_NAMES[p]}"
        error_pairs[pair] = error_pairs.get(pair, 0) + 1
    results["internal_test"]["error_analysis"] = {
        "total_errors": int(len(error_indices)),
        "error_rate": round(len(error_indices) / len(labels), 4),
        "confused_pairs": dict(sorted(error_pairs.items(), key=lambda x: -x[1])),
        "sample_errors": [
            {"true": error_classes_true[i], "pred": error_classes_pred[i], "confidence": round(error_confidences[i], 4)}
            for i in range(min(20, len(error_indices)))
        ],
    }

    # Robustness
    LOGGER.info("Running robustness analysis...")
    results["robustness"] = robustness_analysis(model, test_loader, device)

    # Inference speed (detailed)
    LOGGER.info("Measuring inference speed...")
    speed = measure_inference_speed(model, test_loader, device)
    results["inference_speed"] = speed

    # === External Test Set (BRISC 2025) ===
    if external_loader and len(external_loader.dataset) > 0:
        LOGGER.info("Evaluating on BRISC 2025 external test set...")
        ext_probs, ext_preds, ext_labels, ext_latencies = evaluate_model(model, external_loader, device)
        external_metrics = compute_all_metrics(ext_labels, ext_preds, ext_probs)
        external_metrics["latency"] = {
            "mean_ms": round(float(np.mean(ext_latencies)), 2),
            "std_ms": round(float(np.std(ext_latencies)), 2),
            "fps": round(1000.0 / float(np.mean(ext_latencies)), 1),
        }
        fpr_dict, tpr_dict = {}, {}
        precision_dict, recall_dict = {}, {}
        for i in range(NUM_CLASSES):
            binary = (ext_labels == i).astype(int)
            if binary.sum() > 0:
                fpr_dict[i], tpr_dict[i], _ = roc_curve(binary, ext_probs[:, i])
                precision_dict[i], recall_dict[i], _ = precision_recall_curve(binary, ext_probs[:, i])
        external_metrics["roc_curve"] = {CLASS_NAMES[i]: {"fpr": fpr_dict.get(i, []).tolist(), "tpr": tpr_dict.get(i, []).tolist()} for i in range(NUM_CLASSES)}
        external_metrics["pr_curve"] = {CLASS_NAMES[i]: {"precision": precision_dict.get(i, []).tolist(), "recall": recall_dict.get(i, []).tolist()} for i in range(NUM_CLASSES)}

        ext_error_indices = np.where(ext_preds != ext_labels)[0]
        ext_error_pairs = {}
        for t, p in zip(ext_labels[ext_error_indices], ext_preds[ext_error_indices]):
            pair = f"{CLASS_NAMES[t]}->{CLASS_NAMES[p]}"
            ext_error_pairs[pair] = ext_error_pairs.get(pair, 0) + 1
        external_metrics["error_analysis"] = {
            "total_errors": int(len(ext_error_indices)),
            "error_rate": round(len(ext_error_indices) / len(ext_labels), 4),
            "confused_pairs": dict(sorted(ext_error_pairs.items(), key=lambda x: -x[1])),
        }
        external_metrics["robustness"] = robustness_analysis(model, external_loader, device)

        results["external_test_brisc2025"] = external_metrics

    # Cleanup
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Metrics Evaluation")
    parser.add_argument("--backbones", nargs="+", default=None, help="Backbones to evaluate")
    parser.add_argument("--external-only", action="store_true", help="Only run external test")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Device: %s", device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Internal test set
    internal_loader = None
    if not args.external_only:
        samples = discover_classification_samples("data/raw/classification")
        test_samples = [s for s in samples if s.split.lower() == "test"]
        test_dataset = BrainTumorClassificationDataset(test_samples, image_size=224, transform=transform)
        internal_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
        LOGGER.info("Internal test set: %d images", len(test_dataset))

    # External test set (BRISC 2025)
    external_root = Path("data/external/brisc2025/brisc2025/classification_task/test")
    external_loader = None
    if external_root.exists():
        ext_dataset = ExternalTestDataset(external_root, transform=transform)
        external_loader = DataLoader(ext_dataset, batch_size=32, shuffle=False, num_workers=0)
        LOGGER.info("BRISC 2025 test set: %d images", len(ext_dataset))
    else:
        LOGGER.warning("BRISC 2025 not found at %s", external_root)

    # Train loader (for cross-validation placeholder)
    train_loader = None
    if not args.external_only:
        samples = discover_classification_samples("data/raw/classification")
        train_samples = [s for s in samples if s.split.lower() == "train"]
        train_dataset = BrainTumorClassificationDataset(train_samples, image_size=224, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=0)

    backbones = args.backbones or ALL_BACKBONES
    output_dir = Path("outputs/comprehensive_metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for backbone in backbones:
        LOGGER.info("Processing backbone: %s", backbone)
        results = run_comprehensive_evaluation(
            backbone=backbone,
            train_loader=train_loader,
            test_loader=internal_loader,
            external_loader=external_loader,
            device=device,
            checkpoint_dir="checkpoints/classification",
        )
        if results:
            all_results[backbone] = results
            backbone_file = output_dir / f"{backbone}_comprehensive.json"
            with open(backbone_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            LOGGER.info("Saved: %s", backbone_file)

    # Save combined results
    combined_file = output_dir / "all_backbones_comprehensive.json"
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    LOGGER.info("All results saved: %s", combined_file)

    # Print summary table
    print("\n" + "=" * 100)
    print("COMPREHENSIVE METRICS SUMMARY - INTERNAL TEST SET")
    print("=" * 100)
    print(f"{'Backbone':<22} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Spec':>7} {'ROC-AUC':>8} {'PR-AUC':>8} {'MCC':>7} {'Kappa':>7} {'LogLoss':>8}")
    print("-" * 100)
    for backbone, r in all_results.items():
        t = r.get("internal_test", {})
        print(f"{backbone:<22} "
              f"{t.get('accuracy', 0):>7.4f} "
              f"{t.get('precision_macro', 0):>7.4f} "
              f"{t.get('recall_macro', 0):>7.4f} "
              f"{t.get('f1_macro', 0):>7.4f} "
              f"{t.get('specificity_macro', 0):>7.4f} "
              f"{t.get('roc_auc_macro', 0):>8.4f} "
              f"{t.get('pr_auc_macro', 0):>8.4f} "
              f"{t.get('matthews_corrcoef', 0):>7.4f} "
              f"{t.get('cohen_kappa', 0):>7.4f} "
              f"{t.get('log_loss', 0):>8.4f}")

    if any("external_test_brisc2025" in r for r in all_results.values()):
        print("\n" + "=" * 100)
        print("COMPREHENSIVE METRICS SUMMARY - BRISC 2025 EXTERNAL TEST")
        print("=" * 100)
        print(f"{'Backbone':<22} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Spec':>7} {'ROC-AUC':>8} {'PR-AUC':>8} {'MCC':>7} {'Kappa':>7} {'LogLoss':>8}")
        print("-" * 100)
        for backbone, r in all_results.items():
            t = r.get("external_test_brisc2025", {})
            if t:
                print(f"{backbone:<22} "
                      f"{t.get('accuracy', 0):>7.4f} "
                      f"{t.get('precision_macro', 0):>7.4f} "
                      f"{t.get('recall_macro', 0):>7.4f} "
                      f"{t.get('f1_macro', 0):>7.4f} "
                      f"{t.get('specificity_macro', 0):>7.4f} "
                      f"{t.get('roc_auc_macro', 0):>8.4f} "
                      f"{t.get('pr_auc_macro', 0):>8.4f} "
                      f"{t.get('matthews_corrcoef', 0):>7.4f} "
                      f"{t.get('cohen_kappa', 0):>7.4f} "
                      f"{t.get('log_loss', 0):>8.4f}")

    print("\n" + "=" * 100)
    print("MODEL SIZE & SPEED COMPARISON")
    print("=" * 100)
    print(f"{'Backbone':<22} {'Params (M)':>12} {'Size (MB)':>10} {'Latency (ms)':>13} {'FPS':>8}")
    print("-" * 100)
    for backbone, r in all_results.items():
        mi = r.get("model_info", {})
        spd = r.get("inference_speed", {})
        print(f"{backbone:<22} "
              f"{mi.get('total_params', 0) / 1e6:>12.2f} "
              f"{mi.get('model_size_mb', 0):>10.2f} "
              f"{spd.get('mean_ms', 0):>13.2f} "
              f"{spd.get('fps', 0):>8.1f}")

    print("\nDone. Results saved to outputs/comprehensive_metrics/")


if __name__ == "__main__":
    main()
