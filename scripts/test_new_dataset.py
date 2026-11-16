"""
Test the trained model on a new external dataset.

Dataset: Kaggle Brain Tumor MRI Dataset
Source: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
Mirror: https://zenodo.org/records/12735702

Usage:
  python scripts/test_new_dataset.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.tumor_aware import TumorAwareMultiTaskNet, TumorAwareConfig
from utils.device import get_device


CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

NEW_DATASET_DIR = Path("data/external/kaggle_brain_tumor")
OUTPUT_DIR = Path("outputs/new_dataset_test")


class NewBrainTumorDataset(Dataset):
    """Dataset loader for the Kaggle Brain Tumor MRI Dataset."""

    def __init__(self, root_dir: str | Path, transform=None):
        self.root = Path(root_dir)
        self.transform = transform
        self.samples = []

        # Map folder names to our class indices
        folder_to_class = {
            "glioma": 0,
            "meningioma": 1,
            "pituitary": 2,
            "notumor": 3,  # Kaggle uses "notumor" (no space)
            "no_tumor": 3,  # BRISC uses "no_tumor"
            "no tumor": 3,  # BraTS uses "no tumor"
        }

        for class_folder, class_idx in folder_to_class.items():
            class_dir = self.root / class_folder
            if not class_dir.exists():
                continue

            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
                    self.samples.append((img_path, class_idx))

        print(f"  Found {len(self.samples)} images in {self.root}")

        # Count per class
        for class_name in CLASS_NAMES:
            class_idx = CLASS_TO_IDX[class_name]
            count = sum(1 for _, idx in self.samples if idx == class_idx)
            print(f"    {class_name}: {count} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def load_model(device: torch.device):
    """Load the trained TumorAwareMultiTaskNet model."""
    print("Loading trained model...")

    config = TumorAwareConfig(
        backbone="densenet121",
        pretrained=False,
        num_classes=4,
        segmentation_classes=4,
        dropout=0.3,
        fusion_dim=256,
        use_attention_fusion=True,
        use_segmentation_head=True,
        use_uncertainty_head=True,
    )

    model = TumorAwareMultiTaskNet(config).to(device)

    checkpoint_path = Path("outputs/research_experiments/proposed_model/best.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"  Model loaded from: {checkpoint_path}")
    if "epoch" in checkpoint:
        print(f"  Trained for {checkpoint['epoch']} epochs")
    if "best_val_f1" in checkpoint:
        print(f"  Best validation F1: {checkpoint['best_val_f1']:.4f}")

    return model


def evaluate_model(model, dataloader, device, num_mc_passes=10):
    """Evaluate model with MC Dropout uncertainty estimation."""
    print(f"\nEvaluating with {num_mc_passes} MC Dropout passes...")

    model.train()  # Keep dropout active for uncertainty

    all_preds = []
    all_labels = []
    all_probs = []
    all_entropies = []
    all_confidences = []

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            # MC Dropout: multiple forward passes
            batch_probs = []
            for _ in range(num_mc_passes):
                outputs = model(images)
                if isinstance(outputs, dict):
                    probs = F.softmax(outputs["logits"], dim=1)
                else:
                    probs = F.softmax(outputs, dim=1)
                batch_probs.append(probs)

            # Stack: [num_passes, batch_size, num_classes]
            stacked = torch.stack(batch_probs)
            mean_probs = stacked.mean(dim=0)

            # Compute uncertainty
            entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=1)
            confidence = mean_probs.max(dim=1).values
            preds = mean_probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(mean_probs.cpu().numpy())
            all_entropies.extend(entropy.cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * len(images)}/{len(dataloader.dataset)} images")

    elapsed = time.time() - start_time

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    probs = np.array(all_probs)
    entropies = np.array(all_entropies)
    confidences = np.array(all_confidences)

    # Compute metrics
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    precision = precision_score(labels, preds, average="weighted", zero_division=0)
    recall = recall_score(labels, preds, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(labels, preds)
    cm = confusion_matrix(labels, preds)

    # Per-class accuracy
    per_class_acc = {}
    for i, class_name in enumerate(CLASS_NAMES):
        mask = labels == i
        if mask.sum() > 0:
            per_class_acc[class_name] = float((preds[mask] == labels[mask]).mean())
        else:
            per_class_acc[class_name] = 0.0

    # Uncertainty metrics
    mean_entropy = float(entropies.mean())
    mean_confidence = float(confidences.mean())
    mean_confidence_correct = float(confidences[preds == labels].mean()) if (preds == labels).any() else 0.0
    mean_confidence_incorrect = float(confidences[preds != labels].mean()) if (preds != labels).any() else 0.0

    results = {
        "dataset": "Kaggle Brain Tumor MRI Dataset (Zenodo Mirror)",
        "dataset_source": "https://zenodo.org/records/12735702",
        "total_images": len(dataloader.dataset),
        "num_mc_passes": num_mc_passes,
        "inference_time_sec": round(elapsed, 2),
        "images_per_sec": round(len(dataloader.dataset) / elapsed, 1),
        "metrics": {
            "accuracy": round(float(accuracy), 4),
            "f1_weighted": round(float(f1), 4),
            "f1_macro": round(float(f1_macro), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "mcc": round(float(mcc), 4),
        },
        "per_class_accuracy": {k: round(v, 4) for k, v in per_class_acc.items()},
        "uncertainty": {
            "mean_entropy": round(mean_entropy, 4),
            "mean_confidence": round(mean_confidence, 4),
            "mean_confidence_correct": round(mean_confidence_correct, 4),
            "mean_confidence_incorrect": round(mean_confidence_incorrect, 4),
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            labels, preds, target_names=CLASS_NAMES, zero_division=0
        ),
    }

    return results


def print_results(results):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("RESULTS: Kaggle Brain Tumor MRI Dataset")
    print("=" * 70)

    print(f"\nDataset: {results['dataset']}")
    print(f"Total images: {results['total_images']}")
    print(f"MC Dropout passes: {results['num_mc_passes']}")
    print(f"Inference time: {results['inference_time_sec']}s ({results['images_per_sec']} img/s)")

    print(f"\n--- Classification Metrics ---")
    for metric, value in results["metrics"].items():
        print(f"  {metric:20s}: {value:.4f}")

    print(f"\n--- Per-Class Accuracy ---")
    for class_name, acc in results["per_class_accuracy"].items():
        print(f"  {class_name:20s}: {acc:.4f}")

    print(f"\n--- Uncertainty ---")
    for metric, value in results["uncertainty"].items():
        print(f"  {metric:30s}: {value:.4f}")

    print(f"\n--- Confusion Matrix ---")
    cm = np.array(results["confusion_matrix"])
    header = "".join(f"{name:>12s}" for name in CLASS_NAMES)
    print(f"  {'':12s}{header}")
    for i, row in enumerate(cm):
        row_str = "".join(f"{val:12d}" for val in row)
        print(f"  {CLASS_NAMES[i]:12s}{row_str}")

    print(f"\n--- Classification Report ---")
    print(results["classification_report"])


def main():
    """Main function."""
    print("=" * 70)
    print("Testing Trained Model on New Dataset")
    print("Kaggle Brain Tumor MRI Dataset (Zenodo Mirror)")
    print("=" * 70)

    device = get_device()
    print(f"Device: {device}")

    # Load model
    model = load_model(device)

    # Create dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Test on the Testing split
    test_dir = NEW_DATASET_DIR / "Testing"
    if not test_dir.exists():
        print(f"ERROR: Test directory not found: {test_dir}")
        return

    print(f"\nLoading dataset from: {test_dir}")
    dataset = NewBrainTumorDataset(test_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    # Evaluate
    results = evaluate_model(model, dataloader, device, num_mc_passes=10)

    # Print results
    print_results(results)

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "kaggle_dataset_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Also test on Training split to see train vs test performance
    print("\n" + "=" * 70)
    print("BONUS: Testing on Training split (to check overfitting)")
    print("=" * 70)

    train_dir = NEW_DATASET_DIR / "Training"
    if train_dir.exists():
        train_dataset = NewBrainTumorDataset(train_dir, transform=transform)
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=0)
        train_results = evaluate_model(model, train_dataloader, device, num_mc_passes=10)

        print(f"\nTraining Split Accuracy: {train_results['metrics']['accuracy']:.4f}")
        print(f"Testing Split Accuracy:  {results['metrics']['accuracy']:.4f}")
        print(f"Gap: {train_results['metrics']['accuracy'] - results['metrics']['accuracy']:.4f}")

        # Save training results too
        train_output_file = OUTPUT_DIR / "kaggle_dataset_train_results.json"
        with open(train_output_file, "w") as f:
            json.dump(train_results, f, indent=2)
        print(f"Training results saved to: {train_output_file}")


if __name__ == "__main__":
    main()
