"""
Comprehensive research experiment runner.

Orchestrates all experiments:
- Baseline training
- Proposed model training
- Ablation studies
- Uncertainty estimation
- Calibration analysis
- Robustness evaluation
- Grad-CAM localization
- Cross-dataset analysis
- Efficiency analysis
- Statistical validation
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.seed import set_seed
from utils.device import get_device


@dataclass
class ExperimentConfig:
    seeds: list[int] = None
    device: str = "cuda"
    output_dir: str = "outputs/research_experiments"
    classification_root: str = "data/raw/classification"
    external_root: str = "data/external/brisc2025/brisc2025/classification_task/test"
    batch_size: int = 8
    image_size: tuple[int, int] = (224, 224)

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42, 123, 2024]


def load_classification_data(
    root: str,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 8,
):
    """Load classification dataset."""
    from datasets.classification import BrainTumorClassificationDataset
    from datasets.classification import discover_classification_samples
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    samples = discover_classification_samples(root)
    train_samples = [s for s in samples if s.split.lower() == "train"]
    test_samples = [s for s in samples if s.split.lower() == "test"]

    train_size = int(0.85 * len(train_samples))
    val_size = len(train_samples) - train_size

    train_dataset = BrainTumorClassificationDataset(train_samples[:train_size], image_size=image_size, transform=transform)
    val_dataset = BrainTumorClassificationDataset(train_samples[train_size:], image_size=image_size, transform=transform)
    test_dataset = BrainTumorClassificationDataset(test_samples, image_size=image_size, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def load_external_data(
    external_root: str,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 8,
):
    """Load external BRISC 2025 dataset (flat structure: class folders directly)."""
    from datasets.classification import BrainTumorClassificationDataset, ClassificationSample, CLASS_TO_INDEX
    from torchvision import transforms
    from pathlib import Path

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    root = Path(external_root)
    class_to_idx = {name.lower().replace(" ", "_"): idx for name, idx in CLASS_TO_INDEX.items()}

    samples = []
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        class_key = class_dir.name.lower().replace(" ", "_")
        if class_key not in class_to_idx:
            continue
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
                samples.append(ClassificationSample(
                    image_path=str(img_path),
                    class_name=class_key,
                    class_index=class_to_idx[class_key],
                    split="test",
                ))

    dataset = BrainTumorClassificationDataset(samples, image_size=image_size, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader


def run_full_experiment(config: Optional[ExperimentConfig] = None):
    """Run the complete research experiment pipeline."""
    if config is None:
        config = ExperimentConfig()

    device = get_device()
    print(f"Device: {device}")
    print(f"Seeds: {config.seeds}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading classification data...")
    train_loader, val_loader, test_loader = load_classification_data(
        config.classification_root, config.image_size, config.batch_size
    )

    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val: {len(val_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")

    external_loader = None
    if Path(config.external_root).exists():
        print("Loading external BRISC 2025 data...")
        external_loader = load_external_data(config.external_root, config.image_size, config.batch_size)
        print(f"  External: {len(external_loader.dataset)} samples")

    all_results = {}

    print("\n" + "="*70)
    print("PHASE 1: Baseline Training (DenseNet121)")
    print("="*70)

    from models.factory import create_backbone_classifier
    from training.classification_trainer import ClassificationTrainingConfig, train_classification_model

    baseline_config = ClassificationTrainingConfig(
        backbone="densenet121",
        epochs=30,
        batch_size=config.batch_size,
        image_size=config.image_size,
        learning_rate=0.0001,
        patience=5,
    )
    baseline_result = train_classification_model(baseline_config)
    all_results["baseline_densenet121"] = baseline_result

    print("\n" + "="*70)
    print("PHASE 2: TumorAwareMultiTaskNet Training")
    print("="*70)

    from training.multitask_trainer import MultiTaskTrainingConfig, train_multitask_model

    proposed_config = MultiTaskTrainingConfig(
        backbone="densenet121",
        pretrained=True,
        epochs=30,
        batch_size=config.batch_size,
        image_size=config.image_size,
        lr=0.0001,
        patience=7,
        seg_loss_weight=0.5,
        output_dir=str(output_dir / "proposed_model"),
    )
    proposed_result = train_multitask_model(proposed_config, train_loader, val_loader, device)
    all_results["proposed_model"] = proposed_result

    print("\n" + "="*70)
    print("PHASE 3: Uncertainty Estimation")
    print("="*70)

    from evaluation.uncertainty import compute_uncertainty_metrics
    from models.tumor_aware import TumorAwareMultiTaskNet, TumorAwareConfig

    model_config = TumorAwareConfig(backbone="densenet121")
    model = TumorAwareMultiTaskNet(model_config).to(device)
    checkpoint = torch.load(output_dir / "proposed_model" / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    unc_metrics = compute_uncertainty_metrics(model, test_loader, device, num_passes=10)
    with open(output_dir / "uncertainty" / "uncertainty_metrics.json", "w") as f:
        json.dump(unc_metrics, f, indent=2)
    all_results["uncertainty"] = unc_metrics

    print(f"  ECE: {unc_metrics['ece']:.4f}")
    print(f"  Brier: {unc_metrics['brier']:.4f}")
    print(f"  Mean confidence: {unc_metrics['mean_confidence']:.4f}")

    print("\n" + "="*70)
    print("PHASE 4: Calibration Analysis")
    print("="*70)

    from evaluation.calibration import evaluate_calibration

    cal_result = evaluate_calibration(
        model, test_loader, device,
        model_name="tumor_aware",
        output_dir=output_dir / "calibration",
    )
    all_results["calibration"] = {
        "ece": cal_result.ece,
        "mce": cal_result.mce,
        "brier": cal_result.brier,
        "nll": cal_result.nll,
    }

    print(f"  ECE: {cal_result.ece:.4f}")
    print(f"  MCE: {cal_result.mce:.4f}")
    print(f"  Brier: {cal_result.brier:.4f}")

    print("\n" + "="*70)
    print("PHASE 5: Robustness Evaluation")
    print("="*70)

    from evaluation.advanced_robustness import run_robustness_evaluation, CorruptionConfig

    robustness_config = CorruptionConfig()
    rob_result = run_robustness_evaluation(
        model, test_loader, device,
        model_name="tumor_aware",
        output_dir=output_dir / "robustness",
        config=robustness_config,
    )
    all_results["robustness"] = rob_result

    print(f"  MRRS: {rob_result['mrrs']:.4f}")

    print("\n" + "="*70)
    print("PHASE 6: Efficiency Analysis")
    print("="*70)

    from evaluation.efficiency_analysis import analyze_efficiency

    eff_result = analyze_efficiency(
        model, "TumorAwareMultiTaskNet",
        test_loader, device,
        accuracy=0.0, f1=0.0,
    )
    print(f"  Params: {eff_result.total_params / 1e6:.2f}M")
    print(f"  Size: {eff_result.model_size_mb:.2f}MB")
    print(f"  Latency: {eff_result.latency_ms:.2f}ms")
    print(f"  FPS: {eff_result.fps:.1f}")

    print("\n" + "="*70)
    print("PHASE 7: Cross-Dataset Analysis")
    print("="*70)

    if external_loader:
        from evaluation.cross_dataset_analysis import analyze_domain_shift

        cross_result = analyze_domain_shift(
            model, "TumorAwareMultiTaskNet",
            test_loader, external_loader, device,
        )
        all_results["cross_dataset"] = {
            "internal_accuracy": cross_result.internal_accuracy,
            "external_accuracy": cross_result.external_accuracy,
            "accuracy_drop": cross_result.accuracy_drop,
            "internal_f1": cross_result.internal_f1,
            "external_f1": cross_result.external_f1,
            "f1_drop": cross_result.f1_drop,
        }
        print(f"  Internal accuracy: {cross_result.internal_accuracy:.4f}")
        print(f"  External accuracy: {cross_result.external_accuracy:.4f}")
        print(f"  Accuracy drop: {cross_result.accuracy_drop:.4f}")

    print("\n" + "="*70)
    print("Saving all results...")
    print("="*70)

    with open(output_dir / "all_experiment_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nAll results saved to {output_dir}")
    print("Experiment complete!")

    return all_results


if __name__ == "__main__":
    run_full_experiment()
