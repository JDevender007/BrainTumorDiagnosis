"""
Multi-seed validation: train proposed model across multiple seeds,
collect per-seed metrics, run statistical validation.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.seed import set_seed
from utils.device import get_device
from scripts.run_research_experiments import load_classification_data, load_external_data


def run_multi_seed_validation(
    seeds: list[int] = (123, 2024),
    classification_root: str = "data/raw/classification",
    external_root: str = "data/external/brisc2025/brisc2025/classification_task/test",
    batch_size: int = 8,
    image_size: tuple[int, int] = (224, 224),
    epochs: int = 15,
    output_dir: str = "outputs/research_experiments/multi_seed",
    model_type: str = "turma_net",  # "turma_net" or "tumor_aware"
):
    device = get_device()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading classification data...")
    train_loader, val_loader, test_loader = load_classification_data(
        classification_root, image_size, batch_size
    )
    print(f"  Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    external_loader = None
    if Path(external_root).exists():
        external_loader = load_external_data(external_root, image_size, batch_size)
        print(f"  External: {len(external_loader.dataset)}")

    all_seed_results = {}
    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"SEED {seed}")
        print(f"{'='*70}")

        set_seed(seed)

        if model_type == "turma_net":
            from training.multitask_trainer import MultiTaskTrainingConfig, train_multitask_model
            from models.novel.turma_net import TurmaNet, TurmaNetConfig
            from evaluation.uncertainty import compute_uncertainty_metrics
            from evaluation.calibration import evaluate_calibration
            from evaluation.advanced_robustness import run_robustness_evaluation, CorruptionConfig
            from evaluation.efficiency_analysis import analyze_efficiency

            seed_dir = output_path / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            config = MultiTaskTrainingConfig(
                backbone="densenet121",
                pretrained=True,
                epochs=epochs,
                batch_size=batch_size,
                image_size=image_size,
                lr=0.0001,
                patience=7,
                seg_loss_weight=0.5,
                output_dir=str(seed_dir / "proposed_model"),
                seed=seed,
            )
            result = train_multitask_model(config, train_loader, val_loader, device)

            model_config = TurmaNetConfig(backbone="densenet121", num_classes=4, segmentation_classes=4)
            model = TurmaNet(model_config).to(device)
            ckpt_path = seed_dir / "proposed_model" / "best.pt"
        else:
            from training.multitask_trainer import MultiTaskTrainingConfig, train_multitask_model
            from models.tumor_aware import TumorAwareMultiTaskNet, TumorAwareConfig
            from evaluation.uncertainty import compute_uncertainty_metrics
            from evaluation.calibration import evaluate_calibration
            from evaluation.advanced_robustness import run_robustness_evaluation, CorruptionConfig
            from evaluation.efficiency_analysis import analyze_efficiency

            seed_dir = output_path / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            config = MultiTaskTrainingConfig(
                backbone="densenet121",
                pretrained=True,
                epochs=epochs,
                batch_size=batch_size,
                image_size=image_size,
                lr=0.0001,
                patience=7,
                seg_loss_weight=0.5,
                output_dir=str(seed_dir / "proposed_model"),
                seed=seed,
            )
            result = train_multitask_model(config, train_loader, val_loader, device)

            model_config = TumorAwareConfig(backbone="densenet121")
            model = TumorAwareMultiTaskNet(model_config).to(device)
            ckpt_path = seed_dir / "proposed_model" / "best.pt"
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])

        seed_metrics = {"best_val_f1": result.best_val_f1, "best_epoch": result.best_epoch}

        unc = compute_uncertainty_metrics(model, test_loader, device, num_passes=10)
        seed_metrics["uncertainty"] = unc

        cal = evaluate_calibration(model, test_loader, device, model_name=f"tumor_aware_seed{seed}", output_dir=seed_dir / "calibration")
        seed_metrics["calibration"] = {"ece": cal.ece, "mce": cal.mce, "brier": cal.brier, "nll": cal.nll}

        rob = run_robustness_evaluation(model, test_loader, device, model_name=f"tumor_aware_seed{seed}", output_dir=seed_dir / "robustness", config=CorruptionConfig())
        seed_metrics["robustness"] = rob

        eff = analyze_efficiency(model, "TumorAwareMultiTaskNet", test_loader, device, accuracy=0.0, f1=result.best_val_f1)
        seed_metrics["efficiency"] = {"params": eff.total_params, "size_mb": eff.model_size_mb, "latency_ms": eff.latency_ms, "fps": eff.fps}

        if external_loader:
            from evaluation.cross_dataset_analysis import analyze_domain_shift
            cross = analyze_domain_shift(model, "TumorAwareMultiTaskNet", test_loader, external_loader, device)
            seed_metrics["cross_dataset"] = {
                "internal_accuracy": cross.internal_accuracy,
                "external_accuracy": cross.external_accuracy,
                "accuracy_drop": cross.accuracy_drop,
            }

        with open(seed_dir / "seed_results.json", "w") as f:
            json.dump(seed_metrics, f, indent=2, default=str)

        all_seed_results[f"seed_{seed}"] = seed_metrics
        print(f"  Seed {seed}: val_f1={result.best_val_f1:.4f}")

    with open(output_path / "all_seed_results.json", "w") as f:
        json.dump(all_seed_results, f, indent=2, default=str)

    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS")
    print("="*70)

    from evaluation.statistical_validation import compute_statistics, generate_statistical_table

    stats_input = {}
    for seed_key, seed_data in all_seed_results.items():
        for section in ["uncertainty", "calibration", "robustness", "efficiency", "cross_dataset"]:
            if section in seed_data:
                for k, v in seed_data[section].items():
                    if isinstance(v, (int, float)):
                        key = f"proposed_{section}_{k}"
                        if key not in stats_input:
                            stats_input[key] = []
                        stats_input[key].append(v)

    for metric_name, values in stats_input.items():
        stats = compute_statistics(values, model_name="proposed", metric_name=metric_name)
        print(f"  {metric_name}: {stats.mean:.4f} +/- {stats.std:.4f} (95% CI: [{stats.ci_lower:.4f}, {stats.ci_upper:.4f}])")

    generate_statistical_table(stats_input, output_dir=str(output_path))

    print(f"\nAll multi-seed results saved to {output_path}")
    return all_seed_results


if __name__ == "__main__":
    run_multi_seed_validation()
