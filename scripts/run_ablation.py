"""
Full ablation study: Experiments A-J.
Runs after main experiment completes. Uses backbone from trained models
where available, trains fresh for missing variants.
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
from evaluation.ablation_runner import run_ablation_experiment, generate_ablation_table, AblationResult
from models.tumor_aware import TumorAwareMultiTaskNet, TumorAwareConfig
from models.factory import create_backbone_classifier
from scripts.run_research_experiments import load_classification_data


EXPERIMENTS = {
    "A_baseline": {
        "desc": "Baseline: CNN-2D, image-level classification only",
        "backbone": "cnn2d",
        "use_tumor_aware": False,
        "use_pretrained": None,
    },
    "B_baseline_simclr": {
        "desc": "Baseline + SimCLR pretraining",
        "backbone": "cnn2d",
        "use_tumor_aware": False,
        "use_pretrained": None,
    },
    "C_baseline_seg": {
        "desc": "Baseline + segmentation head (no fusion)",
        "backbone": "densenet121",
        "use_tumor_aware": True,
        "use_pretrained": True,
    },
    "D_tumor_aware": {
        "desc": "TumorAware (DenseNet121, no SimCLR)",
        "backbone": "densenet121",
        "use_tumor_aware": True,
        "use_pretrained": True,
    },
    "E_tumor_aware_simclr": {
        "desc": "TumorAware + SimCLR pretraining",
        "backbone": "densenet121",
        "use_tumor_aware": True,
        "use_pretrained": True,
    },
    "F_full_model": {
        "desc": "Full TumorAwareMultiTaskNet (DenseNet121)",
        "backbone": "densenet121",
        "use_tumor_aware": True,
        "use_pretrained": True,
    },
    "G_no_fusion": {
        "desc": "Full model without cross-attention fusion",
        "backbone": "densenet121",
        "use_tumor_aware": True,
        "use_pretrained": True,
    },
    "H_no_seg_supervision": {
        "desc": "Full model without segmentation supervision",
        "backbone": "densenet121",
        "use_tumor_aware": True,
        "use_pretrained": True,
    },
    "I_no_tumor_features": {
        "desc": "Full model without tumor-aware features",
        "backbone": "densenet121",
        "use_tumor_aware": True,
        "use_pretrained": True,
    },
    "J_no_simclr": {
        "desc": "Full model without SimCLR pretraining",
        "backbone": "densenet121",
        "use_tumor_aware": True,
        "use_pretrained": True,
    },
    "K_turma_full": {
        "desc": "Full TURMA-Net (proposed novel model)",
        "backbone": "densenet121",
        "use_turma_net": True,
    },
    "L_turma_no_atgfm": {
        "desc": "TURMA-Net without Adaptive Tumor-Guided Feature Modulation",
        "backbone": "densenet121",
        "use_turma_net": True,
        "disable_atgfm": True,
    },
    "M_turma_no_consistency": {
        "desc": "TURMA-Net without cross-task consistency regularization",
        "backbone": "densenet121",
        "use_turma_net": True,
        "disable_consistency": True,
    },
    "N_turma_no_uncertainty": {
        "desc": "TURMA-Net without uncertainty-weighted loss (fixed weights)",
        "backbone": "densenet121",
        "use_turma_net": True,
        "disable_uncertainty": True,
    },
}


def build_model(exp_name: str, exp_cfg: dict, device: torch.device) -> torch.nn.Module:
    backbone = exp_cfg["backbone"]

    if exp_cfg.get("use_turma_net", False):
        from models.novel.turma_net import TurmaNet, TurmaNetConfig
        config = TurmaNetConfig(
            backbone=backbone,
            num_classes=4,
            segmentation_classes=4,
        )
        model = TurmaNet(config)

        if exp_cfg.get("disable_atgfm", False):
            model.atgfm = None
            model.adaptive_attention = None

        if exp_cfg.get("disable_consistency", False):
            model.consistency_loss = None

        if exp_cfg.get("disable_uncertainty", False):
            model.log_sigma_cls = None
            model.log_sigma_seg = None
            model.log_sigma_cons = None

        return model.to(device)

    if not exp_cfg["use_tumor_aware"]:
        if backbone == "cnn2d":
            from models.factory import ModelConfig, create_classification_model
            return create_classification_model(
                ModelConfig(
                    model_type="classification",
                    in_channels=3,
                    num_classes=4,
                    classification_input_size=(224, 224),
                    pretrained=True,
                )
            ).to(device)
        return create_backbone_classifier(backbone, num_classes=4, pretrained=True)

    config = TumorAwareConfig(backbone=backbone)
    model = TumorAwareMultiTaskNet(config)

    if exp_name == "G_no_fusion":
        model.fusion = lambda tumor_feat, cls_feat: cls_feat
    elif exp_name == "H_no_seg_supervision":
        model.seg_head = None
    elif exp_name == "I_no_tumor_features":
        model.tumor_projector = None
        model.cls_projector = None
        model.cross_attention = None

    return model.to(device)


def main():
    device = get_device()
    set_seed(42)

    classification_root = "data/raw/classification"
    batch_size = 8
    image_size = (224, 224)
    epochs = 3
    output_dir = Path("outputs/research_experiments/ablation_study")

    print("Loading classification data...")
    train_loader, val_loader, test_loader = load_classification_data(
        classification_root, image_size, batch_size
    )
    print(f"  Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    all_results = []

    for i, (exp_name, exp_cfg) in enumerate(EXPERIMENTS.items()):
        print(f"\n{'='*70}")
        print(f"  [{i+1}/{len(EXPERIMENTS)}] {exp_name}: {exp_cfg['desc']}")
        print(f"{'='*70}", flush=True)

        exp_dir = output_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        try:
            model = build_model(exp_name, exp_cfg, device)
            result = run_ablation_experiment(
                model=model,
                experiment_name=exp_name,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                num_classes=4,
                epochs=epochs,
                lr=0.0001,
                save_dir=exp_dir,
            )
            all_results.append(result)
            print(f"  Result: acc={result.accuracy:.4f}, f1={result.f1:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")

    if all_results:
        generate_ablation_table(all_results, output_dir)
        print(f"\nAblation table saved to {output_dir / 'ablation_table.md'}")
        print(f"Full results saved to {output_dir / 'ablation_results.json'}")

    print(f"\nAblation study complete. {len(all_results)}/{len(EXPERIMENTS)} experiments succeeded.")


if __name__ == "__main__":
    main()
