"""
Train a brain tumor classification baseline with the unified trainer.

Supported backbones:
    cnn2d, densenet121, resnet50, efficientnet_v2_s,
    convnext_tiny, swin_t, vit_b_16

Protocol:
    - validation split held out from the official training data
    - official Test split is never touched during training

Usage (from the project root):

    python -m scripts.train_classification --config configs/classification.yaml
    python -m scripts.train_classification --config configs/classification.yaml --backbone densenet121
    python -m scripts.train_classification --config configs/classification.yaml --epochs 10
    python -m scripts.train_classification --config configs/classification.yaml --smoke
"""

from __future__ import annotations

import argparse
import logging
import sys

from configs import load_merged_config
from training.classification_trainer import (
    SUPPORTED_BACKBONES,
    ClassificationTrainingConfig,
    train_classification_model,
)
from utils.logging import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a brain tumor classification baseline."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/classification.yaml",
        help="Path to the YAML configuration file.",
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        choices=SUPPORTED_BACKBONES,
        help="Override the model backbone.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs.",
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny smoke training loop for verification.",
    )

    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default=None,
        help="Initialize the backbone from a self-supervised checkpoint.",
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional path for a persistent log file.",
    )

    return parser


def main() -> int:
    parser = build_parser()

    args = parser.parse_args()

    setup_logging(
        level=logging.INFO,
        log_file=args.log_file,
    )

    config = load_merged_config(
        ClassificationTrainingConfig(),
        args.config,
        sections=("model", "data", "training", "paths"),
    )

    if args.backbone is not None:
        config.backbone = args.backbone

    if args.epochs is not None:
        config.epochs = args.epochs

    if args.smoke:
        config.epochs = min(config.epochs, 1)
        config.batch_size = 8

        logging.getLogger("scripts.train_classification").info(
            "Smoke mode: 1 epoch, batch size %d.",
            config.batch_size,
        )

    if args.pretrained_checkpoint is not None:
        config.pretrained_checkpoint = args.pretrained_checkpoint

    result = train_classification_model(config)

    logging.getLogger("scripts.train_classification").info(
        "Training finished. Best validation accuracy: %.2f%% "
        "| best epoch: %d | checkpoint: %s",
        result.best_validation_accuracy * 100,
        result.best_epoch,
        result.checkpoint_path,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())