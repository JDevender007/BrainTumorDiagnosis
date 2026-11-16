"""
Train the BraTS 2024 segmentation baseline (3D U-Net).

Usage (from the project root):

    python -m scripts.train_segmentation --config configs/segmentation.yaml
    python -m scripts.train_segmentation --config configs/segmentation.yaml --epochs 10
    python -m scripts.train_segmentation --config configs/segmentation.yaml --smoke
"""

from __future__ import annotations

import argparse
import logging
import sys

from configs import load_merged_config
from training.segmentation_trainer import (
    SegmentationTrainer,
    SegmentationTrainerConfig,
)
from utils.logging import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the BraTS 2024 segmentation baseline."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/segmentation.yaml",
        help="Path to the YAML configuration file.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs.",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Override the checkpoint output directory.",
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny smoke training loop for verification.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint.",
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
        SegmentationTrainerConfig(),
        args.config,
        sections=("model", "dataset", "training", "paths"),
    )

    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir

    trainer = SegmentationTrainer(config)

    result = trainer.run(
        smoke=args.smoke,
        max_epochs=args.epochs,
        resume=args.resume,
    )

    logging.getLogger("scripts.train_segmentation").info(
        "Training finished. Best validation mean foreground dice: %.4f "
        "| epochs completed: %d | early stopped: %s",
        result["best_val_dice_mean"],
        result["epochs_completed"],
        result["early_stopped"],
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())