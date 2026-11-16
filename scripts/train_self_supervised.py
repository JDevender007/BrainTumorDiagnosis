"""
Self-supervised contrastive pretraining (SimCLR) for MRI slice encoders.

The pretrained encoder can initialize supervised classification via:

    python -m scripts.train_classification --config configs/vit_b_16.yaml \
        --pretrained-checkpoint checkpoints/self_supervised/vit_b_16/best.pt

Usage (from the project root):

    python -m scripts.train_self_supervised --config configs/self_supervised.yaml
    python -m scripts.train_self_supervised --backbone densenet121 --epochs 10 --smoke
"""

from __future__ import annotations

import argparse
import logging
import sys

from configs import load_merged_config
from training.self_supervised_trainer import (
    SUPPORTED_BACKBONES,
    SelfSupervisedConfig,
    train_self_supervised,
)
from utils.logging import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Self-supervised SimCLR pretraining."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/self_supervised.yaml",
        help="Path to the YAML configuration file.",
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        choices=SUPPORTED_BACKBONES,
        help="Override the encoder backbone.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of pretraining epochs.",
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny smoke pretraining loop for verification.",
    )

    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from.",
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
        SelfSupervisedConfig(),
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

        logging.getLogger("scripts.train_self_supervised").info(
            "Smoke mode: 1 epoch, batch size %d.",
            config.batch_size,
        )

    if args.resume_from is not None:
        config.resume_from = args.resume_from

    result = train_self_supervised(config)

    logging.getLogger("scripts.train_self_supervised").info(
        "Pretraining finished. Best validation loss: %.4f "
        "| best epoch: %d | checkpoint: %s",
        result.best_loss,
        result.best_epoch,
        result.checkpoint_path,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())