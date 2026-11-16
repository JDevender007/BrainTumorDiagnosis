"""
Evaluate a trained segmentation model on a manifest split.

Usage (from the project root):

    python -m scripts.evaluate_segmentation
    python -m scripts.evaluate_segmentation --checkpoint checkpoints/segmentation/unet3d/best/unet3d_segmentation.pt
    python -m scripts.evaluate_segmentation --split validation
"""

from __future__ import annotations

import argparse
import logging
import sys

from configs import load_merged_config
from evaluation.segmentation_evaluator import (
    SegmentationEvalConfig,
    main as evaluate_main,
)
from utils.logging import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained segmentation model."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/segmentation.yaml",
        help="Path to the YAML configuration file.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Explicit checkpoint path. Defaults to the best checkpoint.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Manifest split to evaluate (train/validation/test).",
    )

    return parser


def main() -> int:
    setup_logging(
        level=logging.INFO,
    )

    parser = build_parser()

    args = parser.parse_args()

    config = load_merged_config(
        SegmentationEvalConfig(),
        args.config,
        sections=("model", "inference", "paths"),
    )

    config.split_name = args.split

    evaluate_main(
        config,
        checkpoint_path=args.checkpoint,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())