"""
BrainTumorDiagnosis
Single-case inference command line tool.

Usage (from the project root):

    python -m inference.predict data/raw/BraTS2024/BraTS-GLI-02062-100

    python -m inference.predict <case_dir> --case-id BraTS-GLI-02062-100
    python -m inference.predict <case_dir> --device cpu
    python -m inference.predict <case_dir> --no-save
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from engine.inference import InferenceConfig, InferenceEngine, MRIInput
from utils.logging import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run brain tumor inference on one MRI case.",
    )

    parser.add_argument(
        "case_directory",
        type=str,
        help="Directory containing the four MRI modality files.",
    )

    parser.add_argument(
        "--case-id",
        type=str,
        default=None,
        help="Optional case identifier (defaults to directory name).",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device: auto, cuda, cpu (default: auto).",
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write result JSON or segmentation NIfTI.",
    )

    parser.add_argument(
        "--result-directory",
        type=str,
        default="outputs/predictions",
        help="Directory for prediction outputs (default: outputs/predictions).",
    )

    return parser


def main() -> int:
    setup_logging(
        level=logging.INFO,
        log_file=None,
    )

    parser = build_parser()

    args = parser.parse_args()

    config = InferenceConfig(
        device=args.device,
        result_directory=args.result_directory,
    )

    engine = InferenceEngine(config)

    engine.load_models()

    mri_input = MRIInput.from_directory(
        args.case_directory,
        case_id=args.case_id,
    )

    result = engine.predict(
        mri_input,
        case_id=args.case_id,
        save_outputs=(not args.no_save),
    )

    summary = {
        "case_id": result.case_id,
        "tumor_detected": result.tumor_detected,
        "tumor_class": result.tumor_class,
        "tumor_class_index": result.tumor_class_index,
        "classification_confidence": round(result.classification_confidence, 4),
        "tumor_voxel_count": result.tumor_voxel_count,
        "tumor_percentage": round(result.tumor_percentage, 4),
        "classification_probabilities": (
            {
                name: round(value, 4)
                for name, value in result.classification_probabilities.items()
            }
            if result.classification_probabilities is not None
            else None
        ),
        "segmentation_output": result.segmentation_output_path,
        "inference_time_seconds": round(result.inference_time_seconds, 3),
    }

    print("=" * 78)
    print("BRAIN TUMOR DIAGNOSIS - INFERENCE RESULT")
    print("=" * 78)

    print(json.dumps(summary, indent=4))

    print("=" * 78)

    return 0


if __name__ == "__main__":
    sys.exit(main())