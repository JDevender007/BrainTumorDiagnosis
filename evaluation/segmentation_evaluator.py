"""
Sliding-window segmentation evaluation for BrainTumorDiagnosis.

Evaluates a trained 3D U-Net on full volumes using:

    - patch sliding window inference with softmax averaging
    - voxel Dice per BraTS 2024 sub-region:
          whole tumor (WT)  = labels {1, 2, 3}
          tumor core (TC)   = labels {1, 3}
          enhancing tumor (ET)     = label {3}
          resection cavity (RC)    = label {4}
    - per-case and aggregate reports saved as JSON

Run as a module from the project root:

    python -m evaluation.segmentation_evaluator
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from datasets.segmentation import (
    load_normalized_case,
    load_split_cases,
)
from datasets.brats import BraTSSample
from engine.checkpoint import CheckpointManager
from models.segmentation import create_segmentation_model
from utils.device import get_device
from utils.metrics import dice_score


LOGGER = logging.getLogger(__name__)

REGIONS = {
    "WT": (1, 2, 3),
    "TC": (1, 3),
    "ET": (3,),
    "RC": (4,),
}

CLASSIC_REGIONS = ("WT", "TC", "ET")

DEFAULT_CHECKPOINT_DIR = "checkpoints/segmentation/unet3d"


@dataclass
class SegmentationEvalConfig:
    """Configuration for segmentation evaluation."""

    in_channels: int = 4
    out_channels: int = 5
    base_channels: int = 16

    patch_size: tuple[int, int, int] = (96, 96, 96)
    overlap: float = 0.5

    manifest_path: str = "data/splits/brats/split_manifest.json"
    split_name: str = "test"
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR

    output_dir: str = "outputs/metrics"


def _window_starts(
    size: int,
    patch: int,
    stride: int,
) -> list[int]:
    """
    Return evenly spaced window start positions that cover the axis.
    """
    if size <= patch:
        return [0]

    last_start = size - patch

    count = int(np.ceil((last_start + stride - 1) / stride)) + 1

    starts = np.linspace(
        0,
        last_start,
        count,
    ).round().astype(int).tolist()

    return sorted(set(starts))


def sliding_window_starts(
    volume_shape: Sequence[int],
    patch_size: Sequence[int],
    overlap: float = 0.5,
) -> list[tuple[slice, slice, slice]]:
    """
    Build a covering grid of patch slices for a volume.
    """
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"overlap must be in [0, 1): {overlap}")

    per_axis: list[list[int]] = []

    for axis_size, patch in zip(
        volume_shape,
        patch_size,
    ):
        stride = max(1, int(patch * (1.0 - overlap)))

        per_axis.append(
            _window_starts(
                axis_size,
                patch,
                stride,
            )
        )

    windows: list[tuple[slice, slice, slice]] = []

    for depth in per_axis[0]:
        for height in per_axis[1]:
            for width in per_axis[2]:
                windows.append(
                    (
                        slice(depth, depth + patch_size[0]),
                        slice(height, height + patch_size[1]),
                        slice(width, width + patch_size[2]),
                    )
                )

    return windows


@torch.no_grad()
def predict_full_volume(
    model: torch.nn.Module,
    case: BraTSSample,
    device: torch.device,
    patch_size: Sequence[int] = (96, 96, 96),
    overlap: float = 0.5,
) -> np.ndarray:
    """
    Predict full-volume labels for one case.

    Returns:
        Integer label volume with shape [D, H, W].
    """
    image, _ = load_normalized_case(case)

    image_tensor = torch.from_numpy(image)

    volume_shape = image.shape[1:]

    windows = sliding_window_starts(
        volume_shape,
        patch_size,
        overlap,
    )

    probability_sum: np.ndarray | None = None

    counts = np.zeros(
        volume_shape,
        dtype=np.float32,
    )

    for spatial_slices in windows:
        patch = image_tensor[
            (slice(None), *spatial_slices)
        ].unsqueeze(0).to(device)

        logits = model(patch)

        if probability_sum is None:
            num_classes = logits.shape[1]

            probability_sum = np.zeros(
                (num_classes, *volume_shape),
                dtype=np.float32,
            )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0].cpu().numpy()

        probability_sum[
            (slice(None), *spatial_slices)
        ] += probabilities

        counts[spatial_slices] += 1.0

    probability_average = probability_sum / np.maximum(
        counts[None],
        1.0,
    )

    return np.argmax(
        probability_average,
        axis=0,
    ).astype(np.int64)


def _region_masks(
    segmentation: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        name: np.isin(
            segmentation,
            labels,
        )
        for name, labels in REGIONS.items()
    }


@torch.no_grad()
def evaluate_segmentation(
    model: torch.nn.Module,
    cases: Sequence[BraTSSample],
    device: torch.device,
    patch_size: Sequence[int] = (96, 96, 96),
    overlap: float = 0.5,
) -> dict:
    """
    Evaluate a segmentation model over a list of cases.

    Returns:
        Dictionary with aggregate metrics and per-case entries.
    """
    model.eval()

    per_case: list[dict] = []

    region_accumulator = {
        region: [] for region in REGIONS
    }

    pixel_correct_total = 0
    pixel_total_total = 0

    for case in cases:
        _, segmentation = load_normalized_case(case)

        prediction = predict_full_volume(
            model,
            case,
            device,
            patch_size,
            overlap,
        )

        pixel_correct = int(np.sum(prediction == segmentation))
        pixel_total = segmentation.size
        pixel_correct_total += pixel_correct
        pixel_total_total += pixel_total

        ground_truth_masks = _region_masks(segmentation)

        region_dice: dict[str, float] = {}

        for region, labels in REGIONS.items():
            prediction_mask = np.isin(
                prediction,
                labels,
            )

            prediction_tensor = torch.from_numpy(
                prediction_mask.astype(np.float32)
            )

            target_tensor = torch.from_numpy(
                ground_truth_masks[region].astype(np.float32)
            )

            dice = float(
                dice_score(
                    prediction_tensor,
                    target_tensor,
                )
            )

            region_dice[region] = dice

            region_accumulator[region].append(dice)

        mean_classic = float(
            np.mean(
                [region_dice[region] for region in CLASSIC_REGIONS]
            )
        )

        per_case.append(
            {
                "case_id": case.case_id,
                "region_dice": region_dice,
                "mean_wt_tc_et_dice": mean_classic,
            }
        )

    aggregate: dict[str, float] = {}

    for region in REGIONS:
        values = region_accumulator[region]

        aggregate[f"{region}_dice_mean"] = float(np.mean(values))
        aggregate[f"{region}_dice_std"] = float(np.std(values))

    aggregate["mean_wt_tc_et_dice"] = float(
        np.mean(
            [
                aggregate["WT_dice_mean"],
                aggregate["TC_dice_mean"],
                aggregate["ET_dice_mean"],
            ]
        )
    )

    aggregate["pixel_accuracy"] = float(
        pixel_correct_total / pixel_total_total
    )

    report = {
        "num_cases": len(cases),
        "aggregate": aggregate,
        "per_case": per_case,
    }

    return report


def _load_model(
    config: SegmentationEvalConfig,
    device: torch.device,
    checkpoint_path: str | Path | None = None,
) -> torch.nn.Module:
    model = create_segmentation_model(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        base_channels=config.base_channels,
    ).to(device)

    manager = CheckpointManager(
        config.checkpoint_dir,
        experiment_name="unet3d_segmentation",
    )

    if checkpoint_path is not None and Path(checkpoint_path).is_file():
        manager.load(
            checkpoint_path,
            model=model,
            map_location=device,
        )
    elif manager.exists_best():
        manager.load_best(
            model=model,
            map_location=device,
        )
    else:
        raise FileNotFoundError(
            "No checkpoint found. Pass --checkpoint or run training first."
        )

    return model


def main(
    config: SegmentationEvalConfig | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict:
    if config is None:
        config = SegmentationEvalConfig()

    device = get_device()

    model = _load_model(
        config,
        device,
        checkpoint_path,
    )

    cases = load_split_cases(
        config.manifest_path,
        config.split_name,
    )

    LOGGER.info(
        "Evaluating %d cases on split '%s' | patch=%s | overlap=%.2f",
        len(cases),
        config.split_name,
        config.patch_size,
        config.overlap,
    )

    report = evaluate_segmentation(
        model,
        cases,
        device,
        patch_size=config.patch_size,
        overlap=config.overlap,
    )

    output_dir = Path(config.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / (
        f"segmentation_evaluation_{config.split_name}.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    LOGGER.info("Evaluation report saved: %s", output_path)

    for key, value in report["aggregate"].items():
        LOGGER.info("%-22s %s", key, f"{value:.4f}")

    return report


def _test() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    import tempfile

    from datasets.segmentation import create_synthetic_brats_cases

    temp_dir = tempfile.TemporaryDirectory()

    cases = create_synthetic_brats_cases(
        Path(temp_dir.name),
        num_cases=2,
        volume_shape=(32, 36, 40),
    )

    model = create_segmentation_model(
        in_channels=4,
        out_channels=5,
        base_channels=2,
    )

    device = torch.device("cpu")

    model.eval()

    prediction = predict_full_volume(
        model,
        cases[0],
        device,
        patch_size=(16, 16, 16),
        overlap=0.5,
    )

    _, segmentation = load_normalized_case(cases[0])

    assert prediction.shape == segmentation.shape

    LOGGER.info(
        "Full-volume prediction shape: %s",
        tuple(prediction.shape),
    )

    assert set(np.unique(prediction)).issubset({0, 1, 2, 3, 4})

    windows = sliding_window_starts(
        (32, 36, 40),
        (16, 16, 16),
        0.5,
    )

    LOGGER.info(
        "Sliding windows for (32, 36, 40): %d",
        len(windows),
    )

    report = evaluate_segmentation(
        model,
        cases,
        device,
        patch_size=(16, 16, 16),
        overlap=0.5,
    )

    LOGGER.info("Aggregate metrics: %s", report["aggregate"])

    assert report["num_cases"] == 2
    assert len(report["per_case"]) == 2

    for region in REGIONS:
        mean = report["aggregate"][f"{region}_dice_mean"]
        assert 0.0 <= mean <= 1.0

    temp_dir.cleanup()

    LOGGER.info("Segmentation evaluator test passed.")


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        _test()
    else:
        main()