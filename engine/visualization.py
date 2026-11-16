"""
BrainTumorDiagnosis MRI Visualization Engine.

Generates PNG visualizations for an analyzed BraTS case:

    - Per-modality axial slice montage
    - Tumor segmentation overlay
    - Prediction summary card

Expected MRI volume layouts (channels first):

    (C, H, W, D)
    (1, C, H, W, D)

The channel order is the BraTS modality order defined in the
configuration (default: t1n, t1c, t2w, t2f).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


LOGGER = logging.getLogger(__name__)

DEFAULT_MODALITIES: tuple[str, ...] = (
    "t1n",
    "t1c",
    "t2w",
    "t2f",
)


@dataclass
class VisualizationConfig:
    """Configuration for the MRI visualization engine."""

    output_directory: str | Path = "outputs/visualizations"
    modalities: tuple[str, ...] = DEFAULT_MODALITIES
    figure_dpi: int = 120

    def __post_init__(self) -> None:
        self.output_directory = Path(self.output_directory)

        if not self.modalities:
            raise ValueError("modalities must not be empty")

        if self.figure_dpi <= 0:
            raise ValueError("figure_dpi must be greater than zero")


@dataclass
class VisualizationResult:
    """Result of a visualization run."""

    case_id: str
    mri_slice_path: str | None = None
    multimodal_path: str | None = None
    segmentation_path: str | None = None
    overlay_path: str | None = None
    prediction_summary_path: str | None = None
    saved_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""
        return asdict(self)


class MRIVisualizer:
    """Create MRI visualization artifacts for one case."""

    def __init__(
        self,
        config: VisualizationConfig | None = None,
    ) -> None:
        self.config = config if config is not None else VisualizationConfig()

        self.config.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        LOGGER.info(
            "Visualizer initialized. Output: %s",
            self.config.output_directory,
        )

    # ------------------------------------------------------------------
    # Input normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _as_3d_channeled(
        mri_volume: Any,
        expected_channels: int,
    ) -> np.ndarray:
        """
        Convert the input MRI volume into (C, H, W, D) float32.

        Raises:
            ValueError: when the input shape cannot be interpreted.
        """
        if isinstance(mri_volume, torch.Tensor):
            array = mri_volume.detach().cpu().numpy()
        else:
            array = np.asarray(mri_volume)

        if array.ndim == 5:
            if array.shape[0] != 1:
                raise ValueError(
                    f"5D MRI volume must be (1, C, H, W, D), "
                    f"received {tuple(array.shape)}"
                )
            array = array[0]

        if array.ndim != 4:
            raise ValueError(
                f"MRI volume must be (C, H, W, D), received {tuple(array.shape)}"
            )

        channels, depth, height, width = array.shape

        if channels != expected_channels:
            LOGGER.warning(
                "MRI volume has %d channels, expected %d. "
                "Visualization will map channels generically.",
                channels,
                expected_channels,
            )

        return np.asarray(
            array,
            dtype=np.float32,
        )

    @staticmethod
    def _normalize_slice(
        slice_2d: np.ndarray,
    ) -> np.ndarray:
        """
        Clip and rescale one axial slice to the [0, 1] range.
        """
        values = slice_2d[np.isfinite(slice_2d)]

        if values.size == 0:
            return np.zeros_like(
                slice_2d,
                dtype=np.float32,
            )

        low, high = np.percentile(
            values,
            [0.5, 99.5],
        )

        if high - low < 1e-8:
            return np.full_like(
                slice_2d,
                0.5,
                dtype=np.float32,
            )

        clipped = np.clip(
            slice_2d,
            low,
            high,
        )

        return (clipped - low) / (high - low)

    # ------------------------------------------------------------------
    # Slice selection
    # ------------------------------------------------------------------

    @staticmethod
    def _select_slice(
        volume_3d: np.ndarray,
        segmentation: np.ndarray | None,
    ) -> int:
        """
        Select the axial slice with the largest tumor area,
        or the middle slice when no tumor information exists.
        """
        if segmentation is not None:
            if segmentation.shape == volume_3d.shape[1:]:
                tumor_scores = segmentation.sum(
                    axis=(
                        0,
                        1,
                    )
                )

                if tumor_scores.max() > 0:
                    return int(np.argmax(tumor_scores))

        return volume_3d.shape[3] // 2

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    @staticmethod
    def _segmentation_from_result(
        inference_result: Any,
        volume_shape: tuple[int, ...],
    ) -> np.ndarray | None:
        """
        Extract a 3D integer segmentation mask from the inference result.
        """
        if inference_result is None:
            return None

        segmentation = getattr(
            inference_result,
            "segmentation",
            None,
        )

        if segmentation is None:
            return None

        if isinstance(segmentation, torch.Tensor):
            segmentation = segmentation.detach().cpu().numpy()

        segmentation = np.asarray(segmentation)

        if segmentation.ndim == 5 and segmentation.shape[0] == 1:
            segmentation = segmentation[0]

        if segmentation.ndim == 4 and segmentation.shape[0] == 1:
            segmentation = segmentation[0]

        if segmentation.ndim == 4:
            segmentation = np.argmax(
                segmentation,
                axis=0,
            )

        if segmentation.ndim != 3:
            LOGGER.warning(
                "Segmentation mask has unexpected shape: %s",
                tuple(segmentation.shape),
            )
            return None

        if segmentation.shape != volume_shape:
            LOGGER.warning(
                "Segmentation shape %s does not match volume %s. Skipping overlay.",
                tuple(segmentation.shape),
                volume_shape,
            )
            return None

        return segmentation.astype(
            np.int64,
            copy=False,
        )

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    def _render_multimodal_montage(
        self,
        volume_3d: np.ndarray,
        slice_index: int,
        case_id: str,
    ) -> Path:
        """
        Render one axial slice per modality.
        """
        channels = volume_3d.shape[0]

        columns = min(channels, 4)

        figure, axes = plt.subplots(
            1,
            columns,
            figsize=(4 * columns, 4),
        )

        if columns == 1:
            axes = [axes]

        for axis_index in range(columns):
            if axis_index < channels:
                slice_2d = volume_3d[
                    axis_index,
                    :,
                    :,
                    slice_index,
                ]
            else:
                slice_2d = np.zeros(
                    volume_3d.shape[1:3],
                    dtype=np.float32,
                )

            modality_name = (
                self.config.modalities[axis_index]
                if axis_index < len(self.config.modalities)
                else f"ch{axis_index}"
            )

            axes[axis_index].imshow(
                self._normalize_slice(slice_2d),
                cmap="gray",
                interpolation="nearest",
            )

            axes[axis_index].set_title(modality_name)

            axes[axis_index].axis("off")

        figure.suptitle(
            f"Case {case_id} - Axial Slice {slice_index}"
        )

        figure.tight_layout()

        output_path = (
            self.config.output_directory / f"{case_id}_multimodal.png"
        )

        figure.savefig(
            output_path,
            dpi=self.config.figure_dpi,
        )

        plt.close(figure)

        return output_path

    def _render_single_slice(
        self,
        volume_3d: np.ndarray,
        slice_index: int,
        case_id: str,
    ) -> Path:
        """
        Render the T1ce slice for quick inspection.
        """
        t1ce_index = 1 if volume_3d.shape[0] >= 2 else 0

        slice_2d = volume_3d[
            t1ce_index,
            :,
            :,
            slice_index,
        ]

        figure, axis = plt.subplots(
            1,
            1,
            figsize=(4, 4),
        )

        axis.imshow(
            self._normalize_slice(slice_2d),
            cmap="gray",
            interpolation="nearest",
        )

        modality_name = (
            self.config.modalities[t1ce_index]
            if t1ce_index < len(self.config.modalities)
            else f"ch{t1ce_index}"
        )

        axis.set_title(
            f"Case {case_id} - {modality_name} Slice {slice_index}"
        )

        axis.axis("off")

        figure.tight_layout()

        output_path = (
            self.config.output_directory / f"{case_id}_mri_slice.png"
        )

        figure.savefig(
            output_path,
            dpi=self.config.figure_dpi,
        )

        plt.close(figure)

        return output_path

    def _render_segmentation_overlay(
        self,
        volume_3d: np.ndarray,
        segmentation: np.ndarray,
        slice_index: int,
        case_id: str,
    ) -> Path:
        """
        Render the tumor mask over the reference modality slice.
        """
        t1ce_index = 1 if volume_3d.shape[0] >= 2 else 0

        modality_name = (
            self.config.modalities[t1ce_index]
            if t1ce_index < len(self.config.modalities)
            else f"ch{t1ce_index}"
        )

        background = self._normalize_slice(
            volume_3d[
                t1ce_index,
                :,
                :,
                slice_index,
            ]
        )

        mask = segmentation[
            :,
            :,
            slice_index,
        ]

        tumor_mask = mask > 0

        figure, axis = plt.subplots(
            1,
            1,
            figsize=(4, 4),
        )

        axis.imshow(
            background,
            cmap="gray",
            interpolation="nearest",
        )

        if tumor_mask.any():
            axis.imshow(
                np.ma.masked_where(~tumor_mask, tumor_mask),
                cmap="autumn",
                alpha=0.45,
                interpolation="nearest",
            )

        axis.set_title(
            f"Case {case_id} - {modality_name} + Tumor Overlay"
        )

        axis.axis("off")

        figure.tight_layout()

        output_path = (
            self.config.output_directory / f"{case_id}_overlay.png"
        )

        figure.savefig(
            output_path,
            dpi=self.config.figure_dpi,
        )

        plt.close(figure)

        return output_path

    def _render_prediction_summary(
        self,
        case_id: str,
        inference_result: Any,
    ) -> Path:
        """
        Render a text summary card of the prediction.
        """
        if inference_result is None:
            return self._render_plain_summary(case_id)

        prediction = getattr(
            inference_result,
            "tumor_class",
            getattr(
                inference_result,
                "prediction",
                "Unknown",
            ),
        )

        confidence = float(
            getattr(
                inference_result,
                "classification_confidence",
                getattr(
                    inference_result,
                    "confidence",
                    0.0,
                ),
            )
        )

        tumor_detected = bool(
            getattr(
                inference_result,
                "tumor_detected",
                getattr(
                    inference_result,
                    "tumor_present",
                    False,
                ),
            )
        )

        probabilities = getattr(
            inference_result,
            "classification_probabilities",
            None,
        )

        lines = [
            f"Case ID: {case_id}",
            f"Tumor detected: {tumor_detected}",
            f"Predicted class: {prediction}",
            f"Confidence: {confidence:.4f}",
        ]

        if probabilities:
            lines.append("")

            for class_name, probability in sorted(
                probabilities.items(),
                key=lambda item: float(item[1]),
                reverse=True,
            ):
                lines.append(
                    f"{str(class_name):<24} {float(probability):.4f}"
                )

        figure, axis = plt.subplots(
            1,
            1,
            figsize=(6, 3),
        )

        axis.axis("off")

        axis.text(
            0.02,
            0.5,
            "\n".join(lines),
            transform=axis.transAxes,
            fontsize=10,
            verticalalignment="center",
            family="monospace",
        )

        axis.set_title("Prediction Summary")

        figure.tight_layout()

        output_path = (
            self.config.output_directory / f"{case_id}_prediction_summary.png"
        )

        figure.savefig(
            output_path,
            dpi=self.config.figure_dpi,
        )

        plt.close(figure)

        return output_path

    def _render_plain_summary(
        self,
        case_id: str,
    ) -> Path:
        figure, axis = plt.subplots(
            1,
            1,
            figsize=(6, 3),
        )

        axis.axis("off")

        axis.text(
            0.02,
            0.5,
            f"Case ID: {case_id}\nNo inference result available.",
            transform=axis.transAxes,
            fontsize=10,
            verticalalignment="center",
            family="monospace",
        )

        axis.set_title("Prediction Summary")

        figure.tight_layout()

        output_path = (
            self.config.output_directory / f"{case_id}_prediction_summary.png"
        )

        figure.savefig(
            output_path,
            dpi=self.config.figure_dpi,
        )

        plt.close(figure)

        return output_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        mri_volume: Any,
        inference_result: Any = None,
        case_id: str = "case",
    ) -> VisualizationResult:
        """
        Generate all visualization artifacts for one case.
        """
        volume_3d = self._as_3d_channeled(
            mri_volume,
            expected_channels=len(self.config.modalities),
        )

        volume_shape = tuple(volume_3d.shape[1:])

        segmentation = self._segmentation_from_result(
            inference_result,
            volume_shape,
        )

        slice_index = self._select_slice(
            volume_3d,
            segmentation,
        )

        multimodal_path = self._render_multimodal_montage(
            volume_3d,
            slice_index,
            case_id,
        )

        mri_slice_path = self._render_single_slice(
            volume_3d,
            slice_index,
            case_id,
        )

        overlay_path: Path | None = None

        if segmentation is not None:
            overlay_path = self._render_segmentation_overlay(
                volume_3d,
                segmentation,
                slice_index,
                case_id,
            )

        prediction_summary_path = self._render_prediction_summary(
            case_id,
            inference_result,
        )

        saved_files = [
            str(multimodal_path),
            str(mri_slice_path),
            str(prediction_summary_path),
        ]

        if overlay_path is not None:
            saved_files.append(str(overlay_path))

        LOGGER.info(
            "Visualizations generated for case %s: %d files.",
            case_id,
            len(saved_files),
        )

        return VisualizationResult(
            case_id=case_id,
            mri_slice_path=str(mri_slice_path),
            multimodal_path=str(multimodal_path),
            segmentation_path=str(overlay_path) if overlay_path is not None else None,
            overlay_path=str(overlay_path) if overlay_path is not None else None,
            prediction_summary_path=str(prediction_summary_path),
            saved_files=saved_files,
        )

    def generate_all(
        self,
        mri_volume: Any,
        inference_result: Any = None,
        case_id: str = "case",
    ) -> VisualizationResult:
        """Alias for generate()."""
        return self.generate(
            mri_volume,
            inference_result=inference_result,
            case_id=case_id,
        )


def create_visualizer(
    config: VisualizationConfig | None = None,
) -> MRIVisualizer:
    """Create an MRI visualizer."""
    return MRIVisualizer(config=config)


def _test_visualizer() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=("%(asctime)s | " "%(levelname)s | " "%(name)s | " "%(message)s"),
    )

    LOGGER.info("=" * 70)
    LOGGER.info("BrainTumorDiagnosis Visualization Test")
    LOGGER.info("=" * 70)

    config = VisualizationConfig(
        output_directory="outputs/visualization_test",
    )

    visualizer = create_visualizer(config)

    volume = np.random.randn(
        4,
        64,
        64,
        32,
    ).astype(
        np.float32,
    )

    segmentation = np.zeros(
        (
            64,
            64,
            32,
        ),
        dtype=np.int64,
    )

    segmentation[
        20:40,
        20:40,
        10:20,
    ] = 1

    result = visualizer.generate(
        volume,
        inference_result=None,
        case_id="VIS-TEST-001",
    )

    assert result.multimodal_path is not None
    assert result.mri_slice_path is not None
    assert result.prediction_summary_path is not None

    assert Path(result.multimodal_path).exists()
    assert Path(result.mri_slice_path).exists()
    assert Path(result.prediction_summary_path).exists()

    overlay_result = visualizer.generate_all(
        volume,
        inference_result=SimpleNamespace(segmentation=segmentation),
        case_id="VIS-TEST-002",
    )

    assert overlay_result.overlay_path is not None
    assert Path(overlay_result.overlay_path).exists()

    LOGGER.info("Visualization test passed.")

    LOGGER.info("=" * 70)


class SimpleNamespace:
    """Minimal namespace used only by the module self-test."""

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


if __name__ == "__main__":
    _test_visualizer()