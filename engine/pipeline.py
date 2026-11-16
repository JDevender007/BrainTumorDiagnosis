from __future__ import annotations

import inspect
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from engine.inference import (
    InferenceConfig,
    InferenceEngine,
    InferenceResult,
    MRIInput,
    load_mri_modalities,
)
from engine.report import (
    ReportData,
    ReportGenerator,
    ReportGeneratorConfig,
    create_report_generator,
)
from engine.visualization import (
    VisualizationConfig,
    VisualizationResult,
    MRIVisualizer,
    create_visualizer,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Logging
# =============================================================================


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PipelineConfig:
    result_directory: str | Path = "outputs/results"
    report_directory: str | Path = "outputs/reports"
    visualization_directory: str | Path = "outputs/visualizations"

    model_directory: str | Path = "checkpoints"

    device: str | None = None
    use_amp: bool = True

    input_channels: int = 4
    classification_classes: int = 4
    segmentation_classes: int = 5

    confidence_threshold: float = 0.50

    enable_classification: bool = True
    enable_segmentation: bool = True

    modalities: tuple[str, ...] = (
        "t1n",
        "t1c",
        "t2w",
        "t2f",
    )

    save_visualizations: bool = True
    save_reports: bool = True

    case_id: str = "PIPELINE-TEST-001"

    def __post_init__(self) -> None:
        self.result_directory = Path(self.result_directory)
        self.report_directory = Path(self.report_directory)
        self.visualization_directory = Path(self.visualization_directory)
        self.model_directory = Path(self.model_directory)

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.input_channels <= 0:
            raise ValueError("input_channels must be greater than zero")

        if self.classification_classes <= 0:
            raise ValueError("classification_classes must be greater than zero")

        if self.segmentation_classes <= 0:
            raise ValueError("segmentation_classes must be greater than zero")

        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")

        self.result_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.report_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.visualization_directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# =============================================================================
# Pipeline Result
# =============================================================================


@dataclass
class PipelineResult:
    case_id: str
    success: bool

    inference_result: InferenceResult | None = None

    report_text: str | None = None
    report_json: str | None = None

    visualization_result: VisualizationResult | None = None

    elapsed_time: float = 0.0

    error: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "case_id": self.case_id,
            "success": self.success,
            "elapsed_time": self.elapsed_time,
            "error": self.error,
            "metadata": self.metadata,
        }

        if self.inference_result is not None:
            if hasattr(self.inference_result, "to_dict"):
                result["inference_result"] = self.inference_result.to_dict()
            else:
                result["inference_result"] = str(self.inference_result)

        if self.visualization_result is not None:
            if hasattr(self.visualization_result, "to_dict"):
                result["visualization_result"] = self.visualization_result.to_dict()
            else:
                result["visualization_result"] = str(self.visualization_result)

        result["report_text"] = self.report_text
        result["report_json"] = self.report_json

        return result


# =============================================================================
# Pipeline
# =============================================================================


class BrainTumorPipeline:
    """
    End-to-end BrainTumorDiagnosis pipeline.

    Flow:

        MRI input
             |
             v
        InferenceEngine
             |
             +----------------+
             |                |
             v                v
       Classification    Segmentation
             |                |
             +--------+-------+
                      |
                      v
                Visualization
                      |
                      v
                   Report
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
    ) -> None:

        self.config = config or PipelineConfig()

        logger.info("Initializing BrainTumorDiagnosis pipeline.")

        self.inference_engine = self._create_inference_engine()

        self.visualizer = self._create_visualizer()

        self.report_generator = self._create_report_generator()

        logger.info("Pipeline initialized.")
        logger.info("Device: %s", self.config.device)
        logger.info("AMP: %s", self.config.use_amp)
        logger.info(
            "Model directory: %s",
            self.config.model_directory,
        )
        logger.info(
            "Classification enabled: %s",
            self.config.enable_classification,
        )
        logger.info(
            "Segmentation enabled: %s",
            self.config.enable_segmentation,
        )

    # =========================================================================
    # Inference configuration
    # =========================================================================

    def _create_inference_engine(self) -> InferenceEngine:
        """
        Create InferenceEngine while matching the existing InferenceConfig API.

        This avoids passing unsupported arguments such as model_directory.
        """

        signature = inspect.signature(InferenceConfig)

        parameters = signature.parameters

        config_values: dict[str, Any] = {}

        candidates: dict[str, Any] = {
            "device": self.config.device,
            "use_amp": self.config.use_amp,
            "amp": self.config.use_amp,
            "input_channels": self.config.input_channels,
            "in_channels": self.config.input_channels,
            "classification_classes": (self.config.classification_classes),
            "num_classes": self.config.classification_classes,
            "segmentation_classes": (self.config.segmentation_classes),
            "confidence_threshold": (self.config.confidence_threshold),
            "enable_classification": (self.config.enable_classification),
            "enable_segmentation": (self.config.enable_segmentation),
            "modalities": self.config.modalities,
            "result_directory": self.config.result_directory,
            "output_directory": self.config.result_directory,
        }

        for name, value in candidates.items():
            if name in parameters:
                config_values[name] = value

        inference_config = InferenceConfig(**config_values)

        logger.info(
            "Inference configuration arguments: %s",
            sorted(config_values.keys()),
        )

        return InferenceEngine(inference_config)

    # =========================================================================
    # Visualization
    # =========================================================================

    def _create_visualizer(self) -> MRIVisualizer:
        signature = inspect.signature(VisualizationConfig)

        parameters = signature.parameters

        candidates: dict[str, Any] = {
            "output_directory": self.config.visualization_directory,
            "result_directory": self.config.visualization_directory,
        }

        values: dict[str, Any] = {}

        for name, value in candidates.items():
            if name in parameters:
                values[name] = value

        visualization_config = VisualizationConfig(**values)

        return create_visualizer(visualization_config)

    # =========================================================================
    # Report generator
    # =========================================================================

    def _create_report_generator(
        self,
    ) -> ReportGenerator:

        signature = inspect.signature(ReportGeneratorConfig)

        parameters = signature.parameters

        candidates: dict[str, Any] = {
            "output_directory": self.config.report_directory,
            "result_directory": self.config.report_directory,
        }

        values: dict[str, Any] = {}

        for name, value in candidates.items():
            if name in parameters:
                values[name] = value

        report_config = ReportGeneratorConfig(**values)

        return create_report_generator(report_config)

    # =========================================================================
    # Input validation
    # =========================================================================

    def validate_mri(
        self,
        mri_volume: Any,
    ) -> np.ndarray:

        if isinstance(mri_volume, torch.Tensor):
            array = mri_volume.detach().cpu().numpy()

        elif isinstance(mri_volume, np.ndarray):
            array = mri_volume

        else:
            array = np.asarray(mri_volume)

        if array.ndim == 3:
            if self.config.input_channels != 1:
                raise ValueError("3D MRI input requires input_channels=1.")

        elif array.ndim == 4:
            if array.shape[0] != self.config.input_channels:
                raise ValueError(
                    "Expected "
                    f"{self.config.input_channels} MRI channels, "
                    f"received {array.shape[0]}."
                )

        elif array.ndim == 5:
            if array.shape[1] != self.config.input_channels:
                raise ValueError(
                    "Expected "
                    f"{self.config.input_channels} MRI channels, "
                    f"received {array.shape[1]}."
                )

        else:
            raise ValueError("MRI input must be 3D, 4D, or 5D.")

        return array.astype(
            np.float32,
            copy=False,
        )

    # =========================================================================
    # Inference
    # =========================================================================

    def run_inference(
        self,
        mri_input: str | Path | np.ndarray | MRIInput,
        case_id: str | None = None,
    ) -> InferenceResult:

        case_id = case_id or self.config.case_id

        if isinstance(mri_input, MRIInput):
            mri_volume = None
        elif isinstance(mri_input, (str, Path)):
            mri_volume = None
            mri_input = MRIInput.from_directory(
                mri_input,
                case_id=case_id,
            )
        else:
            mri_volume = self.validate_mri(mri_input)

        logger.info(
            "Running inference for case: %s",
            case_id,
        )

        if mri_volume is not None:
            logger.info(
                "MRI shape: %s",
                tuple(mri_volume.shape),
            )

        if self.inference_engine.segmentation_model is None:
            self.inference_engine.load_models()

        return self.inference_engine.predict(
            mri_input,
            case_id=case_id,
        )

    # =========================================================================
    # Visualization
    # =========================================================================

    def create_visualizations(
        self,
        mri_volume: Any,
        inference_result: InferenceResult,
        case_id: str,
    ) -> VisualizationResult | None:

        if not self.config.save_visualizations:
            return None

        volume = self.validate_mri(mri_volume)

        logger.info(
            "Creating visualizations for case: %s",
            case_id,
        )

        return self.visualizer.generate(
            mri_volume=volume,
            inference_result=inference_result,
            case_id=case_id,
        )

    # =========================================================================
    # Reporting
    # =========================================================================

    def create_report(
        self,
        inference_result: InferenceResult,
        case_id: str,
        visualization_result: VisualizationResult | None = None,
    ) -> tuple[str | None, str | None]:

        if not self.config.save_reports:
            return None, None

        logger.info(
            "Generating report for case: %s",
            case_id,
        )

        report_data = self._build_report_data(
            inference_result=inference_result,
            case_id=case_id,
            visualization_result=visualization_result,
        )

        text_path = self.report_generator.save_text_report(
            report_data,
            filename=f"brain_tumor_report_{case_id}.txt",
        )

        json_path = self.report_generator.save_json_report(
            report_data,
            filename=f"brain_tumor_report_{case_id}.json",
        )

        return str(text_path), str(json_path)

    # =========================================================================
    # Report data
    # =========================================================================

    def _build_report_data(
        self,
        inference_result: InferenceResult,
        case_id: str,
        visualization_result: VisualizationResult | None,
    ) -> ReportData:

        result_dict: dict[str, Any] = {}

        if hasattr(
            inference_result,
            "to_dict",
        ):
            result_dict = inference_result.to_dict()

        elif hasattr(
            inference_result,
            "__dict__",
        ):
            result_dict = dict(inference_result.__dict__)

        segmentation_metrics = {
            key: result_dict.get(key)
            for key in (
                "tumor_voxel_count",
                "tumor_percentage",
                "tumor_detected",
            )
            if result_dict.get(key) is not None
        }

        segmentation_metrics["segmentation_shape"] = list(
            result_dict.get(
                "segmentation_shape",
                (),
            )
        )

        values = {
            "case_id": case_id,
            "prediction": result_dict.get(
                "tumor_class",
                result_dict.get(
                    "predicted_class",
                    "Unknown",
                ),
            ),
            "confidence": result_dict.get(
                "classification_confidence",
                result_dict.get(
                    "confidence",
                    0.0,
                ),
            ),
            "classification_available": bool(
                result_dict.get(
                    "classification_available",
                    True,
                )
            ),
            "segmentation_available": bool(
                result_dict.get(
                    "segmentation_available",
                    True,
                )
            ),
            "probabilities": result_dict.get(
                "classification_probabilities",
                result_dict.get(
                    "probabilities",
                    {},
                ),
            )
            or {},
            "segmentation_metrics": segmentation_metrics,
            "metadata": {
                "device": self.config.device,
                "modalities": list(self.config.modalities),
                "model_directory": str(self.config.model_directory),
                "segmentation_output": result_dict.get(
                    "segmentation_output_path",
                ),
            },
        }

        signature = inspect.signature(ReportData)

        parameters = signature.parameters

        filtered: dict[str, Any] = {}

        for name in parameters:

            if name in values:
                filtered[name] = values[name]

        return ReportData(**filtered)

    # =========================================================================
    # Path helper
    # =========================================================================

    @staticmethod
    def _extract_path(
        data: dict[str, Any],
        keys: list[str],
    ) -> str | None:

        for key in keys:

            value = data.get(key)

            if value is not None:
                return str(value)

        return None

    # =========================================================================
    # Complete pipeline
    # =========================================================================

    def process(
        self,
        mri_volume: Any,
        case_id: str | None = None,
    ) -> PipelineResult:

        case_id = case_id or self.config.case_id

        start_time = time.perf_counter()

        logger.info("=" * 78)
        logger.info("BrainTumorDiagnosis Pipeline")
        logger.info("=" * 78)
        logger.info(
            "Case ID: %s",
            case_id,
        )

        try:

            inference_result = self.run_inference(
                mri_input=mri_volume,
                case_id=case_id,
            )

            viz_volume = mri_volume

            if isinstance(viz_volume, (str, Path, MRIInput)):
                if isinstance(viz_volume, MRIInput):
                    mri_input_paths = viz_volume
                else:
                    mri_input_paths = MRIInput.from_directory(
                        viz_volume,
                        case_id=case_id,
                    )

                viz_volume = load_mri_modalities(mri_input_paths)

            visualization_result = self.create_visualizations(
                mri_volume=viz_volume,
                inference_result=inference_result,
                case_id=case_id,
            )

            report_text, report_json = self.create_report(
                inference_result=inference_result,
                case_id=case_id,
                visualization_result=(visualization_result),
            )

            elapsed = time.perf_counter() - start_time

            result = PipelineResult(
                case_id=case_id,
                success=True,
                inference_result=inference_result,
                report_text=report_text,
                report_json=report_json,
                visualization_result=(visualization_result),
                elapsed_time=elapsed,
                metadata={
                    "device": self.config.device,
                    "gpu_available": (torch.cuda.is_available()),
                    "gpu": (
                        torch.cuda.get_device_name(0)
                        if torch.cuda.is_available()
                        else None
                    ),
                    "input_channels": (self.config.input_channels),
                    "classification_classes": (self.config.classification_classes),
                    "segmentation_classes": (self.config.segmentation_classes),
                },
            )

            logger.info("Pipeline completed successfully.")

            logger.info(
                "Elapsed time: %.2f seconds",
                elapsed,
            )

            return result

        except Exception as exc:

            elapsed = time.perf_counter() - start_time

            logger.exception("Pipeline failed.")

            return PipelineResult(
                case_id=case_id,
                success=False,
                elapsed_time=elapsed,
                error=str(exc),
            )

    # =========================================================================
    # Test pipeline
    # =========================================================================

    def test(
        self,
        shape: tuple[int, int, int] = (
            32,
            32,
            32,
        ),
    ) -> PipelineResult:

        logger.info("Creating synthetic MRI test volume.")

        test_volume = np.random.randn(
            self.config.input_channels,
            *shape,
        ).astype(np.float32)

        return self.process(
            mri_volume=test_volume,
            case_id=self.config.case_id,
        )


# =============================================================================
# Factory
# =============================================================================


def create_pipeline(
    config: PipelineConfig | None = None,
) -> BrainTumorPipeline:

    return BrainTumorPipeline(config=config)


# =============================================================================
# Test
# =============================================================================


def main() -> None:

    configure_logging()

    logger.info("=" * 78)
    logger.info("BrainTumorDiagnosis Pipeline Test")
    logger.info("=" * 78)

    config = PipelineConfig(
        result_directory=("outputs/pipeline_test/results"),
        report_directory=("outputs/pipeline_test/reports"),
        visualization_directory=("outputs/pipeline_test/visualizations"),
        model_directory="checkpoints",
        device=("cuda" if torch.cuda.is_available() else "cpu"),
        use_amp=True,
        input_channels=4,
        classification_classes=4,
        segmentation_classes=4,
        confidence_threshold=0.50,
        enable_classification=True,
        enable_segmentation=True,
        case_id="PIPELINE-TEST-001",
    )

    logger.info(
        "Device: %s",
        config.device,
    )

    logger.info(
        "GPU available: %s",
        torch.cuda.is_available(),
    )

    if torch.cuda.is_available():
        logger.info(
            "GPU: %s",
            torch.cuda.get_device_name(0),
        )

    logger.info(
        "Model directory: %s",
        config.model_directory,
    )

    logger.info(
        "Classification classes: %d",
        config.classification_classes,
    )

    logger.info(
        "Segmentation classes: %d",
        config.segmentation_classes,
    )

    pipeline = create_pipeline(config)

    logger.info("Pipeline object created successfully.")

    logger.info("=" * 78)

    logger.info("Pipeline initialization test completed.")

    logger.info(
        "Run a real MRI case through pipeline.process() "
        "after model checkpoints are available."
    )

    logger.info("=" * 78)


if __name__ == "__main__":
    main()
