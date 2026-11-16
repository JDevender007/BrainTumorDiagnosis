"""
BrainTumorDiagnosis
Inference Pipeline Facade.

Runs the complete workflow for one BraTS MRI case:

    MRI loading
        -> normalization
        -> tumor segmentation (trained 3D U-Net)
        -> tumor localization
        -> tumor classification (trained DenseNet121)
        -> result JSON + segmentation NIfTI
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from engine.inference import (
    InferenceConfig,
    InferenceEngine,
    InferenceResult,
    MRIInput,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class InferencePipelineConfig:
    """Configuration for the inference pipeline facade."""

    inference: InferenceConfig = field(default_factory=InferenceConfig)

    result_directory: str = "outputs/predictions"


class InferencePipeline:
    """
    High-level pipeline for single-case and batch inference.
    """

    def __init__(
        self,
        config: InferencePipelineConfig | None = None,
        engine: InferenceEngine | None = None,
    ) -> None:
        self.config = config or InferencePipelineConfig()

        self.engine = engine or InferenceEngine(self.config.inference)

    def load_models(self) -> None:
        """Load the trained segmentation and classification models."""

        self.engine.load_models()

    def predict_case(
        self,
        case_directory: str | Path,
        case_id: str | None = None,
        save_outputs: bool | None = None,
    ) -> InferenceResult:
        """
        Run complete inference on one case directory.

        The directory must contain the four modality files:
            {case_id}-t1n.nii, {case_id}-t1c.nii,
            {case_id}-t2w.nii, {case_id}-t2f.nii
        """

        mri_input = MRIInput.from_directory(
            case_directory,
            case_id=case_id,
        )

        if case_id is None:
            case_id = Path(case_directory).name

        result = self.engine.predict(
            mri_input,
            case_id=case_id,
            save_outputs=save_outputs,
        )

        LOGGER.info(
            "Case %s: tumor_detected=%s class=%s confidence=%.4f "
            "tumor_voxels=%d",
            case_id,
            result.tumor_detected,
            result.tumor_class,
            result.classification_confidence,
            result.tumor_voxel_count,
        )

        return result

    def predict_batch(
        self,
        case_directories: list[str | Path],
        case_ids: list[str] | None = None,
    ) -> list[InferenceResult]:
        """
        Run inference over several case directories.

        When ``case_ids`` is provided it must match the number of
        directories; otherwise each directory name is used.
        """

        results: list[InferenceResult] = []

        for index, case_directory in enumerate(case_directories):
            case_id = case_ids[index] if case_ids is not None else None

            results.append(
                self.predict_case(
                    case_directory,
                    case_id=case_id,
                )
            )

        return results

    def save_results_json(
        self,
        results: list[InferenceResult],
        output_path: str | Path,
    ) -> Path:
        """Save a list of results to a JSON file."""

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "results": [result.to_dict() for result in results],
        }

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                payload,
                file,
                indent=4,
            )

        LOGGER.info(
            "Batch results saved: %s",
            output_path,
        )

        return output_path


def run_case_pipeline(
    case_directory: str | Path,
    config: InferencePipelineConfig | None = None,
    case_id: str | None = None,
    save_outputs: bool | None = None,
) -> InferenceResult:
    """
    Convenience function: create a pipeline, load the models and
    run inference for one case.
    """

    pipeline = InferencePipeline(config=config)

    pipeline.load_models()

    return pipeline.predict_case(
        case_directory,
        case_id=case_id,
        save_outputs=save_outputs,
    )


def _test() -> None:
    print("=" * 78)
    print("BrainTumorDiagnosis Inference Pipeline Test")
    print("=" * 78)

    config = InferencePipelineConfig()

    pipeline = InferencePipeline(config=config)

    print(f"Engine device        : {pipeline.engine.device}")
    print(f"Result directory     : {config.result_directory}")
    print(
        f"Segmentation classes : {config.inference.segmentation_classes}"
    )
    print(
        f"Classification model : {config.inference.classification_backbone}"
    )
    print(
        "Configuration test passed.",
    )
    print("=" * 78)


if __name__ == "__main__":
    _test()