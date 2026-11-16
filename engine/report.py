from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import logging


logger = logging.getLogger(__name__)


@dataclass
class ReportGeneratorConfig:
    output_dir: str | Path = "outputs/reports"
    report_filename: str = "brain_tumor_report.txt"
    include_probabilities: bool = True
    include_segmentation: bool = True
    include_metadata: bool = True
    overwrite: bool = True

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)


@dataclass
class ReportData:
    case_id: str
    prediction: str
    confidence: float
    classification_available: bool = False
    segmentation_available: bool = False
    probabilities: Mapping[str, float] = field(default_factory=dict)
    segmentation_metrics: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ReportGenerator:
    def __init__(
        self,
        config: ReportGeneratorConfig | None = None,
    ) -> None:
        self.config = config or ReportGeneratorConfig()

        self.config.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        logger.info("Report generator initialized.")
        logger.info("Output directory: %s", self.config.output_dir)

    @staticmethod
    def _format_percentage(value: float) -> str:
        if value <= 1.0:
            value *= 100.0

        return f"{value:.2f}%"

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _normalize_data(
        self,
        data: ReportData | Mapping[str, Any],
    ) -> ReportData:
        if isinstance(data, ReportData):
            return data

        return ReportData(
            case_id=str(
                data.get(
                    "case_id",
                    data.get("caseId", "UNKNOWN"),
                )
            ),
            prediction=str(
                data.get(
                    "prediction",
                    data.get("predicted_class", "Unknown"),
                )
            ),
            confidence=self._safe_float(
                data.get(
                    "confidence",
                    data.get("prediction_confidence", 0.0),
                )
            ),
            classification_available=bool(
                data.get(
                    "classification_available",
                    data.get("classification", False),
                )
            ),
            segmentation_available=bool(
                data.get(
                    "segmentation_available",
                    data.get("segmentation", False),
                )
            ),
            probabilities=data.get(
                "probabilities",
                data.get("class_probabilities", {}),
            )
            or {},
            segmentation_metrics=data.get(
                "segmentation_metrics",
                {},
            )
            or {},
            metadata=data.get(
                "metadata",
                {},
            )
            or {},
        )

    def generate_text(
        self,
        data: ReportData | Mapping[str, Any],
    ) -> str:
        report = self._normalize_data(data)

        lines: list[str] = []

        lines.append("=" * 78)
        lines.append("BrainTumorDiagnosis Medical AI Analysis Report")
        lines.append("=" * 78)
        lines.append("")

        lines.append(f"Case ID       : {report.case_id}")
        lines.append(f"Generated     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        lines.append("Prediction")
        lines.append("-" * 78)
        lines.append(f"Prediction    : {report.prediction}")
        lines.append(f"Confidence    : {self._format_percentage(report.confidence)}")
        lines.append("")

        lines.append("Analysis Availability")
        lines.append("-" * 78)
        lines.append(
            "Classification: "
            + ("Available" if report.classification_available else "Unavailable")
        )
        lines.append(
            "Segmentation  : "
            + ("Available" if report.segmentation_available else "Unavailable")
        )
        lines.append("")

        if self.config.include_probabilities and report.probabilities:
            lines.append("Classification Probabilities")
            lines.append("-" * 78)

            sorted_probabilities = sorted(
                report.probabilities.items(),
                key=lambda item: self._safe_float(item[1]),
                reverse=True,
            )

            for class_name, probability in sorted_probabilities:
                lines.append(
                    f"{str(class_name):<30} "
                    f"{self._format_percentage(self._safe_float(probability))}"
                )

            lines.append("")

        if self.config.include_segmentation and report.segmentation_metrics:
            lines.append("Segmentation Metrics")
            lines.append("-" * 78)

            for metric_name, metric_value in report.segmentation_metrics.items():
                if isinstance(metric_value, (int, float)):
                    lines.append(
                        f"{str(metric_name):<30} "
                        f"{self._safe_float(metric_value):.4f}"
                    )
                else:
                    lines.append(f"{str(metric_name):<30} " f"{metric_value}")

            lines.append("")

        if self.config.include_metadata and report.metadata:
            lines.append("Analysis Metadata")
            lines.append("-" * 78)

            for key, value in report.metadata.items():
                lines.append(f"{str(key):<30} {value}")

            lines.append("")

        lines.append("=" * 78)
        lines.append("End of BrainTumorDiagnosis Analysis Report")
        lines.append("=" * 78)

        return "\n".join(lines)

    def save_text_report(
        self,
        data: ReportData | Mapping[str, Any],
        filename: str | None = None,
    ) -> Path:
        report_text = self.generate_text(data)

        output_filename = (
            filename if filename is not None else self.config.report_filename
        )

        output_path = self.config.output_dir / output_filename

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if output_path.exists() and not self.config.overwrite:
            raise FileExistsError(f"Report already exists: {output_path}")

        output_path.write_text(
            report_text,
            encoding="utf-8",
        )

        logger.info(
            "Text report saved: %s",
            output_path,
        )

        return output_path

    def generate_json(
        self,
        data: ReportData | Mapping[str, Any],
    ) -> dict[str, Any]:
        report = self._normalize_data(data)

        return {
            "case_id": report.case_id,
            "prediction": report.prediction,
            "confidence": report.confidence,
            "classification_available": (report.classification_available),
            "segmentation_available": (report.segmentation_available),
            "probabilities": dict(report.probabilities),
            "segmentation_metrics": dict(report.segmentation_metrics),
            "metadata": dict(report.metadata),
            "generated_at": datetime.now().isoformat(),
        }

    def save_json_report(
        self,
        data: ReportData | Mapping[str, Any],
        filename: str = "brain_tumor_report.json",
    ) -> Path:
        output_path = self.config.output_dir / filename

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if output_path.exists() and not self.config.overwrite:
            raise FileExistsError(f"Report already exists: {output_path}")

        payload = self.generate_json(data)

        output_path.write_text(
            json.dumps(
                payload,
                indent=4,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        logger.info(
            "JSON report saved: %s",
            output_path,
        )

        return output_path

    def generate_report(
        self,
        data: ReportData | Mapping[str, Any],
        save_json: bool = False,
    ) -> dict[str, Path | str]:
        text_path = self.save_text_report(data)

        result: dict[str, Path | str] = {
            "text": text_path,
        }

        if save_json:
            result["json"] = self.save_json_report(data)

        result["content"] = self.generate_text(data)

        return result


def create_report_generator(
    config: ReportGeneratorConfig | None = None,
) -> ReportGenerator:
    return ReportGenerator(config=config)


def _test_report_generator() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=("%(asctime)s | %(levelname)s | " "%(name)s | %(message)s"),
    )

    logger.info("=" * 78)
    logger.info("BrainTumorDiagnosis Report Generator Test")
    logger.info("=" * 78)

    config = ReportGeneratorConfig(
        output_dir="outputs/reports_test",
        report_filename="brain_tumor_report.txt",
    )

    generator = ReportGenerator(config)

    test_data = ReportData(
        case_id="TEST-BRATS-001",
        prediction="Glioma",
        confidence=0.87,
        classification_available=True,
        segmentation_available=True,
        probabilities={
            "Glioma": 0.87,
            "Meningioma": 0.06,
            "Pituitary": 0.04,
            "No Tumor": 0.03,
        },
        segmentation_metrics={
            "dice": 0.91,
            "iou": 0.86,
            "tumor_volume_voxels": 12450,
        },
        metadata={
            "model": "Hybrid Brain Tumor Model",
            "modalities": "t1n, t1c, t2w, t2f",
            "device": "cuda",
        },
    )

    text = generator.generate_text(test_data)

    print()
    print(text)
    print()

    text_path = generator.save_text_report(test_data)

    json_path = generator.save_json_report(test_data)

    logger.info(
        "Text report: %s",
        text_path,
    )

    logger.info(
        "JSON report: %s",
        json_path,
    )

    logger.info("=" * 78)
    logger.info("Report generator test completed successfully.")
    logger.info("=" * 78)


if __name__ == "__main__":
    _test_report_generator()
