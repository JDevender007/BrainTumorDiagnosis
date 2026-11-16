"""
BrainTumorDiagnosis Streamlit User Interface.

Provides a web interface for:
- MRI modality upload
- Brain tumor classification
- Tumor segmentation
- MRI visualization
- Prediction reports
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image

try:
    import nibabel as nib
except ImportError:
    nib = None

from engine.pipeline import (
    BrainTumorPipeline,
    PipelineConfig,
)


LOGGER = logging.getLogger(__name__)


@dataclass
class UIConfig:
    """Configuration for the Streamlit interface."""

    title: str = "BrainTumorDiagnosis"
    page_title: str = "BrainTumorDiagnosis"
    page_icon: str = "🧠"

    model_directory: str | Path = "checkpoints"
    result_directory: str | Path = "outputs/results"
    report_directory: str | Path = "outputs/reports"
    visualization_directory: str | Path = "outputs/visualizations"

    input_channels: int = 4
    classification_classes: int = 4
    segmentation_classes: int = 5

    confidence_threshold: float = 0.50

    enable_classification: bool = True
    enable_segmentation: bool = True

    use_amp: bool = True

    modalities: tuple[str, ...] = (
        "t1n",
        "t1c",
        "t2w",
        "t2f",
    )


class BrainTumorUI:
    """Main Streamlit user interface."""

    def __init__(self, config: UIConfig | None = None) -> None:
        self.config = config or UIConfig()

        self.pipeline = BrainTumorPipeline(
            PipelineConfig(
                result_directory=self.config.result_directory,
                report_directory=self.config.report_directory,
                visualization_directory=self.config.visualization_directory,
                model_directory=self.config.model_directory,
                device=None,
                use_amp=self.config.use_amp,
                input_channels=self.config.input_channels,
                classification_classes=self.config.classification_classes,
                segmentation_classes=self.config.segmentation_classes,
                confidence_threshold=self.config.confidence_threshold,
                enable_classification=self.config.enable_classification,
                enable_segmentation=self.config.enable_segmentation,
                modalities=self.config.modalities,
                save_visualizations=True,
                save_reports=True,
            )
        )

    def configure_page(self) -> None:
        """Configure Streamlit page."""

        st.set_page_config(
            page_title=self.config.page_title,
            page_icon=self.config.page_icon,
            layout="wide",
            initial_sidebar_state="expanded",
        )

    def render_header(self) -> None:
        """Render application header."""

        st.title("🧠 BrainTumorDiagnosis")

        st.markdown(
            """
            ### MRI-Based Brain Tumor Analysis

            Upload the four BraTS MRI modalities to run classification,
            tumor segmentation, visualization, and report generation.
            """
        )

        st.divider()

    def render_sidebar(self) -> dict[str, Any]:
        """Render sidebar controls."""

        st.sidebar.header("Configuration")

        case_id = st.sidebar.text_input(
            "Case ID",
            value="MRI-CASE-001",
        )

        confidence = st.sidebar.slider(
            "Confidence threshold",
            min_value=0.0,
            max_value=1.0,
            value=self.config.confidence_threshold,
            step=0.05,
        )

        enable_classification = st.sidebar.checkbox(
            "Enable classification",
            value=self.config.enable_classification,
        )

        enable_segmentation = st.sidebar.checkbox(
            "Enable segmentation",
            value=self.config.enable_segmentation,
        )

        st.sidebar.divider()

        st.sidebar.subheader("System")

        st.sidebar.write(f"Model directory: `{self.config.model_directory}`")

        st.sidebar.write(f"Input channels: `{self.config.input_channels}`")

        st.sidebar.write(f"Modalities: `{', '.join(self.config.modalities)}`")

        return {
            "case_id": case_id,
            "confidence_threshold": confidence,
            "enable_classification": enable_classification,
            "enable_segmentation": enable_segmentation,
        }

    def render_upload_section(self) -> dict[str, Any]:
        """Render MRI upload controls."""

        st.header("1. MRI Input")

        st.info(
            "Upload NIfTI files for T1-native, T1-contrast, "
            "T2-weighted, and T2-FLAIR MRI modalities."
        )

        columns = st.columns(4)

        uploads: dict[str, Any] = {}

        labels = {
            "t1n": "T1-native",
            "t1c": "T1-contrast",
            "t2w": "T2-weighted",
            "t2f": "T2-FLAIR",
        }

        for index, modality in enumerate(self.config.modalities):
            with columns[index]:
                uploads[modality] = st.file_uploader(
                    labels[modality],
                    type=["nii", "gz"],
                    key=f"upload_{modality}",
                )

        return uploads

    @staticmethod
    def save_uploaded_file(
        uploaded_file: Any,
        directory: Path,
        filename: str,
    ) -> Path:
        """Save Streamlit uploaded file."""

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = directory / filename

        output_path.write_bytes(uploaded_file.getvalue())

        return output_path

    def prepare_mri_input(
        self,
        uploads: dict[str, Any],
        case_id: str,
    ) -> Path:
        """
        Prepare uploaded MRI files.

        Returns the directory containing the case files.
        """

        temporary_directory = Path(tempfile.mkdtemp(prefix=f"{case_id}_"))

        case_directory = temporary_directory / case_id
        case_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for modality in self.config.modalities:
            uploaded_file = uploads.get(modality)

            if uploaded_file is None:
                raise ValueError(f"Missing MRI modality: {modality}")

            original_name = uploaded_file.name

            if original_name.endswith(".nii.gz"):
                extension = ".nii.gz"
            elif original_name.endswith(".nii"):
                extension = ".nii"
            else:
                extension = ".nii"

            filename = f"{case_id}-{modality}{extension}"

            self.save_uploaded_file(
                uploaded_file,
                case_directory,
                filename,
            )

        return case_directory

    @staticmethod
    def load_preview_image(
        path: Path,
    ) -> np.ndarray | None:
        """Load an image from a generated PNG."""

        if not path.exists():
            return None

        return np.asarray(
            Image.open(path).convert("RGB")
        )

    def render_result_summary(
        self,
        result: Any,
    ) -> None:
        """Display pipeline result summary."""

        st.header("3. Analysis Result")

        inference_result = getattr(
            result,
            "inference_result",
            None,
        )

        if getattr(result, "success", False) is False:
            st.error(
                f"Analysis failed: {getattr(result, 'error', 'unknown error')}"
            )
            return

        columns = st.columns(3)

        with columns[0]:
            st.subheader("Classification")

            if inference_result is None:
                st.warning("Classification unavailable.")
            else:
                prediction = getattr(
                    inference_result,
                    "tumor_class",
                    "Unknown",
                )

                confidence = getattr(
                    inference_result,
                    "classification_confidence",
                    0.0,
                )

                tumor_detected = getattr(
                    inference_result,
                    "tumor_detected",
                    None,
                )

                st.metric(
                    "Prediction",
                    str(prediction),
                )

                st.metric(
                    "Confidence",
                    f"{float(confidence) * 100:.2f}%",
                )

                if tumor_detected is not None:
                    st.metric(
                        "Tumor detected",
                        "Yes" if tumor_detected else "No",
                    )

        with columns[1]:
            st.subheader("Segmentation")

            if inference_result is None:
                st.warning("Segmentation unavailable.")
            else:
                voxel_count = getattr(
                    inference_result,
                    "tumor_voxel_count",
                    0,
                )

                percentage = getattr(
                    inference_result,
                    "tumor_percentage",
                    0.0,
                )

                st.metric(
                    "Tumor voxels",
                    f"{int(voxel_count):,}",
                )

                st.metric(
                    "Tumor volume",
                    f"{float(percentage):.2f}%",
                )

                shape = getattr(
                    inference_result,
                    "segmentation_shape",
                    None,
                )

                if shape:
                    st.write(f"Mask shape: `{tuple(shape)}`")

        with columns[2]:
            st.subheader("Pipeline")

            st.success("Analysis completed.")

            case_id = getattr(
                result,
                "case_id",
                "Unknown",
            )

            elapsed = getattr(
                result,
                "elapsed_time",
                0.0,
            )

            st.write(f"Case ID: `{case_id}`")
            st.write(f"Elapsed: {float(elapsed):.2f}s")

    def render_output_files(
        self,
        result: Any,
    ) -> None:
        """Display generated output files."""

        st.header("4. Generated Outputs")

        visualization_result = getattr(
            result,
            "visualization_result",
            None,
        )

        report_text = getattr(
            result,
            "report_text",
            None,
        )

        report_json = getattr(
            result,
            "report_json",
            None,
        )

        if visualization_result is not None:
            st.subheader("MRI Visualizations")

            image_paths = {
                "MRI Slice": getattr(
                    visualization_result,
                    "mri_slice_path",
                    None,
                ),
                "Multimodal": getattr(
                    visualization_result,
                    "multimodal_path",
                    None,
                ),
                "Segmentation": getattr(
                    visualization_result,
                    "segmentation_path",
                    None,
                ),
                "Overlay": getattr(
                    visualization_result,
                    "overlay_path",
                    None,
                ),
                "Prediction Summary": getattr(
                    visualization_result,
                    "prediction_summary_path",
                    None,
                ),
            }

            columns = st.columns(2)

            column_index = 0

            for name, image_path in image_paths.items():
                if image_path is None:
                    continue

                path = Path(image_path)

                if not path.exists():
                    continue

                with columns[column_index % 2]:
                    st.image(
                        str(path),
                        caption=name,
                        use_container_width=True,
                    )

                column_index += 1

        if report_text or report_json:
            st.subheader("Reports")

            if report_text:
                path = Path(report_text)

                if path.exists():
                    st.download_button(
                        "Download Text Report",
                        data=path.read_bytes(),
                        file_name=path.name,
                        mime="text/plain",
                    )

            if report_json:
                path = Path(report_json)

                if path.exists():
                    st.download_button(
                        "Download JSON Report",
                        data=path.read_bytes(),
                        file_name=path.name,
                        mime="application/json",
                    )

    def render_run_button(
        self,
        uploads: dict[str, Any],
        settings: dict[str, Any],
    ) -> None:
        """Render analysis button and execute pipeline."""

        st.header("2. Run Analysis")

        ready = all(
            uploads.get(modality) is not None for modality in self.config.modalities
        )

        if not ready:
            st.warning("Upload all four MRI modalities before running analysis.")
            return

        if not settings["enable_classification"]:
            st.info("Classification is disabled.")

        if not settings["enable_segmentation"]:
            st.info("Segmentation is disabled.")

        if not st.button(
            "🔬 Run Brain Tumor Analysis",
            type="primary",
            use_container_width=True,
        ):
            return

        case_id = settings["case_id"].strip()

        if not case_id:
            st.error("Enter a Case ID.")
            return

        try:
            with st.spinner("Preparing MRI data..."):
                case_directory = self.prepare_mri_input(
                    uploads,
                    case_id,
                )

            st.success(f"MRI data prepared for `{case_id}`.")

            with st.spinner("Running brain tumor analysis..."):
                result = self.pipeline.process(case_directory)

            st.session_state["pipeline_result"] = result

            st.success("Brain tumor analysis completed successfully.")

        except Exception as exc:
            LOGGER.exception("Pipeline execution failed.")

            st.error(f"Analysis failed: {exc}")

    def render_existing_result(self) -> None:
        """Render stored pipeline result."""

        result = st.session_state.get("pipeline_result")

        if result is None:
            return

        self.render_result_summary(result)
        self.render_output_files(result)

    def render(self) -> None:
        """Render complete interface."""

        self.configure_page()
        self.render_header()

        settings = self.render_sidebar()
        uploads = self.render_upload_section()

        self.render_run_button(
            uploads,
            settings,
        )

        self.render_existing_result()

        st.divider()

        st.caption(
            "BrainTumorDiagnosis | MRI-based brain tumor "
            "classification and segmentation"
        )


def create_ui(
    config: UIConfig | None = None,
) -> BrainTumorUI:
    """Create the BrainTumorDiagnosis UI."""

    return BrainTumorUI(config=config)


def main() -> None:
    """Run the Streamlit application."""

    config = UIConfig()
    application = create_ui(config)

    application.render()


if __name__ == "__main__":
    main()
