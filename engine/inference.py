"""
BrainTumorDiagnosis Inference Engine

Purpose:
    Run inference for the complete brain tumor analysis workflow.

Workflow:
    MRI input
        ↓
    Preprocessing
        ↓
    Tumor segmentation
        ↓
    Tumor localization
        ↓
    Tumor classification
        ↓
    Prediction result

The engine supports:
    1. NIfTI MRI volumes
    2. Four BraTS MRI modalities
    3. 3D segmentation models
    4. 2D classification models
    5. GPU inference with CUDA
    6. CPU fallback
    7. Checkpoint loading
    8. JSON-compatible prediction results
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from datasets.preprocessing import (
    MRI_MODALITIES,
    normalize_volume as canonical_normalize_volume,
)
from models.factory import (
    ModelConfig,
    create_backbone_classifier,
    create_classification_model,
)
from models.segmentation import (
    create_segmentation_model as create_unet3d_segmentation_model,
)


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class InferenceConfig:
    """Configuration for the inference engine."""

    segmentation_checkpoint: str = (
        "checkpoints/segmentation/unet3d/best/unet3d_segmentation.pt"
    )

    classification_checkpoint: str = (
        "checkpoints/classification/densenet121/best.pt"
    )

    classification_backbone: str = "densenet121"

    device: str = "auto"

    use_amp: bool = True

    num_classes: int = 4

    segmentation_classes: int = 5

    in_channels: int = 4

    segmentation_base_channels: int = 16

    segmentation_dropout: float = 0.1

    segmentation_patch_size: tuple[int, int, int] = (96, 96, 96)

    segmentation_overlap: float = 0.5

    classification_image_size: tuple[int, int] = (224, 224)

    save_results: bool = True

    result_directory: str = "outputs/predictions"

    class_names: tuple[str, ...] = (
        "Glioma",
        "Meningioma",
        "Pituitary",
        "No Tumor",
    )

    def __post_init__(self) -> None:
        supported_backbones = {
            "cnn2d",
            "densenet121",
            "resnet50",
            "efficientnet_v2_s",
            "convnext_tiny",
            "swin_t",
            "vit_b_16",
        }

        if self.classification_backbone not in supported_backbones:
            raise ValueError(
                "classification_backbone must be one of "
                f"{sorted(supported_backbones)}: "
                f"{self.classification_backbone}"
            )

        if not 0.0 <= self.segmentation_overlap < 1.0:
            raise ValueError(
                "segmentation_overlap must be in [0, 1): "
                f"{self.segmentation_overlap}"
            )


@dataclass
class MRIInput:
    """Paths for the four MRI modalities."""

    t1n: Path
    t1c: Path
    t2w: Path
    t2f: Path

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        case_id: str | None = None,
    ) -> "MRIInput":
        """
        Build MRI input paths from a BraTS case directory.
        """

        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"MRI directory does not exist: {directory}")

        if case_id is None:
            case_id = directory.name

        paths = {}

        for modality in MRI_MODALITIES:
            candidates = [
                directory / f"{case_id}-{modality}.nii",
                directory / f"{case_id}-{modality}.nii.gz",
            ]

            found = next(
                (path for path in candidates if path.exists()),
                None,
            )

            if found is None:
                raise FileNotFoundError(
                    f"Missing MRI modality '{modality}' "
                    f"for case '{case_id}' in {directory}"
                )

            paths[modality] = found

        return cls(
            t1n=paths["t1n"],
            t1c=paths["t1c"],
            t2w=paths["t2w"],
            t2f=paths["t2f"],
        )


@dataclass
class InferenceResult:
    """Complete inference result."""

    tumor_detected: bool

    tumor_class: str

    tumor_class_index: int

    classification_confidence: float

    tumor_voxel_count: int

    tumor_percentage: float

    segmentation_shape: tuple[int, ...]

    inference_time_seconds: float

    segmentation_output_path: str | None = None

    classification_probabilities: dict[str, float] | None = None

    case_id: str | None = None

    segmentation_mask: np.ndarray | None = None

    @property
    def tumor_present(self) -> bool:
        """Backward-compatible tumor presence property."""

        return self.tumor_detected

    @property
    def segmentation(self) -> np.ndarray | None:
        """Expose the (H, W, D) integer mask for downstream tools."""

        return self.segmentation_mask

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to a JSON-compatible dictionary."""

        serializable = asdict(self)

        serializable.pop("segmentation_mask", None)

        return serializable

    def save_json(self, path: str | Path) -> None:
        """Save inference result as JSON."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            json.dump(
                self.to_dict(),
                file,
                indent=4,
            )


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


def resolve_device(device: str) -> torch.device:
    """Resolve the requested inference device."""

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)

    if resolved.type == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")

    return resolved


# ---------------------------------------------------------------------------
# MRI loading
# ---------------------------------------------------------------------------


def load_nifti(path: str | Path) -> np.ndarray:
    """
    Load a NIfTI image as float32 NumPy data.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"NIfTI file does not exist: {path}")

    image = nib.load(str(path))

    data = image.get_fdata(dtype=np.float32)

    if data.ndim != 3:
        raise ValueError(f"Expected 3D MRI volume, got shape {data.shape}")

    return data


def load_mri_modalities(
    mri_input: MRIInput,
) -> np.ndarray:
    """
    Load all four MRI modalities.

    Returns:
        NumPy array with shape:

        (4, H, W, D)
    """

    volumes = [
        load_nifti(mri_input.t1n),
        load_nifti(mri_input.t1c),
        load_nifti(mri_input.t2w),
        load_nifti(mri_input.t2f),
    ]

    reference_shape = volumes[0].shape

    for index, volume in enumerate(volumes):
        if volume.shape != reference_shape:
            raise ValueError(
                "MRI modality shape mismatch. "
                f"Expected {reference_shape}, "
                f"got {volume.shape} for modality "
                f"{MRI_MODALITIES[index]}"
            )

    return np.stack(
        volumes,
        axis=0,
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def normalize_volume(
    volume: np.ndarray,
) -> np.ndarray:
    """
    Perform non-zero z-score normalization.

    Delegates to the canonical implementation in
    datasets.preprocessing.normalize_volume.
    """

    return canonical_normalize_volume(
        volume,
        clip_percentiles=None,
        mask_background=True,
    )


def preprocess_mri(
    volume: np.ndarray,
) -> torch.Tensor:
    """
    Normalize four MRI modalities.

    Input:
        (4, H, W, D)

    Output:
        torch.Tensor
        (1, 4, H, W, D)
    """

    if volume.ndim != 4:
        raise ValueError(
            f"Expected MRI array with 4 dimensions, " f"got {volume.shape}"
        )

    normalized = np.stack(
        [normalize_volume(volume[index]) for index in range(volume.shape[0])],
        axis=0,
    )

    tensor = torch.from_numpy(normalized).float()

    return tensor.unsqueeze(0)


# ---------------------------------------------------------------------------
# Segmentation model
# ---------------------------------------------------------------------------


def load_segmentation_model(
    config: InferenceConfig,
    device: torch.device,
) -> torch.nn.Module:
    """Create and load the trained 3D U-Net segmentation model."""

    model = create_unet3d_segmentation_model(
        in_channels=config.in_channels,
        out_channels=config.segmentation_classes,
        base_channels=config.segmentation_base_channels,
        dropout=config.segmentation_dropout,
    )

    checkpoint_path = Path(config.segmentation_checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Segmentation checkpoint not found: " f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state_dict = extract_state_dict(checkpoint)

    model.load_state_dict(
        state_dict,
        strict=False,
    )

    model.to(device)
    model.eval()

    LOGGER.info(
        "Segmentation model loaded: %s",
        checkpoint_path,
    )

    return model


# ---------------------------------------------------------------------------
# Classification model
# ---------------------------------------------------------------------------


def load_classification_model(
    config: InferenceConfig,
    device: torch.device,
) -> torch.nn.Module:
    """Create and load the trained classification model."""

    if config.classification_backbone == "cnn2d":
        model = create_classification_model(
            ModelConfig(
                model_type="classification",
                in_channels=config.in_channels,
                num_classes=config.num_classes,
                classification_input_size=(config.classification_image_size),
            )
        )
    else:
        model = create_backbone_classifier(
            backbone=config.classification_backbone,
            num_classes=config.num_classes,
            pretrained=False,
        )

    checkpoint_path = _resolve_classification_checkpoint(
        Path(config.classification_checkpoint),
        config.classification_backbone,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state_dict = extract_state_dict(checkpoint)

    cleaned_state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }

    model.load_state_dict(
        cleaned_state_dict,
        strict=False,
    )

    model.to(device)
    model.eval()

    LOGGER.info(
        "Classification model loaded: %s (%s)",
        checkpoint_path,
        config.classification_backbone,
    )

    return model


def _resolve_classification_checkpoint(
    checkpoint_path: Path,
    backbone: str,
) -> Path:
    """
    Resolve the classification checkpoint path.

    Checks the configured location first, then the canonical
    per-backbone checkpoint-manager layout, then the flat legacy
    checkpoint produced by the standalone DenseNet121 trainer.
    """

    candidates = [
        checkpoint_path,
        Path(f"checkpoints/classification/{backbone}/best.pt"),
        Path(f"checkpoints/classification/{backbone}/last.pt"),
        Path("checkpoints/classification/densenet121/best.pt"),
        Path("checkpoints/densenet121_best.pt"),
    ]

    for candidate in candidates:
        if candidate.exists():
            if candidate != checkpoint_path:
                LOGGER.warning(
                    "Configured classification checkpoint not found: %s. "
                    "Using: %s",
                    checkpoint_path,
                    candidate,
                )

            return candidate

    raise FileNotFoundError(
        "Classification checkpoint not found. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def extract_state_dict(
    checkpoint: Any,
) -> dict[str, torch.Tensor]:
    """
    Extract a PyTorch state dictionary from common
    checkpoint formats.
    """

    if isinstance(checkpoint, dict):

        possible_keys = (
            "model_state_dict",
            "state_dict",
            "model",
        )

        for key in possible_keys:
            value = checkpoint.get(key)

            if isinstance(value, dict):
                return value

        if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            return checkpoint

    raise ValueError("Unable to find model state dictionary " "inside checkpoint.")


# ---------------------------------------------------------------------------
# Segmentation inference
# ---------------------------------------------------------------------------


@torch.inference_mode()
def predict_segmentation(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    use_amp: bool = True,
    patch_size: tuple[int, int, int] | None = None,
    overlap: float = 0.5,
) -> torch.Tensor:
    """
    Run 3D tumor segmentation.

    Input:
        (1, 4, H, W, D)

    Output:
        (H, W, D) integer segmentation mask.

    Volumes larger than ``patch_size`` are predicted with a
    sliding window (softmax probabilities are accumulated and
    averaged), keeping GPU memory within the 4 GB budget.
    """

    spatial_shape = tuple(image.shape[2:])

    if patch_size is None or all(
        size <= patch for size, patch in zip(spatial_shape, patch_size)
    ):
        return _predict_segmentation_whole_volume(
            model,
            image,
            device,
            use_amp,
        )

    LOGGER.info(
        "Volume shape %s exceeds patch size %s. "
        "Using sliding-window inference.",
        spatial_shape,
        tuple(patch_size),
    )

    return _predict_segmentation_windowed(
        model,
        image,
        device,
        use_amp,
        tuple(patch_size),
        overlap,
    )


def _predict_segmentation_whole_volume(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    use_amp: bool,
) -> torch.Tensor:
    """Single-pass segmentation prediction for small volumes."""

    image = image.to(
        device,
        non_blocking=True,
    )

    amp_enabled = use_amp and device.type == "cuda"

    with torch.autocast(
        device_type=device.type,
        enabled=amp_enabled,
    ):
        output = model(image)

    logits = _extract_segmentation_logits(output)

    prediction = torch.argmax(
        logits,
        dim=1,
    )

    return prediction[0].detach().cpu()


def _predict_segmentation_windowed(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    use_amp: bool,
    patch_size: tuple[int, int, int],
    overlap: float,
) -> torch.Tensor:
    """
    Sliding-window prediction with softmax probability averaging.

    This mirrors the evaluation logic in
    evaluation.segmentation_evaluator.predict_full_volume.
    """
    from evaluation.segmentation_evaluator import sliding_window_starts

    volume_shape = tuple(image.shape[2:])

    windows = sliding_window_starts(
        volume_shape,
        patch_size,
        overlap,
    )

    probability_sum: torch.Tensor | None = None

    counts = torch.zeros(
        volume_shape,
        dtype=torch.float32,
    )

    num_classes = None

    image = image.to(
        device,
        non_blocking=True,
    )

    amp_enabled = use_amp and device.type == "cuda"

    for window in windows:

        patch = image[
            :,
            :,
            window[0],
            window[1],
            window[2],
        ]

        with torch.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            output = model(patch)

        logits = _extract_segmentation_logits(output).float()

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0].cpu()

        if probability_sum is None:
            num_classes = probabilities.shape[0]

            probability_sum = torch.zeros(
                (num_classes, *volume_shape),
                dtype=torch.float32,
            )

        probability_sum[
            :,
            window[0],
            window[1],
            window[2],
        ] += probabilities

        counts[
            window[0],
            window[1],
            window[2],
        ] += 1.0

    counts = counts.clamp(min=1.0)

    averaged = probability_sum / counts.unsqueeze(0)

    prediction = torch.argmax(
        averaged,
        dim=0,
    )

    return prediction


def _extract_segmentation_logits(
    output: Any,
) -> torch.Tensor:
    """Normalize segmentation model output to a logits tensor."""

    if isinstance(output, dict):
        output = output.get("segmentation") or output.get("mask") or output.get("out")

    if isinstance(output, (tuple, list)):
        output = output[0]

    if not isinstance(output, torch.Tensor) or output.ndim != 5:
        raise RuntimeError(
            "Segmentation model must return a 5D logits tensor. "
            f"Received: {type(output).__name__} "
            f"{tuple(output.shape) if isinstance(output, torch.Tensor) else ''}"
        )

    return output


# ---------------------------------------------------------------------------
# Tumor localization
# ---------------------------------------------------------------------------


def analyze_segmentation(
    segmentation: torch.Tensor,
) -> tuple[bool, int, float]:
    """
    Calculate tumor presence and tumor volume statistics.

    Background is assumed to be class 0.
    """

    mask = segmentation.numpy()

    tumor_mask = mask > 0

    tumor_voxel_count = int(tumor_mask.sum())

    total_voxels = int(mask.size)

    tumor_percentage = (
        tumor_voxel_count / total_voxels * 100.0 if total_voxels > 0 else 0.0
    )

    tumor_detected = tumor_voxel_count > 0

    return (
        tumor_detected,
        tumor_voxel_count,
        tumor_percentage,
    )


# ---------------------------------------------------------------------------
# Segmentation output
# ---------------------------------------------------------------------------


def save_segmentation_nifti(
    segmentation: torch.Tensor,
    reference_path: str | Path,
    output_path: str | Path,
) -> None:
    """
    Save segmentation mask as NIfTI using the
    reference MRI affine and header.
    """

    reference_path = Path(reference_path)

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference = nib.load(str(reference_path))

    mask = segmentation.numpy().astype(np.uint8)

    image = nib.Nifti1Image(
        mask,
        affine=reference.affine,
        header=reference.header,
    )

    nib.save(
        image,
        str(output_path),
    )

    LOGGER.info(
        "Segmentation saved: %s",
        output_path,
    )


# ---------------------------------------------------------------------------
# Classification preprocessing
# ---------------------------------------------------------------------------


def create_classification_image(
    image: torch.Tensor,
    segmentation: torch.Tensor,
    image_size: tuple[int, int],
    normalize_rgb: bool = False,
) -> torch.Tensor:
    """
    Convert the 3D MRI volume into a representative 2D
    classification input.

    The slice with the largest tumor area is selected and the
    three most informative modalities are kept:
        t1c, t2w, t2f

    For the DenseNet121 backbone (``normalize_rgb=True``) the
    z-scored channels are clipped, scaled to [0, 1] and
    ImageNet-normalized, matching the RGB preprocessing used
    when the model was trained.

    Input:
        image:
            (1, 4, H, W, D)

        segmentation:
            (H, W, D)

    Output:
        (1, 3, target_H, target_W)
    """

    if image.ndim != 5:
        raise ValueError(f"Expected 5D MRI tensor, got {image.shape}")

    if segmentation.ndim != 3:
        raise ValueError("Expected 3D segmentation mask, " f"got {segmentation.shape}")

    tumor_mask = segmentation > 0

    slice_scores = tumor_mask.sum(dim=(0, 1))

    if torch.any(slice_scores > 0):
        slice_index = int(torch.argmax(slice_scores).item())
    else:
        slice_index = segmentation.shape[-1] // 2

    selected = image[
        :,
        1:4,
        :,
        :,
        slice_index,
    ]

    selected = F.interpolate(
        selected,
        size=image_size,
        mode="bilinear",
        align_corners=False,
    )

    if normalize_rgb:
        selected = _encode_rgb(selected)

    return selected


def _encode_rgb(
    image: torch.Tensor,
) -> torch.Tensor:
    """
    Encode z-scored MRI channels as an ImageNet-normalized RGB
    tensor.

    Each channel is clipped to [-2, 2] and scaled to [0, 1]
    before the ImageNet mean/std normalization is applied.
    """

    clipped = image.clamp(-2.0, 2.0)

    gray = (clipped + 2.0) / 4.0

    rgb = gray.expand(
        gray.shape[0],
        3,
        *gray.shape[2:],
    )

    mean = torch.tensor(
        (0.485, 0.456, 0.406),
        dtype=rgb.dtype,
        device=rgb.device,
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        (0.229, 0.224, 0.225),
        dtype=rgb.dtype,
        device=rgb.device,
    ).view(1, 3, 1, 1)

    return (rgb - mean) / std


# ---------------------------------------------------------------------------
# Classification inference
# ---------------------------------------------------------------------------


@torch.inference_mode()
def predict_classification(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    use_amp: bool = True,
) -> tuple[int, float, torch.Tensor]:
    """
    Run tumor classification.
    """

    image = image.to(
        device,
        non_blocking=True,
    )

    amp_enabled = use_amp and device.type == "cuda"

    with torch.autocast(
        device_type=device.type,
        enabled=amp_enabled,
    ):
        output = model(image)

    if isinstance(output, dict):
        output = (
            output.get("classification") or output.get("logits") or output.get("out")
        )

    if isinstance(output, (tuple, list)):
        output = output[0]

    if output.ndim != 2:
        raise RuntimeError(
            "Classification model must return "
            "a 2D tensor. "
            f"Received shape: {tuple(output.shape)}"
        )

    probabilities = torch.softmax(
        output.float(),
        dim=1,
    )

    confidence, prediction = torch.max(
        probabilities,
        dim=1,
    )

    return (
        int(prediction[0].item()),
        float(confidence[0].item()),
        probabilities[0].detach().cpu(),
    )


# ---------------------------------------------------------------------------
# Main inference engine
# ---------------------------------------------------------------------------


class InferenceEngine:
    """
    Complete MRI inference engine.

    Performs:

        MRI loading
        ↓
        preprocessing
        ↓
        segmentation
        ↓
        localization
        ↓
        classification
        ↓
        result generation
    """

    def __init__(
        self,
        config: InferenceConfig | None = None,
    ) -> None:

        self.config = config if config is not None else InferenceConfig()

        self.device = resolve_device(self.config.device)

        self.segmentation_model: torch.nn.Module | None = None

        self.classification_model: torch.nn.Module | None = None

        LOGGER.info("Inference engine initialized.")

        LOGGER.info(
            "Device: %s",
            self.device,
        )

        LOGGER.info(
            "AMP: %s",
            self.config.use_amp,
        )

    def load_models(self) -> None:
        """Load both trained models."""

        self.segmentation_model = load_segmentation_model(
            self.config,
            self.device,
        )

        self.classification_model = load_classification_model(
            self.config,
            self.device,
        )

    def predict(
        self,
        mri_input: MRIInput | np.ndarray,
        case_id: str | None = None,
        save_outputs: bool | None = None,
    ) -> InferenceResult:
        """
        Run complete inference on one MRI case.

        Args:
            mri_input: An :class:`MRIInput` path bundle or a raw
                (C, H, W, D) float32 volume (channels-first).
            case_id: Optional case identifier.
            save_outputs: Override the configured result saving.
        """

        if self.segmentation_model is None:
            raise RuntimeError("Segmentation model is not loaded.")

        if self.classification_model is None:
            raise RuntimeError("Classification model is not loaded.")

        start_time = time.perf_counter()

        LOGGER.info("Starting inference.")

        if isinstance(mri_input, MRIInput):
            image_array = load_mri_modalities(mri_input)
        else:
            image_array = np.asarray(mri_input, dtype=np.float32)

            if image_array.ndim != 4:
                raise ValueError(
                    "Raw MRI volume must be (C, H, W, D), "
                    f"got {tuple(image_array.shape)}"
                )

        image = preprocess_mri(image_array)

        segmentation = predict_segmentation(
            self.segmentation_model,
            image,
            self.device,
            self.config.use_amp,
            patch_size=self.config.segmentation_patch_size,
            overlap=self.config.segmentation_overlap,
        )

        segmentation_mask = segmentation.numpy().astype(np.int64)

        (
            tumor_detected,
            tumor_voxel_count,
            tumor_percentage,
        ) = analyze_segmentation(segmentation)

        classification_input = create_classification_image(
            image,
            segmentation,
            self.config.classification_image_size,
            normalize_rgb=(self.config.classification_backbone != "cnn2d"),
        )

        (
            class_index,
            confidence,
            probabilities,
        ) = predict_classification(
            self.classification_model,
            classification_input,
            self.device,
            self.config.use_amp,
        )

        class_name = (
            self.config.class_names[class_index]
            if class_index < len(self.config.class_names)
            else f"Class_{class_index}"
        )

        probability_dict = {
            (
                self.config.class_names[index]
                if index < len(self.config.class_names)
                else f"Class_{index}"
            ): float(probabilities[index].item())
            for index in range(probabilities.shape[0])
        }

        elapsed = time.perf_counter() - start_time

        segmentation_output_path = None

        should_save = self.config.save_results if save_outputs is None else save_outputs

        if should_save:
            output_directory = Path(self.config.result_directory)

            if case_id is None:
                if isinstance(mri_input, MRIInput):
                    case_id = mri_input.t1n.parent.name
                else:
                    case_id = "unknown_case"

            if isinstance(mri_input, MRIInput):
                segmentation_output = output_directory / f"{case_id}_segmentation.nii.gz"

                save_segmentation_nifti(
                    segmentation,
                    mri_input.t1n,
                    segmentation_output,
                )

                segmentation_output_path = str(segmentation_output)

        result = InferenceResult(
            tumor_detected=tumor_detected,
            tumor_class=class_name,
            tumor_class_index=class_index,
            classification_confidence=confidence,
            tumor_voxel_count=tumor_voxel_count,
            tumor_percentage=tumor_percentage,
            segmentation_shape=tuple(segmentation.shape),
            inference_time_seconds=elapsed,
            segmentation_output_path=(segmentation_output_path),
            classification_probabilities=(probability_dict),
            case_id=case_id,
            segmentation_mask=segmentation_mask,
        )

        if should_save:
            result_directory = Path(self.config.result_directory)

            if case_id is None:
                case_id = "unknown_case"

            result_path = result_directory / f"{case_id}_result.json"

            result.save_json(result_path)

        LOGGER.info("Inference completed.")

        LOGGER.info(
            "Tumor detected: %s",
            tumor_detected,
        )

        LOGGER.info(
            "Predicted class: %s",
            class_name,
        )

        LOGGER.info(
            "Classification confidence: %.4f",
            confidence,
        )

        LOGGER.info(
            "Tumor voxels: %d",
            tumor_voxel_count,
        )

        LOGGER.info(
            "Tumor percentage: %.4f%%",
            tumor_percentage,
        )

        LOGGER.info(
            "Inference time: %.3f seconds",
            elapsed,
        )

        return result


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def configure_logging() -> None:
    """Configure application logging."""

    logging.basicConfig(
        level=logging.INFO,
        format=("%(asctime)s | " "%(levelname)s | " "%(name)s | " "%(message)s"),
    )


# ---------------------------------------------------------------------------
# Module test
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Test the inference engine configuration.

    This test does not require trained checkpoints.
    """

    configure_logging()

    LOGGER.info("=" * 78)

    LOGGER.info("BrainTumorDiagnosis Inference Engine Test")

    LOGGER.info("=" * 78)

    config = InferenceConfig()

    device = resolve_device(config.device)

    LOGGER.info(
        "Device: %s",
        device,
    )

    if device.type == "cuda":
        LOGGER.info(
            "GPU: %s",
            torch.cuda.get_device_name(0),
        )

    LOGGER.info(
        "MRI modalities: %s",
        MRI_MODALITIES,
    )

    LOGGER.info(
        "Input channels: %d",
        config.in_channels,
    )

    LOGGER.info(
        "Classification classes: %d",
        config.num_classes,
    )

    LOGGER.info(
        "Segmentation classes: %d",
        config.segmentation_classes,
    )

    LOGGER.info("Inference configuration test passed.")

    LOGGER.info("=" * 78)


if __name__ == "__main__":
    main()
