from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "classification"
TEST_DIR = DATASET_ROOT / "test"

CHECKPOINT_CANDIDATES = (
    PROJECT_ROOT / "checkpoints" / "classification" / "densenet121" / "best.pt",
    PROJECT_ROOT / "checkpoints" / "classification" / "densenet121" / "last.pt",
    PROJECT_ROOT / "checkpoints" / "densenet121_best.pt",
)

CHECKPOINT_PATH = next(
    (path for path in CHECKPOINT_CANDIDATES if path.exists()),
    CHECKPOINT_CANDIDATES[0],
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "densenet121_evaluation"

CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"
CLASSIFICATION_REPORT_PATH = OUTPUT_DIR / "classification_report.txt"
METRICS_PATH = OUTPUT_DIR / "evaluation_metrics.json"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.json"

IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 0

CLASS_NAMES = (
    "Glioma",
    "Meningioma",
    "Pituitary",
    "No Tumor",
)


logging.basicConfig(
    level=logging.INFO,
    format=("%(asctime)s | %(levelname)s | " "%(name)s | %(message)s"),
)

LOGGER = logging.getLogger(__name__)


class BrainTumorTestDataset(Dataset):
    """Dataset for evaluating the trained DenseNet121 model."""

    VALID_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }

    def __init__(
        self,
        root_dir: Path,
        transform: transforms.Compose,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.samples: List[Tuple[Path, int]] = []

        self.class_to_idx = {
            class_name: index for index, class_name in enumerate(CLASS_NAMES)
        }

        self._collect_samples()

    def _collect_samples(self) -> None:
        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Test directory does not exist: " f"{self.root_dir}"
            )

        class_dirs = {
            directory.name.lower(): directory
            for directory in self.root_dir.iterdir()
            if directory.is_dir()
        }

        for class_name in CLASS_NAMES:
            class_dir = class_dirs.get(class_name.lower())

            if class_dir is None:
                LOGGER.warning(
                    "Class directory missing: %s",
                    class_name,
                )
                continue

            class_index = self.class_to_idx[class_name]

            image_files = sorted(
                file_path
                for file_path in class_dir.rglob("*")
                if (
                    file_path.is_file()
                    and file_path.suffix.lower() in self.VALID_EXTENSIONS
                )
            )

            LOGGER.info(
                "%-12s: %d images",
                class_name,
                len(image_files),
            )

            for image_path in image_files:
                self.samples.append((image_path, class_index))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, int, str]:
        image_path, label = self.samples[index]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to load image: {image_path}") from exc

        if self.transform is not None:
            image = self.transform(image)

        return image, label, str(image_path)


def create_test_transform() -> transforms.Compose:
    """Create deterministic preprocessing for test evaluation."""

    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )


def create_dataloader() -> DataLoader:
    """Create the test dataloader."""

    LOGGER.info("Creating DenseNet121 test dataloader.")

    LOGGER.info(
        "Dataset root: %s",
        DATASET_ROOT,
    )

    LOGGER.info(
        "Test root: %s",
        TEST_DIR,
    )

    dataset = BrainTumorTestDataset(
        TEST_DIR,
        transform=create_test_transform(),
    )

    if len(dataset) == 0:
        raise RuntimeError("No test images were found.")

    LOGGER.info(
        "Total test samples: %d",
        len(dataset),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    LOGGER.info(
        "Test batches: %d",
        len(dataloader),
    )

    return dataloader


def get_device() -> torch.device:
    """Select CUDA when available."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LOGGER.info("Device: %s", device)

    if device.type == "cuda":
        LOGGER.info(
            "GPU: %s",
            torch.cuda.get_device_name(0),
        )

        LOGGER.info(
            "CUDA: %s",
            torch.version.cuda,
        )

        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        LOGGER.info(
            "GPU memory: %.2f GB",
            gpu_memory,
        )

    return device


def create_model(device: torch.device) -> nn.Module:
    LOGGER.info("Creating DenseNet121 model.")

    model = models.densenet121(weights=None)

    num_features = model.classifier.in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.30),
        nn.Linear(num_features, len(CLASS_NAMES)),
    )

    model = model.to(device)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"DenseNet121 checkpoint not found: {CHECKPOINT_PATH}"
        )

    LOGGER.info("Loading checkpoint: %s", CHECKPOINT_PATH)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():
        cleaned_key = key

        if cleaned_key.startswith("module."):
            cleaned_key = cleaned_key[7:]

        cleaned_state_dict[cleaned_key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    model.eval()

    LOGGER.info("DenseNet121 checkpoint loaded successfully.")

    return model


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[
    List[int],
    List[int],
    List[float],
    List[str],
    List[List[float]],
]:
    """Run model inference over the complete test set."""

    LOGGER.info("Starting DenseNet121 test evaluation.")

    all_labels: List[int] = []
    all_predictions: List[int] = []
    all_confidences: List[float] = []
    all_paths: List[str] = []
    all_probabilities: List[List[float]] = []

    total_batches = len(dataloader)

    for batch_index, (
        images,
        labels,
        paths,
    ) in enumerate(dataloader, start=1):

        images = images.to(
            device,
            non_blocking=True,
        )

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

        confidences, predictions = torch.max(
            probabilities,
            dim=1,
        )

        all_labels.extend(labels.cpu().numpy().tolist())

        all_predictions.extend(predictions.cpu().numpy().tolist())

        all_confidences.extend(confidences.cpu().numpy().tolist())

        all_paths.extend(paths)

        all_probabilities.extend(
            probabilities.cpu().numpy().tolist()
        )

        if batch_index == 1 or batch_index % 25 == 0 or batch_index == total_batches:
            LOGGER.info(
                "Evaluated batch %d/%d",
                batch_index,
                total_batches,
            )

    LOGGER.info("Inference completed.")

    return (
        all_labels,
        all_predictions,
        all_confidences,
        all_paths,
        all_probabilities,
    )


def calculate_metrics(
    labels: List[int],
    predictions: List[int],
    probabilities: List[List[float]] | None = None,
) -> Dict[str, float]:
    """Calculate classification metrics."""

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    precision = precision_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
    }

    if probabilities is not None:
        try:
            metrics["roc_auc_ovr"] = float(
                roc_auc_score(
                    labels,
                    probabilities,
                    multi_class="ovr",
                    average="weighted",
                    labels=list(range(len(CLASS_NAMES))),
                )
            )
        except ValueError:
            LOGGER.warning(
                "ROC AUC could not be computed (requires at least one "
                "example per class)."
            )

    return metrics


def save_metrics(
    metrics: Dict[str, float],
) -> None:
    """Save evaluation metrics as JSON."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    LOGGER.info(
        "Metrics saved: %s",
        METRICS_PATH,
    )


def save_classification_report(
    labels: List[int],
    predictions: List[int],
) -> None:
    """Save detailed classification report."""

    report = classification_report(
        labels,
        predictions,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    with open(
        CLASSIFICATION_REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report)

    LOGGER.info(
        "Classification report saved: %s",
        CLASSIFICATION_REPORT_PATH,
    )

    LOGGER.info(
        "\n%s",
        report,
    )


def save_confusion_matrix(
    labels: List[int],
    predictions: List[int],
) -> None:
    """Create and save confusion matrix."""

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
    )

    figure, axis = plt.subplots(figsize=(9, 7))

    image = axis.imshow(
        matrix,
        interpolation="nearest",
    )

    figure.colorbar(image)

    axis.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        title="DenseNet121 Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    threshold = matrix.max() / 2.0

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color=("white" if matrix[row, column] > threshold else "black"),
                fontsize=12,
            )

    figure.tight_layout()

    figure.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    LOGGER.info(
        "Confusion matrix saved: %s",
        CONFUSION_MATRIX_PATH,
    )


def save_predictions(
    labels: List[int],
    predictions: List[int],
    confidences: List[float],
    paths: List[str],
) -> None:
    """Save image-level predictions."""

    prediction_records = []

    for true_label, predicted_label, confidence, path in zip(
        labels,
        predictions,
        confidences,
        paths,
    ):
        prediction_records.append(
            {
                "image": path,
                "true_class": CLASS_NAMES[true_label],
                "predicted_class": CLASS_NAMES[predicted_label],
                "confidence": float(confidence),
                "correct": (true_label == predicted_label),
            }
        )

    with open(
        PREDICTIONS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            prediction_records,
            file,
            indent=4,
        )

    LOGGER.info(
        "Predictions saved: %s",
        PREDICTIONS_PATH,
    )


def print_summary(
    metrics: Dict[str, float],
) -> None:
    """Print final evaluation summary."""

    accuracy = metrics["accuracy"] * 100
    precision = metrics["precision_weighted"] * 100
    recall = metrics["recall_weighted"] * 100
    f1 = metrics["f1_weighted"] * 100

    print()
    print("=" * 78)
    print("DENSENET121 TEST EVALUATION COMPLETE")
    print("=" * 78)
    print(f"Test Accuracy       : {accuracy:.2f}%")
    print(f"Weighted Precision  : {precision:.2f}%")
    print(f"Weighted Recall     : {recall:.2f}%")
    print(f"Weighted F1 Score   : {f1:.2f}%")

    if "roc_auc_ovr" in metrics:
        print(f"ROC AUC (one-vs-rest): {metrics['roc_auc_ovr']:.4f}")
    print()
    print(f"Checkpoint           : {CHECKPOINT_PATH}")
    print(f"Metrics              : {METRICS_PATH}")
    print(f"Classification Report: " f"{CLASSIFICATION_REPORT_PATH}")
    print(f"Confusion Matrix     : " f"{CONFUSION_MATRIX_PATH}")
    print(f"Predictions          : {PREDICTIONS_PATH}")
    print("=" * 78)


def main() -> None:
    """Run complete DenseNet121 evaluation."""

    LOGGER.info("=" * 78)
    LOGGER.info("BrainTumorDiagnosis DenseNet121 Evaluation")
    LOGGER.info("=" * 78)

    device = get_device()

    dataloader = create_dataloader()

    model = create_model(device)

    (
        labels,
        predictions,
        confidences,
        paths,
        probabilities,
    ) = evaluate_model(
        model,
        dataloader,
        device,
    )

    metrics = calculate_metrics(
        labels,
        predictions,
        probabilities,
    )

    save_metrics(metrics)

    save_classification_report(
        labels,
        predictions,
    )

    save_confusion_matrix(
        labels,
        predictions,
    )

    save_predictions(
        labels,
        predictions,
        confidences,
        paths,
    )

    print_summary(metrics)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.warning("Evaluation interrupted by user.")
        sys.exit(1)
    except Exception:
        LOGGER.exception("DenseNet121 evaluation failed.")
        sys.exit(1)
