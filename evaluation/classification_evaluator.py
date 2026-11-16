"""
BrainTumorDiagnosis unified classification test evaluator.

Evaluates any trained backbone on the official Test split:

    cnn2d, densenet121, resnet50, efficientnet_v2_s,
    convnext_tiny, swin_t

The Test split is never used for validation; this evaluator is
run only after training completes.

Usage (from the project root):

    python -m evaluation.classification_evaluator --backbone densenet121
    python -m evaluation.classification_evaluator --backbone cnn2d
    python -m evaluation.classification_evaluator --backbone resnet50 --checkpoint checkpoints/classification/resnet50/best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from datasets.classification import create_classification_dataloaders
from models.factory import (
    ModelConfig,
    create_backbone_classifier,
    create_classification_model,
)


LOGGER = logging.getLogger(__name__)


CLASS_NAMES = (
    "Glioma",
    "Meningioma",
    "Pituitary",
    "No Tumor",
)


SUPPORTED_BACKBONES = (
    "cnn2d",
    "densenet121",
    "resnet50",
    "efficientnet_v2_s",
    "convnext_tiny",
    "swin_t",
    "vit_b_16",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained classification backbone on the test split."
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="cnn2d",
        choices=SUPPORTED_BACKBONES,
        help="Model backbone to evaluate.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path (defaults to checkpoints/classification/{backbone}/best.pt).",
    )

    parser.add_argument(
        "--output-directory",
        type=str,
        default=None,
        help="Output directory (defaults to outputs/{backbone}_evaluation).",
    )

    return parser


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        LOGGER.info("Device: cuda")
        LOGGER.info("GPU: %s", torch.cuda.get_device_name(0))
        return device

    device = torch.device("cpu")
    LOGGER.info("Device: cpu")
    return device


def resolve_checkpoint(
    backbone: str,
    checkpoint: str | None,
) -> Path:
    candidates = [
        checkpoint,
        f"checkpoints/classification/{backbone}/best.pt",
        f"checkpoints/classification/{backbone}/last.pt",
    ]

    for candidate in candidates:
        if candidate is None:
            continue

        path = Path(candidate)

        if path.exists():
            return path

    raise FileNotFoundError(
        "Classification checkpoint not found. Tried: "
        + ", ".join(path for path in candidates if path is not None)
    )


def load_model(
    device: torch.device,
    backbone: str,
    checkpoint_path: Path,
) -> torch.nn.Module:
    LOGGER.info("Loading classification model: %s", backbone)

    LOGGER.info("Loading checkpoint: %s", checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned_state_dict[key] = value

    if backbone == "cnn2d":
        stem_weight = cleaned_state_dict.get("stem.0.weight")
        base_channels = stem_weight.shape[0] if stem_weight is not None else 32
        model = create_classification_model(
            ModelConfig(
                model_type="classification",
                in_channels=3,
                num_classes=4,
                classification_input_size=(224, 224),
                pretrained=False,
                base_channels=base_channels,
            )
        )
    else:
        model = create_backbone_classifier(
            backbone=backbone,
            num_classes=4,
            pretrained=False,
        )

    model.load_state_dict(cleaned_state_dict, strict=False)

    model.to(device)
    model.eval()

    LOGGER.info("Classification model loaded successfully.")

    return model


def evaluate_model(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
) -> tuple[list[int], list[int], list[float], list[list[float]]]:
    LOGGER.info("Starting classification evaluation.")

    all_labels: list[int] = []
    all_predictions: list[int] = []
    all_confidences: list[float] = []
    all_probabilities: list[list[float]] = []

    with torch.no_grad():
        for images, labels in tqdm(
            test_loader,
            desc="Evaluating",
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)

            if isinstance(outputs, dict):
                if "classification" in outputs:
                    logits = outputs["classification"]
                elif "logits" in outputs:
                    logits = outputs["logits"]
                else:
                    raise RuntimeError(
                        "Model output dictionary does not contain "
                        "'classification' or 'logits'."
                    )
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            probabilities = torch.softmax(logits, dim=1)

            confidences, predictions = torch.max(
                probabilities,
                dim=1,
            )

            all_labels.extend(labels.cpu().tolist())

            all_predictions.extend(predictions.cpu().tolist())

            all_confidences.extend(confidences.cpu().tolist())

            all_probabilities.extend(probabilities.cpu().tolist())

    LOGGER.info(
        "Evaluation samples: %d",
        len(all_labels),
    )

    return (
        all_labels,
        all_predictions,
        all_confidences,
        all_probabilities,
    )


def calculate_metrics(
    labels: list[int],
    predictions: list[int],
    probabilities: list[list[float]] | None = None,
) -> dict:
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

    report = classification_report(
        labels,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
    )

    metrics = {
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "classes": list(CLASS_NAMES),
        "test_samples": len(labels),
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
    metrics: dict,
    output_directory: Path,
) -> Path:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_directory / "evaluation_metrics.json"

    with output_path.open(
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
        output_path,
    )

    return output_path


def save_predictions(
    labels: list[int],
    predictions: list[int],
    confidences: list[float],
    output_directory: Path,
) -> Path:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_directory / "predictions.json"

    prediction_data = []

    for index, (
        label,
        prediction,
        confidence,
    ) in enumerate(
        zip(
            labels,
            predictions,
            confidences,
        )
    ):
        prediction_data.append(
            {
                "sample_index": index,
                "true_class": CLASS_NAMES[label],
                "predicted_class": CLASS_NAMES[prediction],
                "confidence": float(confidence),
                "correct": label == prediction,
            }
        )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            prediction_data,
            file,
            indent=4,
        )

    LOGGER.info(
        "Predictions saved: %s",
        output_path,
    )

    return output_path


def print_results(metrics: dict) -> None:
    LOGGER.info("=" * 78)
    LOGGER.info("BrainTumorDiagnosis Classification Evaluation")
    LOGGER.info("=" * 78)

    LOGGER.info(
        "Test samples       : %d",
        metrics["test_samples"],
    )

    LOGGER.info(
        "Accuracy            : %.4f",
        metrics["accuracy"],
    )

    LOGGER.info(
        "Weighted precision  : %.4f",
        metrics["precision_weighted"],
    )

    LOGGER.info(
        "Weighted recall     : %.4f",
        metrics["recall_weighted"],
    )

    LOGGER.info(
        "Weighted F1         : %.4f",
        metrics["f1_weighted"],
    )

    if "roc_auc_ovr" in metrics:
        LOGGER.info(
            "ROC AUC (one-vs-rest): %.4f",
            metrics["roc_auc_ovr"],
        )

    LOGGER.info("-" * 78)
    LOGGER.info("Per-class results")

    report = metrics["classification_report"]

    for class_name in CLASS_NAMES:
        class_metrics = report[class_name]

        LOGGER.info(
            "%-12s | Precision: %.4f | Recall: %.4f | F1: %.4f | Support: %d",
            class_name,
            class_metrics["precision"],
            class_metrics["recall"],
            class_metrics["f1-score"],
            int(class_metrics["support"]),
        )

    LOGGER.info("-" * 78)
    LOGGER.info("Confusion matrix")

    matrix = metrics["confusion_matrix"]

    LOGGER.info("                 Predicted")

    LOGGER.info("                 Glioma  Meningioma  Pituitary  No Tumor")

    for class_name, row in zip(
        CLASS_NAMES,
        matrix,
    ):
        LOGGER.info(
            "%-12s     %6d %10d %10d %10d",
            class_name,
            row[0],
            row[1],
            row[2],
            row[3],
        )

    LOGGER.info("=" * 78)


def main() -> None:
    setup_logging()

    parser = build_parser()

    args = parser.parse_args()

    LOGGER.info("=" * 78)
    LOGGER.info("BrainTumorDiagnosis Classification Evaluation")
    LOGGER.info("=" * 78)
    LOGGER.info("Backbone: %s", args.backbone)

    checkpoint_path = resolve_checkpoint(
        args.backbone,
        args.checkpoint,
    )

    output_directory = Path(
        args.output_directory or f"outputs/{args.backbone}_evaluation"
    )

    device = get_device()

    LOGGER.info("Creating classification test dataloader.")

    _, test_loader = create_classification_dataloaders()

    LOGGER.info(
        "Test batches: %d",
        len(test_loader),
    )

    model = load_model(
        device,
        args.backbone,
        checkpoint_path,
    )

    labels, predictions, confidences, probabilities = evaluate_model(
        model,
        test_loader,
        device,
    )

    metrics = calculate_metrics(
        labels,
        predictions,
        probabilities,
    )

    metrics["checkpoint"] = str(checkpoint_path)

    save_metrics(
        metrics,
        output_directory,
    )

    save_predictions(
        labels,
        predictions,
        confidences,
        output_directory,
    )

    print_results(metrics)

    LOGGER.info("Classification evaluation completed successfully.")


if __name__ == "__main__":
    main()