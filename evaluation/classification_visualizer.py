from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LOGGER = logging.getLogger(__name__)

METRICS_FILE = Path("outputs/metrics/classification_evaluation.json")

OUTPUT_DIRECTORY = Path("outputs/figures/classification")

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "Pituitary",
    "No Tumor",
]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_metrics() -> dict:
    if not METRICS_FILE.exists():
        raise FileNotFoundError(f"Evaluation file not found: {METRICS_FILE}")

    with METRICS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def prepare_output_directory() -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


def save_figure(
    filename: str,
) -> Path:
    output_path = OUTPUT_DIRECTORY / filename

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    LOGGER.info(
        "Figure saved: %s",
        output_path,
    )

    return output_path


def plot_confusion_matrix(
    metrics: dict,
) -> None:
    matrix = np.array(metrics["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(9, 7))

    image = ax.imshow(
        matrix,
        interpolation="nearest",
    )

    fig.colorbar(
        image,
        ax=ax,
    )

    ax.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted Class",
        ylabel="True Class",
        title="Brain Tumor Classification Confusion Matrix",
    )

    threshold = matrix.max() / 2.0

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
                fontsize=12,
            )

    save_figure("confusion_matrix.png")


def plot_normalized_confusion_matrix(
    metrics: dict,
) -> None:
    matrix = np.array(
        metrics["confusion_matrix"],
        dtype=float,
    )

    row_totals = matrix.sum(
        axis=1,
        keepdims=True,
    )

    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix),
        where=row_totals != 0,
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    image = ax.imshow(
        normalized,
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Proportion",
    )

    ax.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted Class",
        ylabel="True Class",
        title="Normalized Confusion Matrix",
    )

    for row in range(normalized.shape[0]):
        for column in range(normalized.shape[1]):
            ax.text(
                column,
                row,
                f"{normalized[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=11,
            )

    save_figure("normalized_confusion_matrix.png")


def plot_per_class_metrics(
    metrics: dict,
) -> None:
    report = metrics["classification_report"]

    precision = [report[name]["precision"] for name in CLASS_NAMES]

    recall = [report[name]["recall"] for name in CLASS_NAMES]

    f1 = [report[name]["f1-score"] for name in CLASS_NAMES]

    x = np.arange(len(CLASS_NAMES))

    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.bar(
        x - width,
        precision,
        width,
        label="Precision",
    )

    ax.bar(
        x,
        recall,
        width,
        label="Recall",
    )

    ax.bar(
        x + width,
        f1,
        width,
        label="F1-score",
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.set_ylabel("Score")

    ax.set_xlabel("Brain Tumor Class")

    ax.set_title("Classification Performance by Class")

    ax.set_xticks(x)

    ax.set_xticklabels(CLASS_NAMES)

    ax.legend()

    for values, offset in [
        (precision, -width),
        (recall, 0),
        (f1, width),
    ]:
        for index, value in enumerate(values):
            ax.text(
                index + offset,
                value + 0.02,
                f"{value:.2f}",
                ha="center",
                fontsize=9,
            )

    save_figure("per_class_metrics.png")


def plot_classification_report(
    metrics: dict,
) -> None:
    report = metrics["classification_report"]

    values = np.array(
        [
            [
                report[name]["precision"],
                report[name]["recall"],
                report[name]["f1-score"],
            ]
            for name in CLASS_NAMES
        ]
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    image = ax.imshow(
        values,
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Score",
    )

    ax.set(
        xticks=np.arange(3),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=[
            "Precision",
            "Recall",
            "F1-score",
        ],
        yticklabels=CLASS_NAMES,
        xlabel="Metric",
        ylabel="Class",
        title="Classification Report",
    )

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(
                column,
                row,
                f"{values[row, column]:.3f}",
                ha="center",
                va="center",
                fontsize=11,
            )

    save_figure("classification_report.png")


def plot_overall_metrics(
    metrics: dict,
) -> None:
    names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
    ]

    values = [
        metrics["accuracy"],
        metrics["precision_weighted"],
        metrics["recall_weighted"],
        metrics["f1_weighted"],
    ]

    fig, ax = plt.subplots(figsize=(9, 6))

    bars = ax.bar(
        names,
        values,
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.set_ylabel("Score")

    ax.set_title("Overall Classification Performance")

    for bar, value in zip(
        bars,
        values,
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.2%}",
            ha="center",
            fontsize=11,
        )

    save_figure("overall_metrics.png")


def main() -> None:
    setup_logging()

    LOGGER.info("=" * 78)
    LOGGER.info("BrainTumorDiagnosis Classification Visualization")
    LOGGER.info("=" * 78)

    LOGGER.info("Loading evaluation metrics.")

    metrics = load_metrics()

    LOGGER.info(
        "Test samples: %d",
        metrics["test_samples"],
    )

    prepare_output_directory()

    plot_confusion_matrix(metrics)

    plot_normalized_confusion_matrix(metrics)

    plot_per_class_metrics(metrics)

    plot_classification_report(metrics)

    plot_overall_metrics(metrics)

    LOGGER.info("=" * 78)
    LOGGER.info("Classification visualization completed successfully.")
    LOGGER.info(
        "Output directory: %s",
        OUTPUT_DIRECTORY,
    )
    LOGGER.info("=" * 78)


if __name__ == "__main__":
    main()
