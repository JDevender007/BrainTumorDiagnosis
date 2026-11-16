"""
BrainTumorDiagnosis Losses Package
"""

from .segmentation import (
    DiceLoss3D,
    SegmentationLoss,
    segmentation_loss,
)

from .classification import (
    ClassificationLoss,
    create_classification_loss,
)


__all__ = [
    "DiceLoss3D",
    "SegmentationLoss",
    "segmentation_loss",
    "ClassificationLoss",
    "create_classification_loss",
]
