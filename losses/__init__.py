"""
BrainTumorDiagnosis Losses Package
"""

from .segmentation import (
    DiceLoss2D,
    DiceLoss3D,
    SegmentationLoss,
    segmentation_loss,
)

from .classification import (
    ClassificationLoss,
    create_classification_loss,
)


__all__ = [
    "DiceLoss2D",
    "DiceLoss3D",
    "SegmentationLoss",
    "segmentation_loss",
    "ClassificationLoss",
    "create_classification_loss",
]
