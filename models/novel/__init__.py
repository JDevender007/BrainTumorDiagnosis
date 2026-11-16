"""
TURMA-Net: Tumor-guided Uncertainty-aware Robust Multi-task Network.

Novel contributions:
1. Adaptive Tumor-Guided Feature Modulation (ATGFM)
2. Cross-Task Consistency Regularization
3. Uncertainty-Weighted Multi-Task Loss
"""

from models.novel.turma_net import TurmaNet, TurmaNetConfig, create_turma_net
from models.novel.losses import TurmaNetLoss, FocalLoss, DiceCELoss, ConsistencyLoss

__all__ = [
    "TurmaNet",
    "TurmaNetConfig",
    "create_turma_net",
    "TurmaNetLoss",
    "FocalLoss",
    "DiceCELoss",
    "ConsistencyLoss",
]
