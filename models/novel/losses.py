"""
Loss functions for TURMA-Net.

Includes:
1. Focal Loss with Label Smoothing for classification
2. Dice + CE Combo Loss for segmentation
3. Cross-Task Consistency Loss
4. Uncertainty-Weighted Multi-Task Loss
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for classification with label smoothing.

    Mathematical formulation:
        FL(p_t) = -α_t (1 - p_t)^γ log(p_t)

    with label smoothing:
        y_smooth = (1 - ε) * y_onehot + ε / K

    Args:
        alpha: weighting factor
        gamma: focusing parameter
        label_smoothing: label smoothing factor
        num_classes: number of classes
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        num_classes: int = 4,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, num_classes] classification logits
            targets: [B] class indices

        Returns:
            loss: scalar focal loss
        """
        # Create smoothed one-hot targets
        with torch.no_grad():
            if self.label_smoothing > 0:
                smooth = torch.full_like(F.softmax(logits, dim=1), self.label_smoothing / self.num_classes)
                smooth.scatter_(1, targets.unsqueeze(1), 1 - self.label_smoothing + self.label_smoothing / self.num_classes)
            else:
                smooth = F.one_hot(targets, self.num_classes).float()

        # Compute cross-entropy
        log_probs = F.log_softmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)

        # Focal weight: (1 - p_t)^γ
        focal_weight = (1 - probs) ** self.gamma

        # Compute focal loss
        loss = -self.alpha * focal_weight * smooth * log_probs
        loss = loss.sum(dim=1).mean()

        return loss


class DiceCELoss(nn.Module):
    """
    Combined Dice Loss and Cross-Entropy Loss for segmentation.

    Mathematical formulation:
        L = λ_dice * L_dice + λ_ce * L_ce

    where:
        L_dice = 1 - (2 * |P ∩ T| + ε) / (|P| + |T| + ε)
        L_ce = -Σ t_k log(p_k)
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth

    def dice_loss(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute dice loss per class and average."""
        num_classes = probs.shape[1]

        # Handle different target dimensions
        if targets.dim() == probs.dim() - 1:
            # targets: [B, H, W] -> one-hot: [B, H, W, C] -> permute: [B, C, H, W]
            targets_one_hot = F.one_hot(targets.long(), num_classes).permute(0, 3, 1, 2).float()
        elif targets.dim() == probs.dim():
            # targets already one-hot or same shape
            targets_one_hot = targets.float()
        else:
            raise ValueError(f"Unexpected targets shape: {targets.shape}")

        # Compute dice per class
        dims = (0, 2, 3)
        intersection = (probs * targets_one_hot).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + targets_one_hot.sum(dim=dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_per_class.mean()

    def forward(self, seg_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seg_logits: [B, num_classes, D, H, W] segmentation logits
            targets: [B, D, H, W] segmentation targets

        Returns:
            loss: combined dice + CE loss
        """
        probs = F.softmax(seg_logits, dim=1)

        dice = self.dice_loss(probs, targets)
        ce = F.cross_entropy(seg_logits, targets.long())

        return self.dice_weight * dice + self.ce_weight * ce


class ConsistencyLoss(nn.Module):
    """
    Cross-Task Consistency Loss.

    Enforces consistency between classification and segmentation predictions.

    Mathematical formulation:
        P_cls = softmax(cls_logits / τ)
        P_seg = softmax(seg_logits / τ).mean(dim=(2,3))

        L_cons = KL(P_cls || P_seg) + KL(P_seg || P_cls)

    Args:
        temperature: softmax temperature for soft labels
    """

    def __init__(self, temperature: float = 2.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        cls_logits: torch.Tensor,
        seg_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            cls_logits: [B, num_classes] classification logits
            seg_logits: [B, num_classes, H, W] segmentation logits

        Returns:
            consistency_loss: scalar loss
        """
        # Classification probabilities
        p_cls = F.softmax(cls_logits / self.temperature, dim=1)  # [B, C]

        # Segmentation probabilities (spatial average)
        p_seg = F.softmax(seg_logits / self.temperature, dim=1)  # [B, C, H, W]
        p_seg_avg = p_seg.mean(dim=(2, 3))  # [B, C]

        # KL divergence: KL(P_cls || P_seg_avg)
        log_p_seg = F.log_softmax(seg_logits.mean(dim=(2, 3)) / self.temperature, dim=1)
        kl_cls_seg = F.kl_div(log_p_seg, p_cls.detach(), reduction='batchmean')

        # KL divergence: KL(P_seg_avg || P_cls)
        log_p_cls = F.log_softmax(cls_logits / self.temperature, dim=1)
        kl_seg_cls = F.kl_div(log_p_cls, p_seg_avg.detach(), reduction='batchmean')

        # Symmetric consistency loss
        return (kl_cls_seg + kl_seg_cls) / 2


class TurmaNetLoss(nn.Module):
    """
    Combined loss for TURMA-Net training.

    Mathematical formulation:
        L_total = (1/2σ₁²)L_cls + (1/2σ₂²)L_seg + (1/2σ₃²)L_cons + log(σ₁) + log(σ₂) + log(σ₃)

    where σ₁, σ₂, σ₃ are learnable task uncertainty weights.

    If use_uncertainty_weights=False, uses fixed weights:
        L_total = λ_cls * L_cls + λ_seg * L_seg + λ_cons * L_cons
    """

    def __init__(
        self,
        num_classes: int = 4,
        cls_weight: float = 1.0,
        seg_weight: float = 0.5,
        consistency_weight: float = 0.1,
        use_uncertainty_weights: bool = True,
        label_smoothing: float = 0.1,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.use_uncertainty_weights = use_uncertainty_weights

        # Individual losses
        self.cls_loss = FocalLoss(
            label_smoothing=label_smoothing,
            gamma=focal_gamma,
            num_classes=num_classes,
        )
        self.seg_loss = DiceCELoss()
        self.consistency_loss = ConsistencyLoss()

        # Uncertainty-weighted loss
        if use_uncertainty_weights:
            self.task_weights = nn.Parameter(torch.zeros(3))
        else:
            self.cls_weight = cls_weight
            self.seg_weight = seg_weight
            self.consistency_weight = consistency_weight

    def forward(
        self,
        cls_logits: torch.Tensor,
        targets: torch.Tensor,
        seg_logits: Optional[torch.Tensor] = None,
        seg_targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            cls_logits: [B, num_classes] classification logits
            targets: [B] classification targets
            seg_logits: [B, num_classes, H, W] segmentation logits (optional)
            seg_targets: [B, H, W] segmentation targets (optional)

        Returns:
            total_loss: combined loss
            loss_dict: dictionary of individual losses and weights
        """
        # Classification loss
        cls_loss = self.cls_loss(cls_logits, targets)

        # Segmentation loss (if available)
        seg_loss = torch.tensor(0.0, device=cls_logits.device)
        if seg_logits is not None and seg_targets is not None:
            # Ensure seg_targets has correct shape [B, H, W]
            if seg_targets.dim() == seg_logits.dim():
                # If targets are one-hot, convert to class indices
                seg_targets = seg_targets.argmax(dim=1)
            seg_loss = self.seg_loss(seg_logits, seg_targets)

        # Consistency loss (if segmentation available)
        cons_loss = torch.tensor(0.0, device=cls_logits.device)
        if seg_logits is not None:
            cons_loss = self.consistency_loss(cls_logits, seg_logits)

        # Combine losses
        if self.use_uncertainty_weights:
            # Uncertainty-weighted combination
            precisions = torch.exp(-self.task_weights)
            total_loss = (
                precisions[0] * cls_loss + self.task_weights[0]
                + precisions[1] * seg_loss + self.task_weights[1]
                + precisions[2] * cons_loss + self.task_weights[2]
            )
            weights = {
                "cls_weight": precisions[0].item(),
                "seg_weight": precisions[1].item(),
                "cons_weight": precisions[2].item(),
            }
        else:
            # Fixed weight combination
            total_loss = (
                self.cls_weight * cls_loss
                + self.seg_weight * seg_loss
                + self.consistency_weight * cons_loss
            )
            weights = {
                "cls_weight": self.cls_weight,
                "seg_weight": self.seg_weight,
                "cons_weight": self.consistency_weight,
            }

        loss_dict = {
            "total_loss": total_loss.item(),
            "cls_loss": cls_loss.item(),
            "seg_loss": seg_loss.item(),
            "cons_loss": cons_loss.item(),
            **weights,
        }

        return total_loss, loss_dict


# Import Optional at module level for type hints
from typing import Optional
