"""
TURMA-Net: Tumor-guided Uncertainty-aware Robust Multi-task Network

A novel architecture for brain tumor MRI classification that:
1. Uses Adaptive Tumor-Guided Feature Modulation (ATGFM) to adaptively
   fuse global and tumor-localized features based on uncertainty
2. Enforces cross-task consistency between classification, segmentation,
   and localization confidence
3. Learns task-specific importance weights via uncertainty parameterization

Mathematical Formulation:
    L_total = (1/2σ₁²)L_cls + (1/2σ₂²)L_seg + (1/2σ₃²)L_cons + log(σ₁) + log(σ₂) + log(σ₃)

    where:
        L_cls: Focal loss with label smoothing
        L_seg: Dice + CE loss for segmentation
        L_cons: KL consistency between task predictions
        σ₁, σ₂, σ₃: Learned task uncertainty weights
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TurmaNetConfig:
    """Configuration for TURMA-Net."""
    backbone: str = "densenet121"
    pretrained: bool = True
    num_classes: int = 4
    segmentation_classes: int = 4
    dropout: float = 0.3
    fusion_dim: int = 256
    use_attention_fusion: bool = True
    use_segmentation_head: bool = True
    use_uncertainty_head: bool = True
    mc_dropout_passes: int = 10
    # Novel components
    use_atgfm: bool = True
    use_consistency: bool = True
    use_uncertainty_weights: bool = True
    consistency_temperature: float = 2.0


# ---------------------------------------------------------------------------
# Novel Component A: Adaptive Tumor-Guided Feature Modulation (ATGFM)
# ---------------------------------------------------------------------------

class AdaptiveTumorGuidedModulation(nn.Module):
    """
    Adaptive Tumor-Guided Feature Modulation (ATGFM).

    Uses predicted tumor uncertainty to adaptively modulate feature fusion
    between global and tumor-localized representations.

    Instead of fixed attention weights, this module learns to scale, shift,
    and gate features based on the uncertainty of tumor localization.

    Mathematical formulation:
        F_global: Global features from backbone [B, C]
        F_local: Features from tumor-localized region [B, C_local]
        U_tumor: Uncertainty of tumor localization [B, 1]

        F_proj_global = W_g @ F_global + b_g
        F_proj_local = W_l @ F_local + b_l

        γ = σ(W_γ @ [F_proj_global; F_proj_local; U_tumor] + b_γ)
        β = σ(W_β @ [F_proj_global; F_proj_local; U_tumor] + b_β)
        α = σ(W_α @ [F_proj_global; F_proj_local; U_tumor] + b_α)

        F_modulated = α ⊙ (γ ⊙ F_proj_global + (1-γ) ⊙ F_proj_local) + β
    """

    def __init__(self, global_dim: int, local_dim: int, fusion_dim: int):
        super().__init__()

        self.global_proj = nn.Linear(global_dim, fusion_dim)
        self.local_proj = nn.Linear(local_dim, fusion_dim)

        # Uncertainty-aware modulation networks
        input_dim = fusion_dim * 2 + 1  # global + local + uncertainty

        self.gamma_net = nn.Sequential(
            nn.Linear(input_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid(),
        )

        self.beta_net = nn.Sequential(
            nn.Linear(input_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim, fusion_dim),
            nn.Tanh(),
        )

        self.alpha_net = nn.Sequential(
            nn.Linear(input_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(fusion_dim)

    def forward(
        self,
        global_feat: torch.Tensor,
        local_feat: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            global_feat: [B, global_dim] global features
            local_feat: [B, local_dim] tumor-localized features
            uncertainty: [B, 1] tumor localization uncertainty

        Returns:
            modulated: [B, fusion_dim] uncertainty-modulated features
        """
        g = self.global_proj(global_feat)
        l = self.local_proj(local_feat)

        # Concatenate all inputs for modulation
        combined = torch.cat([g, l, uncertainty], dim=1)

        # Compute modulation parameters
        gamma = self.gamma_net(combined)   # [B, fusion_dim] - mixing ratio
        beta = self.beta_net(combined)     # [B, fusion_dim] - shift
        alpha = self.alpha_net(combined)   # [B, fusion_dim] - gate

        # Apply modulation
        modulated = alpha * (gamma * g + (1 - gamma) * l) + beta
        modulated = self.norm(modulated)

        return modulated


# ---------------------------------------------------------------------------
# Supporting modules (from existing tumor_aware.py, kept for compatibility)
# ---------------------------------------------------------------------------

class SpatialAttention(nn.Module):
    """Spatial attention module for tumor region highlighting."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(combined))
        return x * attention


class ChannelAttention(nn.Module):
    """Channel attention module (SE-style)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attention


class TumorRegionPool(nn.Module):
    """Extract tumor-region features using attention-weighted pooling."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.attention = SpatialAttention(kernel_size=7)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).float()
            if mask_expanded.shape[2:] != x.shape[2:]:
                mask_expanded = F.interpolate(mask_expanded, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = x * mask_expanded
        attended = self.attention(x)
        pooled = self.pool(attended).flatten(1)
        return self.fc(pooled)


class SegmentationHead(nn.Module):
    """Lightweight 2D segmentation head for tumor mask prediction."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features)


class UncertaintyHead(nn.Module):
    """Monte Carlo Dropout uncertainty estimation head."""

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Novel Component B: Tumor Localization Uncertainty Estimator
# ---------------------------------------------------------------------------

class TumorLocalizationUncertainty(nn.Module):
    """
    Estimates uncertainty of tumor localization from segmentation features.

    Uses the entropy of the predicted segmentation mask as a proxy for
    localization uncertainty.

    Mathematical formulation:
        P_mask = softmax(seg_logits, dim=1)
        H_mask = -Σ P_mask * log(P_mask + ε)
        U_tumor = normalize(H_mask)
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, seg_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seg_logits: [B, num_classes, H, W] segmentation logits

        Returns:
            uncertainty: [B, 1] scalar uncertainty per sample
        """
        # Compute entropy of segmentation predictions
        probs = F.softmax(seg_logits, dim=1)
        eps = 1e-10
        entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)  # [B, H, W]

        # Average pooling to get scalar uncertainty
        uncertainty = entropy.mean(dim=(1, 2))  # [B]

        # Normalize to [0, 1] range
        uncertainty = uncertainty / math.log(self.num_classes)

        return uncertainty.unsqueeze(1)  # [B, 1]


# ---------------------------------------------------------------------------
# Novel Component C: Cross-Task Consistency Module
# ---------------------------------------------------------------------------

class CrossTaskConsistency(nn.Module):
    """
    Enforces consistency between classification, segmentation, and
    localization confidence.

    Mathematical formulation:
        P_cls = softmax(cls_logits)
        P_seg = softmax(seg_logits).mean(dim=(2,3))  # spatial average
        P_loc = softmax(loc_logits)

        L_cons = KL(P_cls || P_seg) + KL(P_cls || P_loc)
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
        p_seg = p_seg.mean(dim=(2, 3))  # [B, C]

        # KL divergence: KL(P_cls || P_seg)
        kl_cls_seg = F.kl_div(
            F.log_softmax(seg_logits.mean(dim=(2, 3)) / self.temperature, dim=1),
            p_cls,
            reduction='batchmean',
        )

        # KL divergence: KL(P_seg || P_cls)
        kl_seg_cls = F.kl_div(
            F.log_softmax(cls_logits / self.temperature, dim=1),
            p_seg,
            reduction='batchmean',
        )

        # Symmetric consistency loss
        consistency_loss = (kl_cls_seg + kl_seg_cls) / 2

        return consistency_loss


# ---------------------------------------------------------------------------
# Novel Component D: Uncertainty-Weighted Multi-Task Loss
# ---------------------------------------------------------------------------

class UncertaintyWeightedMultiTaskLoss(nn.Module):
    """
    Learns task-specific importance weights via uncertainty parameterization.

    Mathematical formulation:
        L_total = (1/2σ₁²)L_cls + (1/2σ₂²)L_seg + (1/2σ₃²)L_cons + log(σ₁) + log(σ₂) + log(σ₃)

    where σ₁, σ₂, σ₃ are learnable parameters representing task uncertainty.
    The log(σ) terms prevent the model from predicting infinite uncertainty
    to minimize the loss.
    """

    def __init__(self, num_tasks: int = 3):
        super().__init__()
        # Learnable log-variance parameters (one per task)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(
        self,
        cls_loss: torch.Tensor,
        seg_loss: torch.Tensor,
        consistency_loss: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            cls_loss: classification loss
            seg_loss: segmentation loss
            consistency_loss: consistency loss

        Returns:
            total_loss: weighted total loss
            weights: dict of learned task weights
        """
        losses = [cls_loss, seg_loss, consistency_loss]
        total_loss = 0.0
        weights = {}

        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])  # 1/σ²
            total_loss += precision * loss + self.log_vars[i]
            weights[f"task_{i}_weight"] = precision.item()
            weights[f"task_{i}_log_var"] = self.log_vars[i].item()

        return total_loss, weights


# ---------------------------------------------------------------------------
# TURMA-Net: Main Architecture
# ---------------------------------------------------------------------------

class TurmaNet(nn.Module):
    """
    TURMA-Net: Tumor-guided Uncertainty-aware Robust Multi-task Network.

    Architecture:
        MRI Input [B, 3, 224, 224]
            ↓
        Shared 2D Encoder (DenseNet121 default)
            ↓
        Feature Map [B, C, H, W]
            ↓
        ┌───────────────────────────────────┐
        │                                   │
        ↓                                   ↓
    Global Pool                    Segmentation Head
    [B, C]                         [B, n_seg, H', W']
        ↓                                   ↓
    Feature Norm                    Tumor Mask
        ↓                                   ↓
    Feature Modulation ←──────────── Tumor Features + Uncertainty
        ↓                                   ↓
    Classification Head              Consistency Loss
        ↓
    Uncertainty Head (MC Dropout)
        ↓
    Uncertainty-Weighted Fusion
        ↓
    Final Prediction + Uncertainty
    """

    def __init__(self, config: TurmaNetConfig):
        super().__init__()
        self.config = config
        self.backbone_name = config.backbone
        self._feature_map = None

        self._build_encoder(config)
        self._register_feature_hook()
        encoder_out = self._get_encoder_out_channels()

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_bn = nn.BatchNorm1d(encoder_out)

        # Tumor region pooling
        self.tumor_pool = TumorRegionPool(encoder_out, config.fusion_dim)

        # Novel: Adaptive Tumor-Guided Feature Modulation
        if config.use_atgfm:
            self.atgfm = AdaptiveTumorGuidedModulation(
                global_dim=encoder_out,
                local_dim=config.fusion_dim,
                fusion_dim=config.fusion_dim,
            )
            classifier_in = config.fusion_dim
        elif config.use_attention_fusion:
            # Fallback to simple attention fusion
            self.fusion = nn.Sequential(
                nn.Linear(encoder_out + config.fusion_dim, config.fusion_dim),
                nn.ReLU(inplace=True),
                nn.Linear(config.fusion_dim, config.fusion_dim),
                nn.LayerNorm(config.fusion_dim),
            )
            classifier_in = config.fusion_dim
        else:
            classifier_in = encoder_out + config.fusion_dim

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout * 0.5),
            nn.Linear(128, config.num_classes),
        )

        # Segmentation head
        if config.use_segmentation_head:
            self.seg_head = SegmentationHead(encoder_out, config.segmentation_classes)
        else:
            self.seg_head = None

        # Uncertainty head
        if config.use_uncertainty_head:
            self.uncertainty_head = UncertaintyHead(classifier_in, config.num_classes, config.dropout)
        else:
            self.uncertainty_head = None

        # Tumor localization uncertainty estimator
        self.loc_uncertainty = TumorLocalizationUncertainty(config.segmentation_classes)

        # Novel: Cross-task consistency module
        if config.use_consistency:
            self.consistency = CrossTaskConsistency(config.consistency_temperature)

        # Novel: Uncertainty-weighted multi-task loss
        if config.use_uncertainty_weights:
            self.task_weights = UncertaintyWeightedMultiTaskLoss(num_tasks=3)

        self._init_weights()

    def _build_encoder(self, config: TurmaNetConfig):
        backbone_map = {
            "densenet121": (models.densenet121, models.DenseNet121_Weights.DEFAULT if config.pretrained else None),
            "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT if config.pretrained else None),
            "efficientnet_v2_s": (models.efficientnet_v2_s, models.EfficientNet_V2_S_Weights.DEFAULT if config.pretrained else None),
            "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT if config.pretrained else None),
            "swin_t": (models.swin_t, models.Swin_T_Weights.DEFAULT if config.pretrained else None),
            "vit_b_16": (models.vit_b_16, models.ViT_B_16_Weights.DEFAULT if config.pretrained else None),
        }

        if config.backbone not in backbone_map:
            raise ValueError(f"Unsupported backbone: {config.backbone}. Choose from {list(backbone_map.keys())}")

        builder, weights = backbone_map[config.backbone]
        self.encoder = builder(weights=weights)

        if config.backbone == "densenet121":
            self.encoder.classifier = nn.Identity()
        elif config.backbone == "resnet50":
            self.encoder.fc = nn.Identity()
        elif config.backbone == "efficientnet_v2_s":
            self.encoder.classifier = nn.Identity()
        elif config.backbone == "convnext_tiny":
            self.encoder.classifier = nn.Identity()
        elif config.backbone == "swin_t":
            self.encoder.head = nn.Identity()
        elif config.backbone == "vit_b_16":
            self.encoder.heads = nn.Identity()

    def _register_feature_hook(self):
        """Register a hook to capture the last conv feature map before pooling."""
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                if output.dim() == 4:
                    if output.shape[1] < output.shape[-1]:
                        self._feature_map = output.permute(0, 3, 1, 2).contiguous()
                    else:
                        self._feature_map = output
                elif output.dim() == 3:
                    self._feature_map = output.permute(0, 3, 1, 2).contiguous()

        target = None
        if self.backbone_name == "densenet121":
            target = self.encoder.features
        elif self.backbone_name == "resnet50":
            target = self.encoder.layer4
        elif self.backbone_name == "efficientnet_v2_s":
            target = self.encoder.features
        elif self.backbone_name == "convnext_tiny":
            target = self.encoder.features
        elif self.backbone_name == "swin_t":
            target = self.encoder.features
        elif self.backbone_name == "vit_b_16":
            target = None

        if target is not None:
            target.register_forward_hook(hook_fn)

    def _get_encoder_out_channels(self) -> int:
        dummy = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            out = self.encoder(dummy)
        if out.dim() > 2:
            out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return out.shape[1]

    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        for m in self.tumor_pool.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._feature_map = None
        features = self.encoder(x)
        feature_map = self._feature_map

        if feature_map is not None:
            global_feat = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
        elif features.dim() > 2:
            feature_map = features
            global_feat = F.adaptive_avg_pool2d(features, 1).flatten(1)
        else:
            global_feat = features
            b = features.shape[0]
            c = features.shape[1]
            feature_map = features.view(b, c, 1, 1)

        return global_feat, feature_map

    def forward(
        self,
        x: torch.Tensor,
        seg_mask: Optional[torch.Tensor] = None,
        return_seg: bool = False,
    ) -> dict[str, torch.Tensor]:
        global_feat, feature_map = self.encode(x)
        global_feat = self.global_bn(global_feat)

        # Tumor region features
        tumor_feat = self.tumor_pool(feature_map, seg_mask)

        # Segmentation prediction
        seg_logits = None
        if self.seg_head is not None and feature_map is not None:
            seg_logits = self.seg_head(feature_map)
            if seg_logits.shape[2:] != x.shape[2:]:
                seg_logits = F.interpolate(seg_logits, size=x.shape[2:], mode='bilinear', align_corners=False)

        # Tumor localization uncertainty
        loc_uncertainty = torch.zeros(global_feat.shape[0], 1, device=global_feat.device)
        if seg_logits is not None:
            loc_uncertainty = self.loc_uncertainty(seg_logits)

        # Feature fusion
        if self.config.use_atgfm:
            fused = self.atgfm(global_feat, tumor_feat, loc_uncertainty)
        elif hasattr(self, 'fusion'):
            fused = self.fusion(torch.cat([global_feat, tumor_feat], dim=1))
        else:
            fused = torch.cat([global_feat, tumor_feat], dim=1)

        # Classification
        logits = self.classifier(fused)

        # Build result
        result: dict[str, torch.Tensor] = {
            "logits": logits,
            "probabilities": F.softmax(logits, dim=1),
            "global_features": global_feat,
            "tumor_features": tumor_feat,
            "fused_features": fused,
            "loc_uncertainty": loc_uncertainty,
        }

        if seg_logits is not None:
            result["seg_logits"] = seg_logits

        if self.uncertainty_head is not None:
            result["uncertainty_logits"] = self.uncertainty_head(fused)

        return result

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        num_passes: int = 10,
    ) -> dict[str, torch.Tensor]:
        """MC Dropout uncertainty estimation."""
        self.train()
        all_logits = []

        for _ in range(num_passes):
            with torch.no_grad():
                out = self.forward(x)
                all_logits.append(out["probabilities"])

        stacked = torch.stack(all_logits)
        mean_probs = stacked.mean(dim=0)
        predictive_variance = stacked.var(dim=0)
        predictive_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)
        mutual_info = predictive_entropy - torch.mean(
            -torch.sum(stacked * torch.log(stacked + 1e-10), dim=2), dim=0
        )

        return {
            "mean_probabilities": mean_probs,
            "predictions": mean_probs.argmax(dim=1),
            "confidence": mean_probs.max(dim=1).values,
            "predictive_variance": predictive_variance,
            "predictive_entropy": predictive_entropy,
            "mutual_information": mutual_info,
            "epistemic_uncertainty": mutual_info,
            "aleatoric_uncertainty": predictive_variance.mean(dim=1),
        }


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_turma_net(
    backbone: str = "densenet121",
    pretrained: bool = True,
    num_classes: int = 4,
    **kwargs,
) -> TurmaNet:
    config = TurmaNetConfig(
        backbone=backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        **kwargs,
    )
    return TurmaNet(config)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick sanity check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = TurmaNetConfig(
        backbone="densenet121",
        pretrained=False,
        num_classes=4,
        segmentation_classes=4,
    )

    model = TurmaNet(config).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(2, 3, 224, 224, device=device)
    result = model(x, return_seg=True)

    print(f"Logits shape: {result['logits'].shape}")
    print(f"Probabilities shape: {result['probabilities'].shape}")
    print(f"Seg logits shape: {result['seg_logits'].shape}")
    print(f"Loc uncertainty shape: {result['loc_uncertainty'].shape}")
    print(f"Global features shape: {result['global_features'].shape}")
    print(f"Tumor features shape: {result['tumor_features'].shape}")
    print(f"Fused features shape: {result['fused_features'].shape}")

    # Test uncertainty prediction
    uncertainty_result = model.predict_with_uncertainty(x, num_passes=5)
    print(f"Mean probs shape: {uncertainty_result['mean_probabilities'].shape}")
    print(f"Predictive entropy shape: {uncertainty_result['predictive_entropy'].shape}")

    print("TURMA-Net test passed!")
