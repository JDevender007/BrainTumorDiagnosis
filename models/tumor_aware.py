"""
Tumor-Aware Multi-Task Brain Tumor Diagnosis Network.

Proposed architecture: TumorAwareMultiTaskNet

Central research question: Does segmentation-guided tumor-aware representation
learning improve classification generalization, robustness, calibration, and
interpretability compared with conventional image-level classification?

Architecture:
    MRI -> Shared Encoder -> [Global Features, Segmentation Features]
    Segmentation -> Tumor Mask -> Tumor Region Pooling
    Global Features + Tumor Features -> Attention Fusion -> Classification
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


@dataclass
class TumorAwareConfig:
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


class AttentionFusion(nn.Module):
    """Attention-based fusion of global and tumor-region features."""

    def __init__(self, global_dim: int, tumor_dim: int, fusion_dim: int):
        super().__init__()
        self.global_proj = nn.Linear(global_dim, fusion_dim)
        self.tumor_proj = nn.Linear(tumor_dim, fusion_dim)

        self.attention = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim, 2),
            nn.Softmax(dim=1),
        )

        self.fusion_norm = nn.LayerNorm(fusion_dim)

    def forward(self, global_feat: torch.Tensor, tumor_feat: torch.Tensor) -> torch.Tensor:
        g = self.global_proj(global_feat)
        t = self.tumor_proj(tumor_feat)

        combined = torch.cat([g, t], dim=1)
        attn_weights = self.attention(combined)

        fused = attn_weights[:, 0:1] * g + attn_weights[:, 1:2] * t
        fused = self.fusion_norm(fused)
        return fused


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


class TumorAwareMultiTaskNet(nn.Module):
    """
    Tumor-Aware Multi-Task Network for Brain Tumor Diagnosis.

    Architecture:
        MRI -> Shared Encoder -> [Global Features, Feature Map]
        Feature Map -> Segmentation Head -> Tumor Mask
        Feature Map + Tumor Mask -> Tumor Region Pooling -> Tumor Features
        Global Features + Tumor Features -> Attention Fusion -> Classification

    Modes:
        Mode A (image-only): MRI -> Encoder -> Classifier
        Mode B (seg-assisted): MRI -> Segmentation -> Tumor Region -> Classifier
        Mode C (proposed): Full tumor-aware fusion
    """

    def __init__(self, config: TumorAwareConfig):
        super().__init__()
        self.config = config
        self.backbone_name = config.backbone
        self._feature_map = None

        self._build_encoder(config)
        self._register_feature_hook()
        encoder_out = self._get_encoder_out_channels()

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_bn = nn.BatchNorm1d(encoder_out)

        self.tumor_pool = TumorRegionPool(encoder_out, config.fusion_dim)

        if config.use_attention_fusion:
            self.fusion = AttentionFusion(encoder_out, config.fusion_dim, config.fusion_dim)
            classifier_in = config.fusion_dim
        else:
            classifier_in = encoder_out + config.fusion_dim

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

        if config.use_segmentation_head:
            self.seg_head = SegmentationHead(encoder_out, config.segmentation_classes)
        else:
            self.seg_head = None

        if config.use_uncertainty_head:
            self.uncertainty_head = UncertaintyHead(classifier_in, config.num_classes, config.dropout)
        else:
            self.uncertainty_head = None

        self._init_weights()

    def _build_encoder(self, config: TumorAwareConfig):
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

        tumor_feat = self.tumor_pool(feature_map, seg_mask)

        if self.config.use_attention_fusion:
            fused = self.fusion(global_feat, tumor_feat)
        else:
            fused = torch.cat([global_feat, tumor_feat], dim=1)

        logits = self.classifier(fused)

        result: dict[str, torch.Tensor] = {
            "logits": logits,
            "probabilities": F.softmax(logits, dim=1),
            "global_features": global_feat,
            "tumor_features": tumor_feat,
            "fused_features": fused,
        }

        if return_seg and self.seg_head is not None and feature_map is not None:
            seg_out = self.seg_head(feature_map)
            if seg_out.shape[2:] != x.shape[2:]:
                seg_out = F.interpolate(seg_out, size=x.shape[2:], mode='bilinear', align_corners=False)
            result["seg_logits"] = seg_out

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


def create_tumor_aware_model(
    backbone: str = "densenet121",
    pretrained: bool = True,
    num_classes: int = 4,
    **kwargs,
) -> TumorAwareMultiTaskNet:
    config = TumorAwareConfig(
        backbone=backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        **kwargs,
    )
    return TumorAwareMultiTaskNet(config)
