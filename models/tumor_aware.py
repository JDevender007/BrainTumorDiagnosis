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
    # Phase 3A "t-only vs g+t" ablation (docs/TODO.md item A):
    #   "g+t" (default) = z = Phi(g, t)   — the executed model
    #   "t"             = z = t           — classifier sees only the
    #     tumor-region feature (global branch contributes nothing)
    fusion_mode: str = "g+t"
    # M_c (paper §III.C): SE-style channel attention inside TumorRegionPool,
    # applied AFTER M_s (CBAM spatial-then-channel order).  Default False =
    # the historically executed model (M_c was never wired in — Phase 0 audit
    # finding #4); Phase 3 ablation "+spatial+channel" sets it to True.
    use_channel_attention: bool = False
    # Segmentation decoder variant: "lightweight" = the historical 7x7-map
    # decoder with no skips (SegmentationHead); "multiscale" = U-Net-style
    # decoder over tapped DenseNet121 scales (MultiScaleSegmentationHead,
    # densenet121 only) — Phase 3 decoder ablation.
    seg_decoder: str = "lightweight"
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
    """Tumor-region feature extraction: M_s -> (optional M_c) -> GAP -> MLP.

    Exact operation (paper §III.C mapped to code, Phase 2 documentation):

        x_masked = x * mask                      # mask interpolated to x's size
        x_s      = M_s(x_masked)                 # SpatialAttention: 7x7 conv over
                                                 # [channel-avg, channel-max] concat
                                                 # -> sigmoid -> elementwise multiply
        x_sc     = M_c(x_s)                      # ChannelAttention (SE-style,
                                                 # reduction=16, shared MLP on avg+max
                                                 # pool -> sigmoid) — only when
                                                 # use_channel_attention=True;
                                                 # CBAM order: spatial THEN channel
        pooled   = GAP(x_sc)                     # AdaptiveAvgPool2d(1)
        t        = MLP(pooled)                   # Linear(in->fusion_dim) -> BatchNorm1d
                                                 # -> ReLU -> Dropout(0.2)

    Note the code applies GAP **after** attention and then an MLP; the paper's
    bare ``t = GAP(F_t)`` under-specifies the MLP — Phase 5 must state it.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_channel_attention: bool = False,
    ):
        super().__init__()
        self.attention = SpatialAttention(kernel_size=7)
        self.channel_attention = (
            ChannelAttention(in_channels) if use_channel_attention else None
        )
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
        if self.channel_attention is not None:
            attended = self.channel_attention(attended)
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
    """Lightweight 2D segmentation head for tumor mask prediction.

    Exact behaviour (Phase 2 documentation, paper §III.D):

        input  F: encoder feature map, for DenseNet121 + 224x224 input this is
               1024ch @ **7x7** (forward-hook output of ``encoder.features``)
        conv3x3 1024->128 (pad 1) -> BN -> ReLU
        bilinear x4                      -> 128ch @ 28x28
        conv3x3 128->64  (pad 1) -> BN -> ReLU
        bilinear x2                      ->  64ch @ 56x56
        conv1x1  64->num_classes         -> num_classes @ 56x56
        (TumorAwareMultiTaskNet.forward then interpolates to input size)

    **No skip connections and no multi-scale taps** — it decodes the 7x7 map
    alone.  The multi-scale alternative is :class:`MultiScaleSegmentationHead`
    (config ``seg_decoder="multiscale"``) used by the Phase 3 ablation.
    """

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


class MultiScaleSegmentationHead(nn.Module):
    """U-Net-style decoder over tapped DenseNet121 encoder scales.

    Phase 3 decoder ablation variant (config ``seg_decoder="multiscale"``).
    Only supported with ``backbone="densenet121"``; taps are registered on
    ``encoder.transition{1,2,3}`` by ``TumorAwareMultiTaskNet``.

    For a 224x224 input the taps are::

        transition1 -> 128ch @ 28x28
        transition2 -> 256ch @ 14x14
        transition3 -> 512ch @  7x 7
        deepest (encoder.features output) -> 1024ch @ 7x7

    Stages (each conv is 3x3, pad 1, no bias, + BN + ReLU):

        cat(deepest, transition3)  -> 256ch @  7x7
        up x2 + cat(transition2)   -> 128ch @ 14x14
        up x2 + cat(transition1)   ->  64ch @ 28x28
        up x2                      ->  64ch @ 56x56
        conv1x1 -> num_classes     -> num_classes @ 56x56

    As with the lightweight head, ``TumorAwareMultiTaskNet.forward``
    interpolates the result to the input resolution.
    """

    def __init__(self, num_classes: int, deepest_channels: int = 1024):
        super().__init__()
        # (deepest1024 + transition3 512) @7x7
        self.stage0 = self._conv_block(deepest_channels + 512, 256)
        # up-sampled 256 + transition2 256 @14x14
        self.stage1 = self._conv_block(256 + 256, 128)
        # up-sampled 128 + transition1 128 @28x28
        self.stage2 = self._conv_block(128 + 128, 64)
        # up-sampled 64, no skip @56x56
        self.stage3 = self._conv_block(64, 64)
        self.head = nn.Conv2d(64, num_classes, 1)
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, deepest: torch.Tensor, skips: list[torch.Tensor]
    ) -> torch.Tensor:
        """``skips`` = encoder taps in forward order [t1, t2, t3]."""
        if len(skips) != 3:
            raise RuntimeError(
                "MultiScaleSegmentationHead expects 3 encoder taps "
                f"(transition1/2/3), got {len(skips)}. Taps are registered "
                "only for backbone='densenet121'."
            )
        # taps arrive in forward (execution) order: transition1, 2, 3
        transition1, transition2, transition3 = skips
        x = self.stage0(torch.cat([deepest, transition3], dim=1))
        x = self.stage1(torch.cat([self.upsample(x), transition2], dim=1))
        x = self.stage2(torch.cat([self.upsample(x), transition1], dim=1))
        x = self.stage3(self.upsample(x))
        return self.head(x)


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
        # intermediate encoder taps for seg_decoder="multiscale"
        # (filled by forward hooks on encoder.transition{1,2,3})
        self._scale_taps: list[torch.Tensor] = []

        self._build_encoder(config)
        self._register_feature_hook()
        encoder_out = self._get_encoder_out_channels()

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_bn = nn.BatchNorm1d(encoder_out)

        self.tumor_pool = TumorRegionPool(
            encoder_out,
            config.fusion_dim,
            use_channel_attention=config.use_channel_attention,
        )

        if config.fusion_mode not in ("g+t", "t"):
            raise ValueError(
                f"fusion_mode must be 'g+t' or 't', got {config.fusion_mode!r}"
            )

        if config.fusion_mode == "t":
            # t-only ablation: z = t (fusion_dim); no fusion module is built
            # so its parameters never exist in this variant's state dict.
            self.fusion = None
            classifier_in = config.fusion_dim
        elif config.use_attention_fusion:
            self.fusion = AttentionFusion(encoder_out, config.fusion_dim, config.fusion_dim)
            classifier_in = config.fusion_dim
        else:
            self.fusion = None
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
            if config.seg_decoder not in ("lightweight", "multiscale"):
                raise ValueError(
                    f"seg_decoder must be 'lightweight' or 'multiscale', "
                    f"got {config.seg_decoder!r}"
                )
            if config.seg_decoder == "multiscale":
                if config.backbone != "densenet121":
                    raise ValueError(
                        "seg_decoder='multiscale' supports only "
                        "backbone='densenet121' (encoder taps are "
                        "backbone-specific)"
                    )
                self.seg_head: nn.Module | None = MultiScaleSegmentationHead(
                    config.segmentation_classes
                )
                self._register_multiscale_taps()
            else:
                self.seg_head = SegmentationHead(
                    encoder_out, config.segmentation_classes
                )
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

    def _register_multiscale_taps(self) -> None:
        """Capture transition1/2/3 outputs for the multi-scale seg decoder."""

        def tap_hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self._scale_taps.append(output)

        features = self.encoder.features
        for name in ("transition1", "transition2", "transition3"):
            getattr(features, name).register_forward_hook(tap_hook)

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
        self._scale_taps = []
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

    def fuse(self, global_feat: torch.Tensor, tumor_feat: torch.Tensor) -> torch.Tensor:
        """z = Φ(g, t) — the exact fusion op (named to avoid the paper's
        feature-map symbol F; Phase 2 code↔paper sync).

        ``use_attention_fusion=True`` (default — the executed model)::

            g' = W_g g,  t' = W_t t                 # Linear -> fusion_dim (256)
            w  = softmax(MLP([g'; t']))             # Linear(512->256) -> ReLU ->
                                                    # Linear(256->2) -> Softmax(2)
            z  = LayerNorm(w0*g' + w1*t')           # 2-way softmax-weighted
                                                    # convex combination + LayerNorm

        This is **not** the plain concatenation ``g ⊥ t`` the paper's §III.E
        notation implies — Phase 5 must rewrite the paper (or the ablation
        ``use_attention_fusion=False`` must be reported).

        ``use_attention_fusion=False`` (ablation only): ``z = concat(g, t)``.

        ``fusion_mode="t"`` (Phase 3A t-only ablation): ``z = t`` — only the
        tumor-region feature reaches the classifier.
        """
        if getattr(self.config, "fusion_mode", "g+t") == "t":
            return tumor_feat
        if self.config.use_attention_fusion:
            return self.fusion(global_feat, tumor_feat)
        return torch.cat([global_feat, tumor_feat], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        seg_mask: Optional[torch.Tensor] = None,
        return_seg: bool = False,
    ) -> dict[str, torch.Tensor]:
        global_feat, feature_map = self.encode(x)
        global_feat = self.global_bn(global_feat)

        tumor_feat = self.tumor_pool(feature_map, seg_mask)

        fused = self.fuse(global_feat, tumor_feat)

        logits = self.classifier(fused)

        result: dict[str, torch.Tensor] = {
            "logits": logits,
            "probabilities": F.softmax(logits, dim=1),
            "global_features": global_feat,
            "tumor_features": tumor_feat,
            "fused_features": fused,
        }

        if return_seg and self.seg_head is not None and feature_map is not None:
            if isinstance(self.seg_head, MultiScaleSegmentationHead):
                seg_out = self.seg_head(feature_map, self._scale_taps)
            else:
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
        """MC Dropout uncertainty estimation (T = ``num_passes``, default 10).

        Phase 2 bug fix: the previous implementation called ``self.train()``,
        which also put **BatchNorm in train mode** — the 10 passes then mixed
        batch-statistics noise into every uncertainty number (and mutated the
        running stats).  Now the model is set to ``eval()`` first and ONLY the
        Dropout layers are re-activated, matching the known-good path in
        ``evaluation/uncertainty.py``.  The model's previous train/eval mode
        is restored before returning.

        Dropout sites sampled (config.dropout = 0.3): classifier 0.3 / 0.15,
        tumor-pool MLP 0.2, uncertainty head 0.3.

        NOTE: all pre-Phase-2 uncertainty numbers (ECE, entropy, MI, T=10
        curves) were produced by the buggy path -> ``TODO-RERUN`` (Phase 3F).
        """
        was_training = self.training
        self.eval()
        for module in self.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                module.train()
        try:
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
        finally:
            self.train(was_training)

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
