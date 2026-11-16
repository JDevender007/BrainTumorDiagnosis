# 04 ARCHITECTURE AUDIT

## Classification Architecture

### 2D Classification Models

All classification models operate on **2D RGB slices** of size 224×224.

| Model | Architecture | Input | Output | Parameters |
|-------|-------------|-------|--------|------------|
| CNN-2D | Custom residual CNN (4 residual blocks) | [B, 3, 224, 224] | [B, 4] | 21.40M |
| DenseNet121 | torchvision DenseNet121 | [B, 3, 224, 224] | [B, 4] | 6.96M |
| ResNet50 | torchvision ResNet50 | [B, 3, 224, 224] | [B, 4] | 23.52M |
| EfficientNetV2-S | torchvision EfficientNetV2-S | [B, 3, 224, 224] | [B, 4] | 20.18M |
| ConvNeXt-Tiny | torchvision ConvNeXt-Tiny | [B, 3, 224, 224] | [B, 4] | 27.82M |
| Swin-T | torchvision Swin-T | [B, 3, 224, 224] | [B, 4] | 27.52M |
| ViT-B/16 | torchvision ViT-B/16 | [B, 3, 224, 224] | [B, 4] | 85.80M |

**Classification is 2D slice-based.** No volume aggregation is performed. Each slice is classified independently.

### 3D Classification Model (UNUSED)

| Model | Architecture | Input | Output | Parameters |
|-------|-------------|-------|--------|------------|
| BrainTumorClassifier3D | 3-block 3D CNN | [B, 4, 32, 32, 32] | [B, 4] | ~0.5M |

**Status:** EXISTS but NOT TRAINED or EVALUATED.

---

## Segmentation Architecture

### 3D U-Net (UNet3D)

| Aspect | Value |
|--------|-------|
| Input | [B, 4, D, H, W] (4 MRI modalities) |
| Output | [B, 4, D, H, W] (4 segmentation classes) |
| Architecture | 4-level encoder-decoder with skip connections |
| Normalization | InstanceNorm3D |
| Activation | LeakyReLU(0.01) |
| Patch size | 96×96×96 |
| Base channels | 8 |
| Parameters | ~2.5M (estimated) |

**Encoder:**
- enc1: Conv3D(4→8) + Conv3D(8→8)
- enc2: MaxPool3D + Conv3D(8→16) + Conv3D(16→16)
- enc3: MaxPool3D + Conv3D(16→32) + Conv3D(32→32)
- bottleneck: MaxPool3D + Conv3D(32→64) + Conv3D(64→64)

**Decoder:**
- dec3: ConvTranspose3D(64→32) + skip + Conv3D(64→32) + Conv3D(32→32)
- dec2: ConvTranspose3D(32→16) + skip + Conv3D(32→16) + Conv3D(16→16)
- dec1: ConvTranspose3D(16→8) + skip + Conv3D(16→8) + Conv3D(8→8)
- output: Conv3D(8→4, 1×1×1)

**Status:** TRAINED (partial — epoch 30/100, best val Dice 0.5243)

---

## Proposed Model Architecture (TumorAwareMultiTaskNet)

### Architecture Overview

```
Input: MRI 2D slice [B, 3, 224, 224]
        ↓
    Shared 2D Encoder (DenseNet121 default)
        ↓
    ┌───────────────┐
    │ Feature Map   │  (captured via forward hook)
    │ [B, C, H, W]  │  (C=1024 for DenseNet121)
    └───────┬───────┘
            │
    ┌───────┴───────┐
    │               │
    ↓               ↓
Global Pool    Segmentation Head
[B, C]         [B, 4, H', W']
    │               │
    ↓               ↓
BN1d          Tumor Mask (upsampled)
    │               │
    │               ↓
    │         Tumor Region Pool
    │         (SpatialAttention + AdaptiveAvgPool + FC)
    │         [B, fusion_dim]
    │               │
    ↓               ↓
    └───────┬───────┘
            ↓
    Attention Fusion
    (learnable attention weights)
    [B, fusion_dim]
            │
    ┌───────┴───────┐
    │               │
    ↓               ↓
Classifier      Uncertainty Head
[B, 4]         (MC Dropout)
                [B, 4]
```

### Component Details

| Component | Input | Output | Parameters |
|-----------|-------|--------|------------|
| Shared Encoder | [B, 3, 224, 224] | [B, 1024, 7, 7] | 6.96M |
| Global Pool | [B, 1024, 7, 7] | [B, 1024] | 0 |
| Global BN | [B, 1024] | [B, 1024] | 2,048 |
| Segmentation Head | [B, 1024, 7, 7] | [B, 4, 28, 28] | ~0.5M |
| Tumor Region Pool | [B, 1024, 7, 7] + mask | [B, 256] | ~0.3M |
| Attention Fusion | [B, 1024] + [B, 256] | [B, 256] | ~0.5M |
| Classifier | [B, 256] | [B, 4] | ~0.1M |
| Uncertainty Head | [B, 256] | [B, 4] | ~0.1M |
| **Total** | | | **~9.03M** |

### Critical Architecture Issues

| Issue | Impact | Severity |
|-------|--------|----------|
| **2D segmentation head** — NOT connected to 3D U-Net | The "tumor mask" is predicted from 2D features, not from the actual 3D segmentation | **HIGH** |
| **No cross-domain bridge** — 2D classification and 3D segmentation operate in different domains | Cannot leverage actual 3D tumor localization for 2D classification | **HIGH** |
| **Segmentation head is trained end-to-end** — not using pre-trained 3D U-Net weights | The 2D segmentation head learns from scratch, not from the trained 3D U-Net | **MEDIUM** |
| **Feature hook captures last layer** — not multi-scale | Only captures the final feature map, not multi-scale features | **LOW** |

### Proposed Model Modes

| Mode | Description | Used in Experiments |
|------|-------------|-------------------|
| Mode A (image-only) | MRI → Encoder → Classifier (no tumor guidance) | YES |
| Mode B (seg-assisted) | MRI → Segmentation → Tumor Region → Classifier | YES |
| Mode C (proposed) | Full tumor-aware fusion with attention | YES |

---

## Architecture Comparison Table

| Aspect | Baseline Classifiers | TumorAwareMultiTaskNet | 3D U-Net |
|--------|--------------------|-----------------------|----------|
| Dimensionality | 2D | 2D | 3D |
| Input | RGB slice | RGB slice | 4-channel volume |
| Encoder | torchvision backbone | torchvision backbone | Custom Conv3D |
| Pretraining | ImageNet | ImageNet | None |
| Task | Classification | Classification + Segmentation | Segmentation |
| Uncertainty | MC Dropout (in evaluator) | MC Dropout (built-in) | None |
| Parameters | 6.96M-85.80M | 9.03M | ~2.5M |

---

## Summary

| Finding | Status |
|---------|--------|
| Classification is 2D | **CONFIRMED** |
| Segmentation is 3D | **CONFIRMED** |
| Proposed model is 2D | **CONFIRMED** |
| No 2D-3D domain bridge | **CONFIRMED** |
| Segmentation head is 2D, not connected to 3D U-Net | **CONFIRMED** |
| Architecture is a straightforward combination | **CONFIRMED** |
