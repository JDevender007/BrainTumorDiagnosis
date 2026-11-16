# 14 SEGMENTATION AUDIT

## Current Segmentation Implementation

### Model: 3D U-Net (UNet3D)

| Aspect | Value |
|--------|-------|
| Input | [B, 4, 96, 96, 96] |
| Output | [B, 4, 96, 96, 96] |
| Architecture | 4-level encoder-decoder |
| Normalization | InstanceNorm3D |
| Base channels | 8 |
| Training status | 30/100 epochs (undertrained) |

### Segmentation Labels (BraTS 2024)

| Label | Description | Available? |
|-------|-------------|------------|
| 0 | Background | YES |
| 1 | NETC | YES |
| 2 | SNFH | YES |
| 3 | ET | YES |
| 4 | RC | YES |

**CRITICAL:** Documentation references "BraTS 2021" labels (WT, TC, ET) but the model uses BraTS 2024 labels (NETC, SNFH, ET, RC). These are DIFFERENT label schemas.

---

## Segmentation Evaluation

### Metrics Implemented

| Metric | Region | Status |
|--------|--------|--------|
| Dice | WT, TC, ET | IMPLEMENTED |
| Hausdorff 95% | WT, TC, ET | IMPLEMENTED |

### Reported Results

| Region | Dice | Hausdorff 95% |
|--------|------|---------------|
| Whole Tumor (WT) | 0.9005 | 4.69 mm |
| Tumor Core (TC) | 0.5803 | 34.20 mm |
| Enhancing Tumor (ET) | 0.6010 | 16.33 mm |

### Segmentation Concerns

| Issue | Severity | Description |
|-------|----------|-------------|
| Model undertrained (30/100 epochs) | HIGH | Performance likely to improve with more training |
| TC and ET Dice are low | HIGH | 0.58 and 0.60 are below clinical utility threshold |
| Hausdorff for TC is very high (34.2mm) | HIGH | Large segmentation errors |
| No IoU reported | MEDIUM | Standard metric missing |
| No Sensitivity/Specificity | MEDIUM | Standard medical imaging metrics missing |
| No per-case statistics | MEDIUM | Only mean, no median/SD/range |
| No ASSD | LOW | Average symmetric surface distance |
| No surface Dice | LOW | Robust to outliers |
| Label schema mismatch | HIGH | Documentation says BraTS 2021, code uses BraTS 2024 |

---

## Segmentation-Classification Integration

### Current Integration

The proposed model (TumorAwareMultiTaskNet) has a **2D segmentation head** that predicts tumor regions from 2D feature maps. This is **NOT connected** to the 3D U-Net.

| Component | 3D U-Net | TumorAware Seg Head |
|-----------|----------|-------------------|
| Dimensionality | 3D | 2D |
| Input | 4-channel volume | 2D feature map |
| Output | 4-class volume | 4-class 2D map |
| Training | BraTS patches | End-to-end with classification |
| Weights | Separate checkpoint | Shared with classifier |

### Integration Gap

The 3D U-Net segmentation and the 2D classification operate in **different domains**. The proposed model's segmentation head learns a 2D approximation of tumor regions, which is fundamentally different from the 3D volumetric segmentation.

**This is a significant architectural limitation for publication.**

---

## Required Segmentation Improvements

1. **Complete training** — Train for full 100 epochs
2. **Add metrics** — IoU, Sensitivity, Specificity, ASSD
3. **Per-case statistics** — Median, SD, range, 95% CI
4. **Correct label schema** — Ensure documentation matches code
5. **Bridge 2D-3D gap** — Connect 3D U-Net to 2D classification
6. **Multi-seed evaluation** — Statistical validation

---

## Segmentation Audit Summary

| Criterion | Status |
|-----------|--------|
| 3D U-Net implementation | IMPLEMENTED |
| Dice evaluation | IMPLEMENTED |
| Hausdorff evaluation | IMPLEMENTED |
| Training completion | INCOMPLETE (30/100) |
| Additional metrics | NOT IMPLEMENTED |
| Per-case statistics | NOT IMPLEMENTED |
| 2D-3D integration | NOT IMPLEMENTED |
| Label schema consistency | INCONSISTENT |
