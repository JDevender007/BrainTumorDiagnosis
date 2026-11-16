# TURMA-Net Architecture Audit

**Date:** 2026-09-04
**Auditor:** Automated inspection

---

## 1. Architecture Overview

TURMA-Net (Tumor-guided Uncertainty-aware Robust Multi-task Network) is a 2D multi-task architecture for brain tumor MRI classification.

**Encoder:** DenseNet121 (pretrained on ImageNet)
**Input:** RGB MRI slices [B, 3, 224, 224]
**Output:** Classification logits [B, 4] + segmentation logits [B, 4, H, W]

---

## 2. Component Classification

### EXISTING / STANDARD Components

| Component | Origin | Status |
|-----------|--------|--------|
| DenseNet121 backbone | torchvision (ImageNet pretrained) | Standard |
| Global average pooling | Standard CNN technique | Standard |
| Batch normalization | Ioffe & Szegedy 2015 | Standard |
| Spatial attention (avg+max pooling + conv) | CBAM (Woo et al. 2018) | Standard |
| Channel attention (SE-style squeeze-excitation) | SENet (Hu et al. 2018) | Standard |
| Tumor region pooling (attention-weighted) | Spatial attention + pooling | Standard |
| Segmentation head (conv decoder + bilinear upsampling) | Standard U-Net decoder | Standard |
| MC Dropout uncertainty estimation | Gal & Ghahramani 2016 | Standard |
| Focal Loss | Lin et al. 2017 | Standard |
| Dice Loss | Milletari et al. 2016 | Standard |
| Cross-entropy loss | Standard | Standard |
| Label smoothing | Szegedy et al. 2016 | Standard |
| AdamW optimizer | Loshchilov & Hutter 2019 | Standard |
| Cosine annealing LR | Loshchilov & Hutter 2017 | Standard |
| Mixed precision training (AMP) | Standard PyTorch | Standard |
| Gradient accumulation | Standard technique | Standard |
| Gradient clipping | Standard technique | Standard |

### PROPOSED / NOVEL Components

| Component | Novelty Claim | Assessment |
|-----------|---------------|------------|
| **ATGFM** (Adaptive Tumor-Guided Feature Modulation) | Uncertainty-conditioned feature fusion with learned gamma/beta/alpha parameters | **Genuinely novel** - combines uncertainty estimation with adaptive feature modulation in a way not found in standard attention mechanisms |
| **TumorLocalizationUncertainty** | Estimates localization uncertainty from segmentation entropy | **Novel application** - entropy-based uncertainty from segmentation is standard, but using it to condition feature fusion is new |
| **CrossTaskConsistency** | KL divergence between classification and segmentation predictions | **Standard technique** - knowledge distillation / consistency regularization (Hinton et al. 2015, Feng et al. 2019) |
| **UncertaintyWeightedMultiTaskLoss** | Learned task weights via log-variance parameterization | **NOT novel** - directly from Kendall et al. 2018 "Multi-Task Learning Using Uncertainty to Weigh Losses" |

---

## 3. Novel Component Analysis

### 3.1 ATGFM (Adaptive Tumor-Guided Feature Modulation)

**Mathematical formulation:**
```
F_global = W_g @ global_feat + b_g
F_local = W_l @ tumor_feat + b_l

combined = [F_global; F_local; U_tumor]

gamma = sigmoid(W_gamma @ combined + b_gamma)  # mixing ratio
beta = tanh(W_beta @ combined + b_beta)        # shift
alpha = sigmoid(W_alpha @ combined + b_alpha)  # gate

F_modulated = alpha * (gamma * F_global + (1-gamma) * F_local) + beta
```

**Why this is novel:**
1. Standard attention fusion (TumorAwareMultiTaskNet) uses a fixed learned attention weight that doesn't depend on uncertainty
2. ATGFM conditions the fusion on the *uncertainty* of tumor localization
3. When uncertainty is high (poor localization), the model learns to rely more on global features
4. When uncertainty is low (good localization), the model can trust tumor-localized features
5. The modulation parameters (gamma, beta, alpha) are all functions of the uncertainty, not just the features

**Comparison with TumorAwareMultiTaskNet:**
- TumorAwareMultiTaskNet: `fused = attn * global + (1-attn) * tumor` where `attn = softmax(W @ [global; tumor])`
- TURMA-Net: `fused = alpha * (gamma * global + (1-gamma) * tumor) + beta` where `gamma, alpha, beta = f(global, tumor, uncertainty)`

**Is this genuinely better?** Unknown - requires experimental validation.

### 3.2 CrossTaskConsistency

**Mathematical formulation:**
```
P_cls = softmax(cls_logits / temperature)
P_seg = softmax(seg_logits / temperature).mean(dim=(2,3))

L_cons = (KL(P_cls || P_seg) + KL(P_seg || P_cls)) / 2
```

**Assessment:** This is standard knowledge distillation / consistency regularization. The temperature-scaled KL divergence between classification and segmentation predictions is a well-known technique. Not novel.

### 3.3 UncertaintyWeightedMultiTaskLoss

**Mathematical formulation:**
```
L_total = (1/2σ₁²)L_cls + (1/2σ₂²)L_seg + (1/2σ₃²)L_cons + log(σ₁) + log(σ₂) + log(σ₃)
```

**Assessment:** This is directly from Kendall et al. 2018 "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics". Not novel.

---

## 4. What TURMA-Net Actually Introduces

**Genuinely new:**
1. ATGFM - uncertainty-conditioned feature modulation (the specific architecture of using uncertainty to gate/shift features)

**New combination (not individually novel):**
2. Combining ATGFM with multi-task learning (classification + segmentation)
3. Using segmentation uncertainty to condition feature fusion
4. The specific architecture design for brain tumor classification

**NOT new:**
5. Cross-task consistency loss (standard)
6. Uncertainty-weighted loss (Kendall et al. 2018)
7. MC Dropout uncertainty (Gal & Ghahramani 2016)
8. Spatial/channel attention (CBAM, SENet)
9. Segmentation head (standard decoder)
10. Focal loss, Dice loss (standard)

---

## 5. Honest Assessment

The core novel contribution is ATGFM. Everything else is either standard or borrowed from existing work. The question is:

**Does ATGFM actually improve performance over standard attention fusion?**

This can only be answered by the ablation experiment comparing:
- TumorAwareMultiTaskNet (standard attention fusion)
- TURMA-Net with ATGFM (uncertainty-conditioned fusion)
- TURMA-Net without ATGFM (should degrade to similar to TumorAwareMultiTaskNet)

If ATGFM does not improve performance, the novelty claim is weak.

---

## 6. Key Experimental Questions

1. Does ATGFM improve classification accuracy over standard attention fusion?
2. Does the uncertainty mechanism actually separate correct from incorrect predictions?
3. Does the consistency loss help or hurt classification?
4. Does the uncertainty-weighted loss help or hurt training?
5. Is the improvement statistically significant across multiple seeds?
6. Does the improvement justify the additional parameters and computation?

---

*Audit completed: 2026-09-04*
