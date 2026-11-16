# 20 MISSING EXPERIMENTS

## P0 — Mandatory Before Publication

### P0.1: Dataset Integrity Verification

| Experiment | Description | Status |
|-----------|-------------|--------|
| Patient ID extraction for classification | Determine if patient IDs can be obtained | NOT PERFORMED |
| Duplicate detection | Perceptual hash comparison across splits | NOT PERFORMED |
| Near-duplicate detection | Image similarity analysis | NOT PERFORMED |
| External contamination check | Hash comparison between internal and external | NOT PERFORMED |

### P0.2: Patient-Level Data Split

| Experiment | Description | Status |
|-----------|-------------|--------|
| Patient-level split for classification | If patient IDs available, implement patient-level split | NOT POSSIBLE (no patient IDs) |
| Verify no patient overlap | Ensure no patient appears in train and test | NOT VERIFIED |

### P0.3: Methodology Clarification

| Experiment | Description | Status |
|-----------|-------------|--------|
| 2D vs 3D clarification | Document exactly what the system does | NOT DONE |
| Validation split reconciliation | Fix 0.2 vs 0.15 inconsistency | NOT DONE |
| Dataset count reconciliation | Fix documentation errors | NOT DONE |

### P0.4: Baseline Comparison

| Experiment | Description | Status |
|-----------|-------------|--------|
| Simple CNN baseline | Train and evaluate | EXISTS (CNN-2D) |
| ResNet50 baseline | Train and evaluate | EXISTS |
| DenseNet121 baseline | Train and evaluate | EXISTS |
| EfficientNet baseline | Train and evaluate | EXISTS |
| Transformer baseline | Train and evaluate | EXISTS (Swin, ViT) |
| Published method comparison | Compare with recent papers | NOT DONE |

### P0.5: Novel Method Implementation

| Experiment | Description | Status |
|-----------|-------------|--------|
| Implement proposed method | ATGFM + consistency + uncertainty weights | NOT DONE |
| Sanity check training | Verify model trains and converges | NOT DONE |
| Verify gradient flow | Check gradients propagate correctly | NOT DONE |

### P0.6: Ablation Study

| Experiment | Description | Status |
|-----------|-------------|--------|
| Baseline only | Classification without tumor guidance | EXISTS |
| + Tumor guidance | Add ATGFM module | NOT DONE |
| + Uncertainty | Add uncertainty-modulated fusion | NOT DONE |
| + Consistency | Add cross-task consistency | NOT DONE |
| + Learned weights | Add uncertainty-weighted loss | NOT DONE |
| Full model | All components combined | NOT DONE |

### P0.7: Multi-Seed Validation

| Experiment | Description | Status |
|-----------|-------------|--------|
| 5-seed training | Train with 5 different seeds | NOT DONE |
| Mean ± SD reporting | Report statistics across seeds | NOT DONE |
| 95% confidence intervals | Compute bootstrap CI | NOT DONE |
| Statistical significance | McNemar's test for comparisons | NOT DONE |

### P0.8: External Validation Protocol

| Experiment | Description | Status |
|-----------|-------------|--------|
| No-retraining protocol | Ensure no tuning on external data | NOT VERIFIED |
| Per-class external metrics | Report per-class F1, precision, recall | NOT DONE |
| External calibration | Evaluate calibration on external data | NOT DONE |
| External uncertainty | Evaluate uncertainty on external data | NOT DONE |

---

## P1 — Strongly Recommended

### P1.1: Calibration Analysis

| Experiment | Description | Status |
|-----------|-------------|--------|
| Per-model ECE comparison | Compare calibration across models | NOT SYSTEMATIC |
| Temperature scaling comparison | Compare native vs temperature-scaled | NOT DONE |
| Calibration-utility tradeoff | Plot accuracy vs ECE | NOT DONE |
| External calibration shift | Measure calibration degradation | NOT DONE |

### P1.2: Uncertainty Analysis

| Experiment | Description | Status |
|-----------|-------------|--------|
| Uncertainty-error correlation | Does uncertainty predict errors? | NOT DONE |
| AUROC for error detection | Quantify uncertainty utility | NOT DONE |
| Risk-coverage curve | Selective prediction analysis | NOT DONE |
| Selective accuracy | Accuracy at different coverage levels | NOT DONE |

### P1.3: Robustness Analysis

| Experiment | Description | Status |
|-----------|-------------|--------|
| MRI-specific artifacts | Bias field, Gibbs ringing, motion | NOT DONE |
| Severity curves | Accuracy vs severity level | NOT DONE |
| Per-corruption analysis | Which artifacts are most problematic | NOT DONE |
| Robustness-calibration interaction | Does calibration degrade under corruption | NOT DONE |

### P1.4: Segmentation Analysis

| Experiment | Description | Status |
|-----------|-------------|--------|
| Complete training | Train for full 100 epochs | NOT DONE |
| Additional metrics | IoU, Sensitivity, Specificity | NOT DONE |
| Per-case statistics | Median, SD, range | NOT DONE |
| HD95 per-case | Distribution of Hausdorff distances | NOT DONE |

### P1.5: Explainability Analysis

| Experiment | Description | Status |
|-----------|-------------|--------|
| Quantitative localization | IoU with tumor masks | NOT DONE |
| Pointing game accuracy | Is max activation inside tumor? | NOT DONE |
| Correct vs incorrect comparison | Does explainability differ? | NOT DONE |
| Multi-model comparison | Which model localizes better? | NOT DONE |

### P1.6: Failure Analysis

| Experiment | Description | Status |
|-----------|-------------|--------|
| False positive analysis | Why does model predict tumor when none exists? | NOT DONE |
| False negative analysis | Why does model miss tumors? | NOT DONE |
| High-confidence errors | Why is model confident but wrong? | NOT DONE |
| Low-confidence correct | Why is model uncertain but right? | NOT DONE |

---

## P2 — Optional

### P2.1: Additional Methods

| Experiment | Description | Status |
|-----------|-------------|--------|
| Additional SSL method | BYOL, MoCo, etc. | NOT DONE |
| Knowledge distillation | Train smaller student model | NOT DONE |
| Test-time augmentation | Aggregate predictions | NOT DONE |
| Ensemble methods | Combine multiple models | NOT DONE |

### P2.2: Advanced Analysis

| Experiment | Description | Status |
|-----------|-------------|--------|
| t-SNE visualization | Embedding space analysis | NOT DONE |
| UMAP visualization | Manifold analysis | NOT DONE |
| Domain shift matrix | Cross-dataset performance | NOT DONE |
| Feature distribution analysis | KL divergence between domains | NOT DONE |

---

## Experiment Priority Summary

| Priority | Count | Description |
|----------|-------|-------------|
| P0 | 20+ | Mandatory for publication |
| P1 | 15+ | Strongly recommended |
| P2 | 8+ | Optional enhancements |
| **Total** | **43+** | **Comprehensive experiment suite** |
