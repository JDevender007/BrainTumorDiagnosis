# MASTER AUDIT — BrainTumorDiagnosis Publication Upgrade

**Audit Date:** 2026-09-04
**Auditor Role:** Senior Medical Imaging Research Scientist / IEEE TMI Reviewer
**Project Root:** D:\BrainTumorDiagnosis

---

## EXECUTIVE SUMMARY

This document records the complete independent audit of the BrainTumorDiagnosis project before any publication-upgrade modifications. The audit covers dataset integrity, data leakage, architecture verification, novelty assessment, training methodology, evaluation rigor, and publication readiness.

---

## 1. PROJECT OVERVIEW

| Attribute | Value | Verification Status |
|-----------|-------|-------------------|
| Project name | BrainTumorDiagnosis | VERIFIED |
| Total Python files | ~88 | VERIFIED |
| Total lines of code | ~15,000+ | VERIFIED |
| Test count claimed | 73 | PARTIALLY VERIFIED (tests exist, execution not verified in this audit) |
| Primary framework | PyTorch | VERIFIED |
| Target hardware | RTX 3050 Laptop (4GB VRAM) | VERIFIED |

---

## 2. CRITICAL FINDINGS SUMMARY

### SEVERITY: HIGH

| # | Finding | Impact | Status |
|---|---------|--------|--------|
| H1 | Classification is 2D slice-based, NOT 3D volumetric | Methodology misrepresentation risk | CONFIRMED |
| H2 | Documentation states "7,897 train images" but actual count is 9,291 | Dataset integrity concern | CONFIRMED |
| H3 | Validation split ratio discrepancy: code=20%, docs=15% | Training protocol inconsistency | CONFIRMED |
| H4 | Segmentation metadata shows epoch 30 / Dice 0.5243, docs claim 0.5669 | Checkpoint integrity concern | CONFIRMED |
| H5 | Random seeds inconsistent across documentation files [42,123,456] vs [42,123,2024] | Reproducibility concern | CONFIRMED |
| H6 | No patient-level identity checking for classification dataset | Potential data leakage | UNKNOWN (requires file-level inspection) |
| H7 | External validation accuracy (99.2%) exceeds internal (97-98%) | Requires investigation | CONFIRMED |

### SEVERITY: MEDIUM

| # | Finding | Impact | Status |
|---|---------|--------|--------|
| M1 | Proposed model TumorAwareMultiTaskNet uses 2D encoder only | Architecture limitation | CONFIRMED |
| M2 | Segmentation is 3D U-Net but classification is 2D — different domains | Cross-domain integration gap | CONFIRMED |
| M3 | README has duplicate Swin-T row with different FPS values | Documentation error | CONFIRMED |
| M4 | paper.txt claims "15,947 total images" but actual sum is ~12,065 | Documentation error | CONFIRMED |
| M5 | No perceptual hash or duplicate detection for classification images | Near-duplicate risk | NOT PERFORMED |
| M6 | All ROC-AUC values reported as 1.0000 — suspicious ceiling effect | Metric reliability concern | CONFIRMED |

### SEVERITY: LOW

| # | Finding | Impact | Status |
|---|---------|--------|--------|
| L1 | Proposed model epochs: config=50, docs say "reduced to 30" | Config inconsistency | CONFIRMED |
| L2 | CBAM3D exists in attention.py but is not used by any training pipeline | Unused component | CONFIRMED |
| L3 | BrainTumorClassifier3D (3D CNN) exists but is not trained/evaluated | Unused model | CONFIRMED |

---

## 3. 2D VS 3D METHODOLOGY CLARIFICATION

### Classification

| Aspect | Actual | Documentation Claim |
|--------|--------|-------------------|
| Input dimensionality | **2D** (single RGB slice) | Ambiguous — implies 3D in some places |
| Input tensor shape | [B, 3, 224, 224] | Correct in code |
| Volume handling | **NONE** — each slice classified independently | Not clearly stated |
| Slice aggregation | **NONE** — no patient-level aggregation | Not clearly stated |
| Voxel spacing | N/A (2D images) | N/A |
| Orientation | N/A (2D images) | N/A |

**VERDICT:** Classification is **2D slice-based**. The system does NOT perform 3D volumetric classification. Each 2D MRI slice is classified independently. There is no mechanism to aggregate predictions across slices from the same patient.

### Segmentation

| Aspect | Actual | Documentation Claim |
|--------|--------|-------------------|
| Input dimensionality | **3D** (volumetric patches) | Correct |
| Input tensor shape | [B, 4, 96, 96, 96] | Correct |
| Volume handling | Patch-based with sliding window | Correct |
| Modality | 4-channel (t1n, t1c, t2w, t2f) | Correct |
| Output channels | 4 (BraTS 2024 labels) | Correct |
| Voxel spacing | Not explicitly resampled | Not documented |

**VERDICT:** Segmentation is **3D volumetric** using a 3D U-Net on BraTS patches.

### Proposed Model (TumorAwareMultiTaskNet)

| Aspect | Actual |
|--------|--------|
| Encoder | 2D torchvision backbone (DenseNet121 default) |
| Feature extraction | 2D feature maps via forward hook |
| Segmentation head | 2D lightweight decoder (NOT 3D) |
| Tumor region pooling | 2D attention-weighted pooling |
| Classification | 2D MLP on fused features |
| Uncertainty | MC Dropout on 2D features |

**VERDICT:** The proposed model is **entirely 2D**. Its "segmentation head" predicts a 2D tumor mask from a 2D feature map — this is fundamentally different from the 3D U-Net segmentation used for BraTS volumes.

---

## 4. DATASET VERIFICATION

### 4.1 Internal Classification Dataset

| Attribute | README.md | generate_paper.py | classification.yaml | ACTUAL |
|-----------|-----------|-------------------|--------------------|----|
| Train glioma | 3,018 | 3,018 | — | **3,018** |
| Train meningioma | 2,183 | 2,183 | — | **2,183** |
| Train pituitary | 2,145 | 2,145 | — | **2,145** |
| Train no_tumor | 1,945 | 1,945 | — | **1,945** |
| **Train total** | **9,291** | **9,291** | — | **9,291** |
| Test glioma | 603 | 603 | — | **603** |
| Test meningioma | 436 | 436 | — | **436** |
| Test pituitary | 429 | 429 | — | **429** |
| Test no_tumor | 389 | 389 | — | **389** |
| **Test total** | **1,857** | **1,857** | — | **1,857** |
| **Grand total** | **11,148** | **11,148** | — | **11,148** |

**paper.txt claims:**
- "Train split: 7,897 images" — **INCORRECT** (actual: 9,291)
- "Total: 9,754" — **INCORRECT** (actual: 11,148)
- "Total images available: 15,947" — **INCORRECT** (actual: ~12,065)

**Braintumor.txt claims:**
- "Train: 7,897 images" — **INCORRECT** (actual: 9,291)
- "Total: 11,148 images" — **CONSISTENT** with actual (but inconsistent with its own "7,897" claim)

### 4.2 External Datasets

| Dataset | Claimed Count | Verified | Source |
|---------|--------------|----------|--------|
| BRISC 2025 | 1,000 images | PARTIALLY VERIFIED | External directory exists |
| Kaggle Brain Tumor | 1,311 images | PARTIALLY VERIFIED | External directory exists |
| BraTS 2024 | 200 cases (129/26/45 split) | PARTIALLY VERIFIED | NIfTI volumes present |

### 4.3 Validation Split

| Source | Claim | Actual Code |
|--------|-------|-------------|
| classification.yaml | `validation_split: 0.2` | **0.2 (80/20)** |
| dataset.yaml | `validation_ratio: 0.15` | **0.15 (85/15)** |
| paper.txt | "85/15 split" | **CONTRADICTED** by classification.yaml |
| Braintumor.txt | "1,394 validation" | 1,394 = 15% of 9,291 → matches dataset.yaml |

**CRITICAL:** Two different validation ratios exist in the codebase. The classification trainer uses 0.2 (80/20), while dataset.yaml specifies 0.15 (85/15). The documentation appears to reference the 0.15 ratio.

---

## 5. DATA LEAKAGE ASSESSMENT

### 5.1 Classification Data Split

The classification dataset has a **pre-defined** train/test split (separate directories). The trainer further splits the training data into train/validation using `split_classification_samples()` with stratification by class.

| Leakage Risk | Assessment |
|-------------|------------|
| Train/test overlap | **LOW** — separate directories, but no patient ID check |
| Patient-level leakage | **UNKNOWN** — no patient IDs in filenames, cannot verify |
| Near-duplicate images | **UNKNOWN** — no perceptual hash analysis performed |
| Augmented image leakage | **LOW** — augmentation is applied on-the-fly only |

### 5.2 BraTS Segmentation Data Split

The BraTS dataset uses patient-level splitting via `split_brats_cases_by_patient()`, which groups cases by patient ID (extracted from case folder names like `BraTS-GLI-02062-100`).

| Leakage Risk | Assessment |
|-------------|------------|
| Patient-level split | **YES** — implemented correctly via patient ID extraction |
| Case-level split | **YES** — manifest-based, reproducible |
| Patient overlap across splits | **NONE** — verified by code design |

### 5.3 Overall Leakage Status

**Leakage detected: UNKNOWN for classification (patient IDs unavailable), NO for BraTS segmentation.**

---

## 6. EXISTING COMPONENT INVENTORY

### 6.1 Models

| Model | Type | Parameters | Status |
|-------|------|-----------|--------|
| CNN-2D (residual) | 2D classifier | 21.40M | TRAINED + EVALUATED |
| DenseNet121 | 2D classifier | 6.96M | TRAINED + EVALUATED |
| ResNet50 | 2D classifier | 23.52M | TRAINED + EVALUATED |
| EfficientNetV2-S | 2D classifier | 20.18M | TRAINED + EVALUATED |
| ConvNeXt-Tiny | 2D classifier | 27.82M | TRAINED + EVALUATED |
| Swin Transformer-T | 2D classifier | 27.52M | TRAINED + EVALUATED |
| ViT-B/16 | 2D classifier | 85.80M | TRAINED + EVALUATED |
| 3D U-Net | 3D segmentor | — | TRAINED (partial, epoch 30/100) |
| TumorAwareMultiTaskNet | 2D multi-task | 9.03M | TRAINED + EVALUATED |
| BrainTumorClassifier3D | 3D classifier | — | NOT TRAINED |
| CBAM3D | 3D attention | — | NOT TRAINED (module only) |
| SimCLR | SSL encoder | — | EXISTS (not verified if trained) |

### 6.2 Evaluation Components

| Component | Status |
|-----------|--------|
| Classification evaluator (11 metrics) | IMPLEMENTED |
| Comprehensive metrics (30+ metrics) | IMPLEMENTED |
| Robustness (9 corruption types, MRRS) | IMPLEMENTED |
| Calibration (ECE, MCE, Brier, NLL) | IMPLEMENTED |
| Uncertainty (MC Dropout) | IMPLEMENTED |
| Explainability (Grad-CAM) | IMPLEMENTED |
| Cross-dataset analysis | IMPLEMENTED |
| Statistical validation | EXISTS (not verified) |
| Efficiency analysis | EXISTS (not verified) |
| Ablation runner | EXISTS (not verified) |
| Segmentation evaluator | IMPLEMENTED |

### 6.3 Training Components

| Component | Status |
|-----------|--------|
| Unified classification trainer | IMPLEMENTED |
| Segmentation trainer (3D U-Net) | IMPLEMENTED |
| Self-supervised trainer (SimCLR) | IMPLEMENTED |
| Multi-task trainer | IMPLEMENTED |
| Training manager (stop/resume) | IMPLEMENTED |

---

## 7. DOCUMENTATION INCONSISTENCIES

| Issue | Source A | Source B | Resolution |
|-------|----------|----------|------------|
| Train image count | paper.txt: 7,897 | README: 9,291 | **README is correct** |
| Total image count | paper.txt: 9,754 | README: 11,148 | **README is correct** |
| Grand total | paper.txt: 15,947 | Actual: ~12,065 | **paper.txt is wrong** |
| Validation ratio | classification.yaml: 0.2 | dataset.yaml: 0.15 | **Must reconcile** |
| Validation count | paper.txt: 1,394 | 0.2×9291=1,858 | **Depends on which ratio** |
| Random seeds | paper.txt: [42,123,456] | Braintumor.txt: [42,123,2024] | **Must verify** |
| Segmentation Dice | README: 0.5669 | metadata.json: 0.5243 | **metadata.json is actual** |
| Proposed model epochs | config: 50 | docs: 30 | **Must verify** |
| Swin-T FPS | README line 30: 121.6 | README line 36: 126.1 | **Duplicate row** |

---

## 8. NOVELTY PRELIMINARY ASSESSMENT

### What Exists

| Technique | Present | Implementation Quality |
|-----------|---------|----------------------|
| Multi-backbone comparison | YES | GOOD |
| Attention mechanisms (CBAM) | YES | GOOD |
| Self-supervised learning (SimCLR) | YES | EXISTS |
| Multi-task learning | YES | EXISTS |
| Grad-CAM explainability | YES | GOOD |
| MC Dropout uncertainty | YES | EXISTS |
| Robustness evaluation | YES | GOOD |
| Calibration analysis | YES | EXISTS |
| External validation | YES | EXISTS |
| Tumor-guided feature fusion | YES | EXISTS (in TumorAwareMultiTaskNet) |

### What Is Missing for Novel Contribution

| Gap | Impact |
|-----|--------|
| No genuine novelty in combining existing techniques | Publication rejection risk |
| TumorAwareMultiTaskNet is a straightforward combination of existing components | Weak novelty |
| No novel loss function | Missing contribution |
| No novel training strategy | Missing contribution |
| No rigorous comparison with published methods | Missing baseline |
| No mathematical formulation of the method | Missing for publication |

---

## 9. OVERALL ASSESSMENT

| Dimension | Score (1-10) | Notes |
|-----------|-------------|-------|
| Code quality | 7 | Well-structured, consistent |
| Documentation accuracy | 4 | Multiple inconsistencies |
| Dataset integrity | 5 | Counts verified, but patient-level leakage unknown |
| Methodology rigor | 5 | 2D/3D ambiguity, validation split inconsistency |
| Novelty | 3 | Combination of existing techniques only |
| Experimental completeness | 6 | Many metrics, but missing key experiments |
| Statistical validation | 3 | Multi-seed exists but not rigorously analyzed |
| Reproducibility | 5 | Checkpoints exist, but seeds inconsistent |
| Publication readiness | 3 | Major gaps in novelty, rigor, and validation |

**OVERALL PUBLICATION READINESS: 35/90 = 38.9% → NOT PUBLICATION READY**

---

## 10. REQUIRED ACTIONS (PRIORITY ORDER)

### P0 — Must Fix Before Any Publication Claim

1. Reconcile documentation dataset counts (paper.txt is wrong)
2. Resolve validation split ratio inconsistency (0.2 vs 0.15)
3. Clarify 2D vs 3D methodology in all documentation
4. Verify or obtain patient IDs for classification dataset
5. Investigate why external validation accuracy exceeds internal
6. Audit all reported metrics against actual checkpoint outputs

### P1 — Required for Publication

7. Design genuinely novel method (not just combination)
8. Implement rigorous baseline comparison
9. Perform ablation study
10. Multi-seed statistical validation with confidence intervals
11. Proper calibration evaluation
12. Uncertainty error detection analysis
13. Failure analysis
14. Literature comparison table

### P2 — Recommended

15. MRI-specific robustness (bias field, motion, Gibbs ringing)
16. Explainability localization quantification
17. t-SNE/UMAP representation visualization
18. Domain generalization analysis
19. Computational efficiency comparison
20. Reproducibility manifest generation

---

*This audit will be updated as investigation proceeds.*
