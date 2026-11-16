# 03 DATA LEAKAGE AUDIT

## Classification Dataset Leakage Analysis

### Split Architecture

```
data/raw/classification/
├── train/        (9,291 images — used for training + internal validation split)
│   ├── glioma/      (3,018)
│   ├── meningioma/  (2,183)
│   ├── pituitary/   (2,145)
│   └── no tumor/    (1,945)
└── test/         (1,857 images — NEVER used during training)
    ├── glioma/      (603)
    ├── meningioma/  (436)
    ├── pituitary/   (429)
    └── no tumor/    (389)
```

**Training data further split:**
- Train split: 80% of training data (stratified by class)
- Validation split: 20% of training data (stratified by class)

### Leakage Risk Assessment

#### 1. Train/Test Overlap
- **Risk:** LOW
- **Reasoning:** The train/ and test/ directories are pre-defined and non-overlapping. The code in `datasets/classification.py:discover_classification_samples()` reads from separate directories and assigns `split="train"` or `split="test"` based on directory.
- **Verification needed:** File-level hash comparison to confirm no identical images exist across directories.

#### 2. Patient-Level Leakage
- **Risk:** **HIGH — CANNOT VERIFY**
- **Reasoning:** Filenames do not contain patient IDs. The images are named by numeric IDs (e.g., `1049.jpg`) that do not encode patient information. Without patient IDs, it is impossible to determine whether the same patient appears in both train and test splits.
- **Impact:** If the same patient has multiple MRI slices, some slices could be in train while others are in test, creating information leakage.
- **Mitigation:** Requires external dataset metadata or manual inspection.

#### 3. Validation Split Leakage
- **Risk:** LOW
- **Reasoning:** The validation split is created by `split_classification_samples()` which uses stratified per-class splitting with a fixed seed. The split is performed on the training data only, and the test set is never touched.
- **Code location:** `datasets/classification.py:200-241`

#### 4. Near-Duplicate Images
- **Risk:** **UNKNOWN — NOT TESTED**
- **Reasoning:** No perceptual hash analysis, no image similarity comparison, no duplicate detection has been performed.
- **Impact:** Brain MRI datasets often contain near-duplicate slices from the same volume. If adjacent slices are split across train/test, this constitutes subtle leakage.

#### 5. Augmented Image Leakage
- **Risk:** LOW
- **Reasoning:** Augmentation is applied on-the-fly during training (in `create_train_transform()`). No augmented images are saved to disk. The same original image cannot appear differently in train vs test.
- **Code location:** `training/classification_trainer.py:276-299`

#### 6. External Dataset Contamination
- **Risk:** **UNKNOWN**
- **Reasoning:** The BRISC 2025 and Kaggle datasets are described as external test sets. However:
  - No verification that external images are not duplicates of internal images
  - No verification that external images were not used during any hyperparameter tuning
  - The external validation accuracy (99.2%) exceeds internal accuracy (97-98%), which is suspicious

### Quantitative Leakage Indicators

| Indicator | Value | Concern Level |
|-----------|-------|---------------|
| External accuracy > Internal accuracy | 99.2% vs 97-98% | **HIGH** |
| All ROC-AUC = 1.0000 | Perfect separation | **MEDIUM** |
| No patient ID checking | — | **HIGH** |
| No duplicate detection | — | **MEDIUM** |
| Pre-defined split (no user control) | — | **LOW** |

### Why External > Internal?

The BRISC 2025 external validation reports **higher** accuracy (99.2%) than internal test set (97-98%). Possible explanations:

1. **Dataset contamination** — BRISC images may overlap with training data
2. **Easier dataset** — BRISC may have clearer, less ambiguous cases
3. **Class distribution** — BRISC may have a more favorable class balance
4. **Image quality** — BRISC may have higher-quality images
5. **Preprocessing advantage** — External preprocessing may accidentally help
6. **Selection bias** — External test set may not represent the true difficulty

**This requires immediate investigation before any publication claim.**

---

## BraTS Segmentation Leakage Analysis

### Split Architecture
- Patient-level split via `split_brats_cases_by_patient()`
- Patient ID extracted from folder names: `BraTS-GLI-02062-100` → patient ID `BraTS-GLI-02062`
- Split manifest saved to JSON for reproducibility

### Leakage Risk Assessment

| Risk | Assessment |
|------|------------|
| Patient-level split | **VERIFIED — IMPLEMENTED** |
| Cross-split patient overlap | **NONE** (enforced by code) |
| Case-level leakage | **LOW** (manifest-based) |

**BraTS segmentation leakage risk: LOW**

---

## Overall Leakage Status

```
LEAKAGE DETECTED: UNKNOWN (classification) / NO (BraTS segmentation)

Classification:
- Patient-level leakage: CANNOT VERIFY (no patient IDs)
- Near-duplicate leakage: NOT TESTED
- External contamination: NOT TESTED
- Train/test overlap: LOW RISK (separate directories)

BraTS Segmentation:
- Patient-level split: VERIFIED
- Leakage: LOW RISK

RECOMMENDATION: Perform perceptual hash analysis and file-level
duplicate detection on classification dataset before publication.
```
