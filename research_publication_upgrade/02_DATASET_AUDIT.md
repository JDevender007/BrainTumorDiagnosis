# 02 DATASET AUDIT

## Classification Dataset

### Source
- **Location:** `data/raw/classification/`
- **Source:** Unknown — no URL, license, or citation provided
- **Public/Private:** UNKNOWN — likely from Kaggle brain tumor MRI dataset
- **Version:** Unknown

### Counts (VERIFIED from code and README)

| Split | Glioma | Meningioma | Pituitary | No Tumor | Total |
|-------|--------|------------|-----------|----------|-------|
| Train | 3,018 | 2,183 | 2,145 | 1,945 | **9,291** |
| Test | 603 | 436 | 429 | 389 | **1,857** |
| **Total** | **3,621** | **2,619** | **2,574** | **2,334** | **11,148** |

### Class Distribution

| Class | Train % | Test % | Imbalance Ratio |
|-------|---------|--------|-----------------|
| Glioma | 32.5% | 32.5% | 1.0 |
| Meningioma | 23.5% | 23.5% | 1.0 |
| Pituitary | 23.1% | 23.1% | 1.0 |
| No Tumor | 20.9% | 20.9% | 1.0 |

**Note:** Class distribution is approximately balanced across train and test splits (stratified split confirmed).

### Image Properties
- **Format:** JPG/JPEG
- **Color:** RGB (3 channels)
- **Resolution:** Variable (resized to 224×224 for training)
- **Preprocessing:** ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Augmentation:** RandomHorizontalFlip, RandomRotation(10°), RandomAffine, ColorJitter

### Patient-Level Information
- **Patient IDs:** NOT available in filenames
- **Patient-level split:** NOT possible without patient IDs
- **Duplicate detection:** NOT performed
- **Near-duplicate detection:** NOT performed

### Critical Issues

| Issue | Severity | Status |
|-------|----------|--------|
| No patient ID information | HIGH | UNKNOWN — cannot verify patient-level leakage |
| No dataset source citation | MEDIUM | MISSING |
| No license information | MEDIUM | MISSING |
| No acquisition details | MEDIUM | MISSING |
| No voxel spacing / resolution metadata | LOW | MISSING |
| Pre-defined train/test split (no patient-level control) | HIGH | CONFIRMED |

---

## BraTS 2024 Segmentation Dataset

### Source
- **Location:** `data/raw/BraTS2024/`
- **Source:** BraTS 2024 Challenge (post-treatment)
- **Public/Private:** Challenge dataset (requires registration)
- **Version:** BraTS 2024

### Counts
- **Total cases:** ~200 (exact count requires filesystem inspection)
- **Split:** Patient-level 70/15/15 (train/val/test)
- **Split manifest:** `data/splits/brats/split_manifest.json`

### MRI Properties
- **Modalities:** T1n, T1c, T2w, T2FLAIR (4 channels)
- **Resolution:** Variable (resampled to isotropic)
- **Patch size:** 96×96×96
- **Sampling:** Foreground-biased random sampling

### Segmentation Labels (BraTS 2024)
| Label | Description |
|-------|-------------|
| 0 | Background |
| 1 | NETC (Net enhancing tumor core) |
| 2 | SNFH (Subependymal non-enhancing FLAIR hyperintensity) |
| 3 | ET (Enhancing tumor) |
| 4 | RC (Resection cavity) |

**CRITICAL NOTE:** These are BraTS 2024 labels, NOT the standard BraTS 2021/2023 labels (background, edema, enhancing, non-enhancing). The documentation incorrectly references "BraTS 2021" in some places.

### Patient-Level Split
- **Implementation:** `extract_patient_id()` from case folder names (e.g., `BraTS-GLI-02062-100`)
- **Verified:** Code correctly extracts patient ID and groups cases
- **Leakage risk:** LOW — patient-level split implemented

---

## External Datasets

### BRISC 2025
- **Location:** `data/external/brisc2025/`
- **Count:** 1,000 images (claimed)
- **Classes:** Glioma, Meningioma, Pituitary, No Tumor
- **Source:** BRISC 2025 challenge
- **License:** UNKNOWN
- **Verification:** Directory exists, but exact count not verified in this audit

### Kaggle Brain Tumor
- **Location:** `data/external/kaggle_brain_tumor/`
- **Count:** 1,311 images (claimed)
- **Source:** Zenodo (zenodo.org/records/12735702)
- **License:** UNKNOWN
- **Verification:** Directory exists, but exact count not verified in this audit

---

## Dataset Audit Summary

| Dataset | Verified | Patient IDs | Patient-Level Split | Leakage Risk |
|---------|----------|-------------|--------------------|-|
| Internal Classification | PARTIAL | NO | NO | UNKNOWN |
| BraTS 2024 | PARTIAL | YES | YES | LOW |
| BRISC 2025 | PARTIAL | UNKNOWN | N/A (test only) | LOW |
| Kaggle Brain Tumor | PARTIAL | UNKNOWN | N/A (test only) | LOW |
