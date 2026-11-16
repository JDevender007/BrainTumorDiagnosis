# 09 EXTERNAL VALIDATION AUDIT

## External Datasets

### BRISC 2025

| Attribute | Value |
|-----------|-------|
| Location | data/external/brisc2025/ |
| Count | 1,000 images (claimed) |
| Classes | Glioma, Meningioma, Pituitary, No Tumor |
| Source | BRISC 2025 challenge |
| Protocol | NO RETRAINING — test-time only |

### Kaggle Brain Tumor

| Attribute | Value |
|-----------|-------|
| Location | data/external/kaggle_brain_tumor/ |
| Count | 1,311 images (claimed) |
| Source | Zenodo (zenodo.org/records/12735702) |
| Protocol | NO RETRAINING — test-time only |

---

## External Validation Results

### BRISC 2025 (from README.md)

| Model | Internal Acc | External Acc | Δ Accuracy |
|-------|-------------|-------------|------------|
| Swin-T | 97.09% | **99.20%** | **+2.11%** |
| DenseNet121 | 97.95% | **98.40%** | **+0.45%** |
| ResNet50 | 97.04% | **98.00%** | **+0.96%** |
| EfficientNetV2-S | 98.06% | **97.90%** | -0.16% |
| ViT-B/16 | 97.52% | **97.80%** | **+0.28%** |
| ConvNeXt-Tiny | 97.42% | **97.50%** | **+0.08%** |
| CNN-2D | 95.26% | **96.40%** | **+1.14%** |

### Critical Finding: External > Internal

**6 out of 7 models show HIGHER accuracy on external data than internal test set.**

This is **anomalous** and requires investigation.

---

## Investigation: Why External > Internal?

### Hypothesis 1: Dataset Contamination
- **Possibility:** BRISC images may overlap with training data
- **Test:** Hash comparison between BRISC images and training images
- **Status:** NOT PERFORMED

### Hypothesis 2: Easier Dataset
- **Possibility:** BRISC may have clearer, less ambiguous cases
- **Test:** Compare image quality, class balance, annotation quality
- **Status:** NOT PERFORMED

### Hypothesis 3: Class Distribution Difference
- **Possibility:** BRISC may have more balanced or easier classes
- **Test:** Compare per-class distributions
- **Status:** NOT PERFORMED

### Hypothesis 4: Preprocessing Advantage
- **Possibility:** BRISC preprocessing may accidentally help
- **Test:** Compare preprocessing pipelines
- **Status:** NOT PERFORMED

### Hypothesis 5: Domain Shift in Opposite Direction
- **Possibility:** Internal test set may be harder than external
- **Test:** Compare difficulty metrics
- **Status:** NOT PERFORMED

### Hypothesis 6: Image Quality Difference
- **Possibility:** BRISC images may be higher quality
- **Test:** Compare resolution, noise, artifacts
- **Status:** NOT PERFORMED

---

## External Validation Concerns

| Issue | Severity | Description |
|-------|----------|-------------|
| External > Internal accuracy | **CRITICAL** | Requires immediate investigation |
| No duplicate detection | HIGH | May be dataset contamination |
| No per-class external analysis | MEDIUM | Only aggregate metrics |
| No calibration on external data | MEDIUM | Calibration only on internal |
| No uncertainty on external data | MEDIUM | Uncertainty only on internal |
| No robustness on external data | MEDIUM | Robustness only on internal |
| No domain shift quantification | MEDIUM | No feature distribution analysis |
| No external dataset documentation | LOW | Source, license, version unknown |

---

## Required Actions

1. **Hash comparison** — Compare BRISC images against training images
2. **Per-class analysis** — Report per-class metrics on external data
3. **Calibration on external** — Evaluate calibration shift
4. **Uncertainty on external** — Evaluate uncertainty shift
5. **Domain shift analysis** — Feature distribution comparison
6. **Document external datasets** — Source, license, version, acquisition

---

## External Validation Status

```
External validation: EXISTS but SUSPICIOUS
- External > Internal accuracy: UNEXPLAINED
- Dataset contamination: NOT TESTED
- Domain shift quantification: NOT PERFORMED
- Per-class analysis: NOT PERFORMED
- Calibration/uncertainty on external: NOT PERFORMED
```
