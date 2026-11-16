# 07 EVALUATION AUDIT

## Classification Evaluation

### Metrics Computed

| Metric | Implementation | Status |
|--------|---------------|--------|
| Accuracy | sklearn accuracy_score | IMPLEMENTED |
| Precision (macro) | sklearn precision_score | IMPLEMENTED |
| Recall (macro) | sklearn recall_score | IMPLEMENTED |
| F1-Score (macro) | sklearn f1_score | IMPLEMENTED |
| Specificity | Custom (confusion matrix) | IMPLEMENTED |
| ROC-AUC (OVR) | sklearn roc_auc_score | IMPLEMENTED |
| PR-AUC | sklearn average_precision_score | IMPLEMENTED |
| MCC | sklearn matthews_corrcoef | IMPLEMENTED |
| Cohen's Kappa | sklearn cohen_kappa_score | IMPLEMENTED |
| Log Loss | sklearn log_loss | IMPLEMENTED |
| Per-class F1 | sklearn classification_report | IMPLEMENTED |
| Confusion Matrix | sklearn confusion_matrix | IMPLEMENTED |

### Evaluation Protocol

| Aspect | Method |
|--------|--------|
| Test set | Separate test directory (1,857 images) |
| Evaluation mode | Model.eval(), torch.no_grad() |
| Batch processing | Standard DataLoader iteration |
| Device | CUDA if available |

### Evaluation Concerns

| Issue | Severity | Description |
|-------|----------|-------------|
| All ROC-AUC = 1.0000 | HIGH | Perfect AUC for all 7 backbones is suspicious — may indicate easier dataset or overfitting |
| No confidence intervals | HIGH | Single evaluation, no bootstrap CI |
| No multi-seed evaluation | HIGH | Results from single training run |
| No per-class analysis on external data | MEDIUM | External validation shows aggregate metrics only |

---

## Robustness Evaluation

### Corruption Types Implemented

| Category | Corruption | Severities |
|----------|-----------|------------|
| Noise | Gaussian noise | σ = 0.05, 0.1, 0.2 |
| Noise | Salt-and-pepper | 0.01, 0.05, 0.1 |
| Blur | Gaussian blur | σ = 1.0, 2.0, 3.0 |
| Blur | Motion blur | length = 3, 5, 7 |
| Intensity | Brightness | 0.5, 0.7, 1.3, 1.5 |
| Intensity | Contrast | 0.5, 0.7, 1.3, 1.5 |
| Compression | JPEG quality | 10, 30, 50 |
| Geometric | Rotation | 5°, 10°, 15° |
| Geometric | Scale | 0.8, 0.9, 1.1, 1.2 |

### MRRS (MRI Robustness Retention Score)

- **Definition:** Geometric mean of performance retention across all corruptions
- **Formula:** MRRS = (∏ retention_i)^(1/n)
- **Range:** [0, 1], higher is better

### Robustness Concerns

| Issue | Severity | Description |
|-------|----------|-------------|
| No MRI-specific artifacts | HIGH | Missing: motion, Gibbs ringing, bias field, ghosting |
| No severity curve analysis | MEDIUM | Only aggregate MRRS reported |
| MRRS is a single number | MEDIUM | Hides per-corruption performance patterns |
| No comparison with other robustness metrics | LOW | Only MRRS reported |

---

## Calibration Evaluation

### Metrics Computed

| Metric | Implementation | Status |
|--------|---------------|--------|
| ECE | Custom (15 bins) | IMPLEMENTED |
| MCE | Custom (15 bins) | IMPLEMENTED |
| Brier Score | Custom | IMPLEMENTED |
| NLL | Custom | IMPLEMENTED |
| Temperature Scaling | LBFGS optimization | IMPLEMENTED |
| Reliability Diagram | matplotlib | IMPLEMENTED |

### Calibration Concerns

| Issue | Severity | Description |
|-------|----------|-------------|
| No temperature-scaled results reported | MEDIUM | Temperature scaling implemented but results not in documentation |
| No comparison across models | MEDIUM | Calibration evaluated per model but not compared |
| No calibration on external data | MEDIUM | Only internal test set calibration |
| ECE bin count fixed at 15 | LOW | May not be optimal for dataset size |

---

## Uncertainty Evaluation

### Methods Implemented

| Method | Implementation | Status |
|--------|---------------|--------|
| MC Dropout | 10 forward passes | IMPLEMENTED |
| Predictive variance | Variance across passes | IMPLEMENTED |
| Predictive entropy | Entropy of mean probabilities | IMPLEMENTED |
| Mutual information | Entropy - expected entropy | IMPLEMENTED |

### Uncertainty Concerns

| Issue | Severity | Description |
|-------|----------|-------------|
| No uncertainty-error correlation analysis | HIGH | Does not test if high uncertainty predicts errors |
| No AUROC for error detection | HIGH | Missing key uncertainty utility metric |
| No risk-coverage curve | HIGH | Missing selective prediction analysis |
| No selective accuracy analysis | MEDIUM | How does accuracy change with coverage? |
| MC Dropout passes = 10 | LOW | Could use more passes for stability |

---

## Explainability Evaluation

### Methods Implemented

| Method | Implementation | Status |
|--------|---------------|--------|
| Grad-CAM | pytorch-grad-cam | IMPLEMENTED |
| Grad-CAM++ | pytorch-grad-cam | EXISTS |
| ViT adaptation | CLS-gradient weights | IMPLEMENTED |

### Explainability Concerns

| Issue | Severity | Description |
|-------|----------|-------------|
| No localization quantification | HIGH | IoU with tumor masks, pointing game accuracy |
| No comparison across models | MEDIUM | Grad-CAM exists but not systematically compared |
| No quantitative evaluation | HIGH | Only visual inspection |
| No uncertainty-aware explanations | LOW | Explanations are deterministic |

---

## Segmentation Evaluation

### Metrics Implemented

| Metric | BraTS Region | Status |
|--------|-------------|--------|
| Dice | WT, TC, ET | IMPLEMENTED |
| Hausdorff 95% | WT, TC, ET | IMPLEMENTED |

### Segmentation Concerns

| Issue | Severity | Description |
|-------|----------|-------------|
| Dice scores are low (TC=0.58, ET=0.60) | HIGH | Model is undertrained (30/100 epochs) |
| No IoU reported | MEDIUM | Only Dice |
| No Sensitivity/Specificity | MEDIUM | Standard medical imaging metrics missing |
| No ASSD | LOW | Average symmetric surface distance |
| No surface Dice | LOW | Robust to outliers |
| No per-case statistics | MEDIUM | Only mean reported, no median/SD |

---

## Overall Evaluation Assessment

| Dimension | Rating (1-10) | Notes |
|-----------|---------------|-------|
| Metric coverage | 7 | Many metrics computed |
| Statistical rigor | 2 | No CI, no significance tests |
| Robustness analysis | 5 | Good corruption coverage, missing MRI-specific |
| Calibration analysis | 4 | Basic metrics, no comparative analysis |
| Uncertainty analysis | 3 | MC Dropout implemented, no utility analysis |
| Explainability analysis | 3 | Grad-CAM only, no quantitative evaluation |
| Segmentation analysis | 3 | Low Dice, limited metrics |
| External validation | 4 | Exists but suspicious results |
