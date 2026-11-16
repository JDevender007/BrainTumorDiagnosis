# HYPOTHESES

## Formal Hypotheses

### H1: Tumor-Guided Feature Modulation Improves Classification

**Formal statement:**
Let M_base be a baseline classifier and M_tg be the same classifier with Adaptive Tumor-Guided Feature Modulation. Then:

F1(M_tg) > F1(M_base) on the internal test set.

**Test:** Paired bootstrap test, 5 seeds, significance level α=0.05.

**Effect size:** Cohen's d > 0.3 (small-to-medium effect).

---

### H2: Uncertainty-Aware Fusion Improves Calibration

**Formal statement:**
Let M_base be a baseline classifier and M_unc be the same classifier with uncertainty-modulated fusion. Then:

ECE(M_unc) < ECE(M_base)

**Test:** Paired comparison of ECE values, 5 seeds.

**Target:** ECE < 0.05.

---

### H3: Consistency Regularization Improves Generalization

**Formal statement:**
Let M_base be a baseline classifier and M_cons be the same classifier with cross-task consistency regularization. Then:

F1_external(M_cons) > F1_external(M_base) on BRISC 2025.

**Test:** External validation comparison, 5 seeds.

**Target:** Performance drop from internal to external < 2%.

---

### H4: Uncertainty Predicts Errors

**Formal statement:**
Let U(x) be the predictive uncertainty for input x and E(x) be the indicator function for misclassification. Then:

AUROC(U, E) > 0.75

**Test:** AUROC computation on test set.

**Target:** AUROC > 0.75, AUPRC > 0.5.

---

### H5: Tumor-Guided Models Are More Robust

**Formal statement:**
Let M_base be a baseline classifier and M_tg be the tumor-guided classifier. Then:

MRRS(M_tg) > MRRS(M_base)

**Test:** MRRS comparison across 9 corruption types.

**Target:** MRRS > 0.80.

---

## Hypothesis-Experiment Mapping

| Hypothesis | Experiment | Metric | Statistical Test |
|-----------|-----------|--------|-----------------|
| H1 | Baseline vs +ATGFM | F1 | Paired bootstrap |
| H2 | Baseline vs +Uncertainty | ECE | Paired comparison |
| H3 | Baseline vs +Consistency | External F1 | Paired bootstrap |
| H4 | Uncertainty analysis | AUROC | Bootstrap CI |
| H5 | Robustness analysis | MRRS | Paired comparison |

## Status

**HYPOTHESES: FORMALIZED**
**EXPERIMENTS: MAPPED**
**STATISTICAL TESTS: SPECIFIED**
**READY FOR IMPLEMENTATION: YES**
