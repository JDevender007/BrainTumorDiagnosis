# 08 STATISTICAL AUDIT

## Current Statistical Analysis Status

### What Exists

| Component | Implementation | Status |
|-----------|---------------|--------|
| Multi-seed training script | scripts/run_multi_seed.py | EXISTS |
| Statistical validation module | evaluation/statistical_validation.py | EXISTS |
| Seeds used | [42, 123, 456] or [42, 123, 2024] | INCONSISTENT |
| Multi-seed epochs | 15 | SHORTENED |

### What Is Missing

| Component | Status | Priority |
|-----------|--------|----------|
| Bootstrap confidence intervals | NOT IMPLEMENTED | HIGH |
| McNemar's test | NOT IMPLEMENTED | HIGH |
| Paired bootstrap test | NOT IMPLEMENTED | HIGH |
| DeLong test for AUC | NOT IMPLEMENTED | MEDIUM |
| Effect size (Cohen's d) | NOT IMPLEMENTED | MEDIUM |
| Multiple comparison correction | NOT IMPLEMENTED | MEDIUM |
| Per-seed results table | NOT VERIFIED | HIGH |
| Mean ± SD reporting | NOT VERIFIED | HIGH |

---

## Critical Statistical Issues

### 1. No Confidence Intervals

All reported metrics are **point estimates** without confidence intervals. This is unacceptable for publication.

**Required:**
- 95% bootstrap CI for accuracy, F1, AUC
- Mean ± SD across seeds
- Per-class confidence intervals

### 2. No Significance Testing

No statistical tests are performed to determine whether differences between models are significant.

**Required:**
- McNemar's test for paired classification comparisons
- Paired bootstrap for accuracy/F1 differences
- DeLong test for AUC differences

### 3. No Multi-Seed Results

The documentation does not report multi-seed results. The seeds [42, 123, 456] are mentioned but results are not shown.

**Required:**
- 5+ seeds minimum
- Mean ± SD for all metrics
- Statistical significance of proposed vs baseline

### 4. No Effect Size

No effect size measures are reported.

**Required:**
- Cohen's d for paired comparisons
- Cliff's delta for non-parametric comparisons

### 5. No Multiple Comparison Correction

If comparing 7+ models, Bonferroni or FDR correction is needed.

---

## Recommended Statistical Protocol

### For Classification Comparison

```python
# Bootstrap 95% CI for accuracy
n_bootstrap = 1000
accuracies = []
for _ in range(n_bootstrap):
    indices = np.random.choice(len(y_true), len(y_true), replace=True)
    accuracies.append(accuracy_score(y_true[indices], y_pred[indices]))
ci_lower = np.percentile(accuracies, 2.5)
ci_upper = np.percentile(accuracies, 97.5)

# McNemar's test for paired comparisons
from statsmodels.stats.contingency_tables import mcnemar
table = [[both_correct, model1_wrong_model2_correct],
         [model1_correct_model2_wrong, both_wrong]]
result = mcnemar(table, exact=True)
```

### For Multi-Seed Analysis

```python
# Report mean ± SD across seeds
results = []
for seed in [42, 123, 456, 789, 1024]:
    model = train(seed=seed)
    metrics = evaluate(model)
    results.append(metrics)

mean_acc = np.mean([r['accuracy'] for r in results])
std_acc = np.std([r['accuracy'] for r in results])
ci_95 = 1.96 * std_acc / np.sqrt(len(results))
```

---

## Statistical Audit Summary

| Criterion | Status |
|-----------|--------|
| Confidence intervals | NOT PRESENT |
| Significance testing | NOT PRESENT |
| Multi-seed evaluation | NOT VERIFIED |
| Effect size | NOT PRESENT |
| Multiple comparison correction | NOT PRESENT |
| Per-class statistics | NOT PRESENT |
| Bootstrap analysis | NOT PRESENT |

**STATISTICAL RIGOR: INSUFFICIENT FOR PUBLICATION**
