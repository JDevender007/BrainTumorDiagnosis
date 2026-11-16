# Novelty Evidence

**Date:** 2026-09-05

---

## 1. What TURMA-Net Introduces

TURMA-Net adds three mechanisms to the TumorAwareMultiTaskNet baseline:

### 1.1 Adaptive Tumor-Guided Feature Modulation (ATGFM)

Standard attention fusion (TumorAwareMultiTaskNet):
```
fused = softmax(W @ [global; tumor]) * global + (1 - attn) * tumor
```

ATGFM (TURMA-Net):
```
combined = [global_proj; local_proj; uncertainty]
gamma = sigmoid(W_gamma @ combined)  # mixing ratio
beta = tanh(W_beta @ combined)       # shift
alpha = sigmoid(W_alpha @ combined)  # gate
fused = alpha * (gamma * global + (1-gamma) * tumor) + beta
```

**Key difference:** The fusion is conditioned on the *uncertainty of tumor localization*. When uncertainty is high, the model learns to rely more on global features. When low, it trusts tumor-localized features.

### 1.2 Cross-Task Consistency

KL divergence between classification and segmentation predictions:
```
L_cons = (KL(P_cls || P_seg) + KL(P_seg || P_cls)) / 2
```

**Assessment:** Standard knowledge distillation (Hinton et al. 2015). Not novel.

### 1.3 Uncertainty-Weighted Multi-Task Loss

Learned task weights via log-variance:
```
L_total = (1/2sigma1^2) * L_cls + (1/2sigma2^2) * L_seg + (1/2sigma3^2) * L_cons + log(sigma1) + log(sigma2) + log(sigma3)
```

**Assessment:** Directly from Kendall et al. 2018 "Multi-Task Learning Using Uncertainty to Weigh Losses". Not novel.

---

## 2. Ablation Evidence

| Configuration | Accuracy | F1 | AUC | vs Baseline |
|---------------|----------|-----|-----|-------------|
| Best Baseline (EfficientNetV2-S) | 0.9682 | 0.9693 | 0.9950 | -- |
| DenseNet121 Baseline | 0.9499 | 0.9512 | 0.9953 | -1.8% |
| TumorAwareMultiTaskNet | 0.8961 | 0.8965 | 0.9811 | -7.2% |
| TURMA-Net Full | 0.8584 | 0.8555 | 0.9614 | -11.0% |
| TURMA w/o ATGFM | 0.7270 | 0.7264 | 0.8997 | -24.1% |
| TURMA w/o Consistency | 0.8584 | 0.8555 | 0.9614 | -11.0% |
| TURMA w/o Uncertainty | 0.8584 | 0.8555 | 0.9614 | -11.0% |

### Key findings:

1. **ATGFM helps significantly within TURMA-Net:** Removing ATGFM drops accuracy from 0.8584 to 0.7270 (+13.1% improvement from ATGFM). This is the most important ablation result.

2. **Consistency loss has no effect:** G has identical results to E. The consistency loss appears to be zero or negligible in practice (likely because segmentation predictions are random without segmentation targets during classification training).

3. **Uncertainty weighting has no effect:** H has identical results to E. Same issue - the uncertainty loss terms may be too small to matter.

4. **Multi-task models underperform single-task baselines:** Both TumorAware (0.8961) and TURMA-Net (0.8584) are worse than all baselines (0.95-0.97). This is with only 3 epochs of training.

---

## 3. Honest Assessment

### What is genuinely new:
- ATGFM: Uncertainty-conditioned feature modulation is architecturally novel

### What is NOT new:
- Cross-task consistency (standard distillation)
- Uncertainty-weighted loss (Kendall et al. 2018)
- MC Dropout (Gal & Ghahramani 2016)
- Spatial/channel attention (CBAM, SENet)
- Segmentation head (standard decoder)
- Focal loss, Dice loss (standard)

### Does the novelty matter?

**ATGFM shows clear benefit within the TURMA-Net framework** (+13% accuracy when added). However, the overall TURMA-Net still underperforms single-task baselines. This means:

1. The ATGFM mechanism works as intended (uncertainty-conditioned fusion helps)
2. But the multi-task framework adds enough noise/difficulty to offset the gains
3. With more training epochs (we only used 3), the multi-task models might catch up

### Verdict:

**NOVELTY NOT YET FULLY ESTABLISHED**

ATGFM is a genuine architectural contribution. But it has not been shown to outperform existing baselines in classification accuracy. The novelty claim depends on:
- Whether ATGFM can be shown to improve a simpler architecture without the multi-task overhead
- Whether more training (10+ epochs) allows TURMA-Net to match or exceed baselines
- Whether the uncertainty mechanism provides meaningful uncertainty estimates (not yet evaluated)

The contribution is **promising but incomplete**.

---

## 4. Computational Cost

| Model | Params | Latency (GPU) |
|-------|--------|---------------|
| DenseNet121 Baseline | 6.96M | ~48ms |
| TumorAware | 9.03M | ~50ms |
| TURMA-Net | 9.49M | ~52ms |

TURMA-Net adds ~36% more parameters and ~8% more latency than the baseline. This is a modest increase.

---

*Assessment date: 2026-09-05*
