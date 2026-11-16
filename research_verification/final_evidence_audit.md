# Final Evidence Audit

**Date:** 2026-09-05

---

## Evidence Classification

For each important claim, classify the evidence status.

### Architecture Claims

| Claim | Status | Evidence |
|-------|--------|----------|
| TURMA-Net has ATGFM | VERIFIED BY CODE | turma_net.py:61-148 |
| ATGFM uses uncertainty to modulate features | VERIFIED BY CODE | turma_net.py:118-148 |
| TURMA-Net has cross-task consistency | VERIFIED BY CODE | turma_net.py:300-354 |
| TURMA-Net has uncertainty-weighted loss | VERIFIED BY CODE | turma_net.py:361-404 |
| TURMA-Net uses DenseNet121 backbone | VERIFIED BY CODE | turma_net.py:515-542 |
| TURMA-Net has segmentation head | VERIFIED BY CODE | turma_net.py:219-237 |
| TURMA-Net has MC Dropout uncertainty | VERIFIED BY CODE | turma_net.py:240-250 |

### Performance Claims

| Claim | Status | Evidence |
|-------|--------|----------|
| TURMA-Net outperforms baselines | CONTRADICTED | Ablation: acc=0.8584 vs best baseline 0.9682 |
| ATGFM improves over no-ATGFM | VERIFIED BY EXPERIMENT | E=0.8584 vs F=0.7270 (+13.1%) |
| Consistency loss helps | NOT VERIFIED | G=E (identical results, loss appears zero) |
| Uncertainty weighting helps | NOT VERIFIED | H=E (identical results, loss appears zero) |
| TURMA-Net outperforms TumorAware | CONTRADICTED | E=0.8584 vs D=0.8961 |
| Multi-task learning helps | CONTRADICTED | D=0.8961, E=0.8584 vs A=0.9499 |

### Statistical Claims

| Claim | Status | Evidence |
|-------|--------|----------|
| Results are statistically significant | NOT VERIFIED | No multi-seed statistical tests completed |
| Results are reproducible across seeds | NOT VERIFIED | Only 1 seed completed (ablation) |
| External validation passes | NOT VERIFIED | External evaluation not completed |

### Dataset Claims

| Claim | Status | Evidence |
|-------|--------|----------|
| Internal train = 9,291 | VERIFIED BY CODE | File count confirmed |
| Internal test = 1,857 | VERIFIED BY CODE | File count confirmed |
| BRISC = 1,000 | VERIFIED BY CODE | File count confirmed |
| Kaggle = 1,311 | VERIFIED BY CODE | File count confirmed |
| BraTS = 200 volumes | VERIFIED BY CODE | Directory structure confirmed |
| Patient-level split for classification | NOT VERIFIED | No patient IDs in filenames |
| No data leakage | PARTIALLY VERIFIED | 218 shared filenames (size-matched=0), Kaggle overlap untested |

### Efficiency Claims

| Claim | Status | Evidence |
|-------|--------|----------|
| ~48ms inference (RTX 3050) | VERIFIED BY EXPERIMENT | 47.8ms measured |
| ~20.6 FPS | VERIFIED BY EXPERIMENT | 20.9 FPS (1000/47.8) |
| ~2.1 GB VRAM | NOT VERIFIED | Only 118MB measured for inference (training VRAM not measured) |
| 6.96M params (baseline) | VERIFIED BY EXPERIMENT | Measured |
| 9.03M params (TumorAware) | VERIFIED BY EXPERIMENT | Measured |
| 9.49M params (TURMA-Net) | VERIFIED BY EXPERIMENT | Measured |

---

## Overall Assessment

| Category | Count | % |
|----------|-------|---|
| VERIFIED BY EXPERIMENT | 8 | 30% |
| VERIFIED BY CODE | 10 | 37% |
| PARTIALLY VERIFIED | 2 | 7% |
| NOT VERIFIED | 5 | 19% |
| CONTRADICTED | 4 | 15% |

**Total claims: 27**

**Key contradictions:**
1. TURMA-Net does NOT outperform baselines
2. Multi-task learning does NOT help (within 3-epoch budget)
3. Consistency and uncertainty losses have NO effect

---

*Audit date: 2026-09-05*
