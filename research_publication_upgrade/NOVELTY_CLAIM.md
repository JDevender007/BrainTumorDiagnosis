# NOVELTY CLAIM

## Claimed Novelty

**Status: NOT YET ESTABLISHED**

The existing project combines established techniques:
- CNN/Transformer backbones (ImageNet pretrained)
- CBAM attention
- SimCLR self-supervised learning
- Multi-task learning
- Grad-CAM explainability
- MC Dropout uncertainty

**This is NOT a novel contribution.** It is an engineering assembly of published methods.

---

## Proposed Novelty (Under Development)

### Adaptive Tumor-Guided Feature Modulation (ATGFM)

Uses predicted tumor uncertainty to adaptively modulate feature fusion between global and tumor-localized representations.

**Differentiation from existing work:**
- Existing: Fixed attention weights for fusion
- Proposed: Uncertainty-dependent adaptive modulation

### Cross-Task Consistency Regularization

Enforces consistency between classification, segmentation, and localization confidence.

**Differentiation from existing work:**
- Existing: Tasks optimized independently
- Proposed: Explicit cross-task consistency loss

### Uncertainty-Weighted Multi-Task Loss

Learns task-specific importance weights via uncertainty parameterization.

**Differentiation from existing work:**
- Existing: Fixed loss weights
- Proposed: Learned, adaptive task weights

---

## Novelty Verification Status

| Criterion | Status |
|-----------|--------|
| Literature search | NOT PERFORMED |
| Similar work identified | UNKNOWN |
| Novelty confirmed | NOT YET |
| Differentiation documented | NOT YET |

**NOVELTY CLAIM CANNOT BE MADE UNTIL LITERATURE SEARCH IS COMPLETE.**
