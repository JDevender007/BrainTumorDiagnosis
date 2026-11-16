# 05 NOVELTY AUDIT

## Existing Project Contribution Analysis

### What the Existing System Does

1. **Multi-backbone comparison** — Compares 7 torchvision backbones for brain tumor MRI classification
2. **3D segmentation** — Trains a 3D U-Net on BraTS volumes
3. **Proposed model (TumorAwareMultiTaskNet)** — Combines:
   - 2D backbone encoder (ImageNet pretrained)
   - 2D segmentation head (lightweight, not connected to 3D U-Net)
   - Attention-weighted tumor region pooling
   - Attention-based feature fusion
   - MC Dropout uncertainty
4. **Evaluation suite** — Robustness, calibration, uncertainty, explainability
5. **External validation** — BRISC 2025, Kaggle datasets

### What Is Novel in the Existing System?

**Assessment: MINIMAL GENUINE NOVELTY**

The existing system combines well-established techniques:

| Component | Pre-existing Technique | Novelty Claim |
|-----------|----------------------|---------------|
| Multi-backbone comparison | Standard practice | NONE |
| CBAM attention | Woo et al. 2018 | NONE |
| SimCLR pretraining | Chen et al. 2020 | NONE |
| Multi-task learning | Caruana 1997 | NONE |
| Grad-CAM explainability | Selvaraju et al. 2017 | NONE |
| MC Dropout uncertainty | Gal & Ghahramani 2016 | NONE |
| Tumor-guided feature pooling | Similar to ROI pooling | MINIMAL |
| Attention-based feature fusion | Standard attention mechanism | MINIMAL |

### Novelty Gap Analysis

The system combines these techniques but does not introduce:
1. A novel architecture design
2. A novel loss function
3. A novel training strategy
4. A novel theoretical insight
5. A novel evaluation protocol

**The contribution is "combination of existing techniques" — this is NOT sufficient for publication in TMI or similar venues.**

---

## Literature Novelty Assessment

### Related Published Work

| Category | Representative Papers | What They Do |
|----------|---------------------|--------------|
| Brain tumor classification | Various (2020-2025) | CNN/Transformer classification on MRI slices |
| Multi-task brain tumor | Various | Joint classification + segmentation |
| Tumor-guided classification | Various | Use segmentation to guide classification |
| Self-supervised MRI | Various | Contrastive pretraining on medical images |
| Uncertainty in medical AI | Various | MC Dropout, ensemble, evidential methods |
| Calibration in medical AI | Various | Temperature scaling, Platt scaling |
| Robustness in medical AI | Various | Corruption robustness, adversarial robustness |

### Key Question: Does a similar method already exist?

**Preliminary Assessment:**

A method combining "2D backbone + segmentation-guided feature fusion + uncertainty" likely already exists in the literature. The specific combination in TumorAwareMultiTaskNet (attention pooling + attention fusion + MC Dropout) is a straightforward assembly of published components.

**Without a comprehensive literature search, we cannot claim novelty.**

---

## Novelty Decision Gate

| Criterion | Status |
|-----------|--------|
| Does the method introduce a new architecture? | NO — assembly of existing components |
| Does the method introduce a new loss function? | NO |
| Does the method introduce a new training strategy? | NO |
| Does the method solve an unsolved problem? | NO — brain tumor classification is well-studied |
| Does the method provide new theoretical insight? | NO |
| Does the method achieve significantly better results? | UNKNOWN — no rigorous comparison |
| Is the method different from published work? | UNKNOWN — requires literature search |

**NOVELTY STATUS: WEAK TO NOT NOVEL**

The existing system needs a genuinely novel contribution to be publication-worthy. The combination of existing techniques is not sufficient.

---

## Required Actions

1. **Comprehensive literature search** on brain tumor classification methods
2. **Identify genuine research gap** not addressed by existing work
3. **Design novel method** that addresses the gap
4. **Verify novelty** against published literature
5. **Define testable hypothesis** with measurable outcomes
