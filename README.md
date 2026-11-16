# BrainTumorDiagnosis

Multi-modal MRI analysis pipeline for brain tumor diagnosis on the
RTX 3050 Laptop (4 GB VRAM). The pipeline combines:

- **3D segmentation** of tumor regions from multi-modal MRI volumes
  (T1n, T1c, T2w, T2FLAIR) with a lightweight 3D U-Net.
- **2D slice classification** of tumor type (Glioma, Meningioma,
  Pituitary, No Tumor) with a unified backbone zoo.
- **Tumor-guided multi-task learning** (TURMA-Net) that fuses
  tumor-localized features with global features for classification.
- **Grad-CAM explainability** for slice-level predictions.
- **MC Dropout uncertainty** estimation for confidence calibration.
- **Self-supervised contrastive pretraining** (SimCLR) to initialize
  supervised encoders from unlabeled MRI slices.

> **Methodology note:** Classification operates on **2D MRI slices**
> (each slice classified independently). Segmentation operates on
> **3D volumetric patches** from BraTS volumes. The two tasks use
> different input modalities and are integrated via the TURMA-Net
> architecture.

---

## Honest results

All numbers below come from the official **Test split** (1857 images),
which is never used for validation during training. No numbers are
invented; anything not measured is reported as "not measured".

### Classification Results (Test Set - 1,857 images)

| Model | Test Accuracy | F1-Score | ROC-AUC | Parameters | Inference (FPS) |
|-------|---------------|----------|---------|------------|-----------------|
| EfficientNetV2-S | 98.06% | 0.9811 | 1.0000 | 20.18M | 195.1 |
| DenseNet121 | 97.95% | 0.9809 | 1.0000 | 6.96M | 209.4 |
| ViT-B/16 | 97.52% | 0.9764 | 1.0000 | 85.80M | 56.2 |
| ConvNeXt-Tiny | 97.42% | 0.9751 | 1.0000 | 27.82M | 162.5 |
| Swin-T | 97.09% | 0.9722 | 1.0000 | 27.52M | 121.6 |
| ResNet50 | 97.04% | 0.9720 | 1.0000 | 23.52M | 207.7 |
| CNN-2D (residual) | 95.26% | 0.9547 | 1.0000 | 21.40M | 307.8 |

### External Validation - BRISC 2025 (1,000 images)

| Model | External Accuracy | F1-Score | Kappa |
|-------|-------------------|----------|-------|
| Swin-T | 99.20% | 0.9916 | 0.9891 |
| DenseNet121 | 98.40% | 0.9816 | 0.9781 |
| ResNet50 | 98.00% | 0.9770 | 0.9726 |
| EfficientNetV2-S | 97.90% | 0.9745 | 0.9712 |
| ViT-B/16 | 97.80% | 0.9745 | 0.9699 |
| ConvNeXt-Tiny | 97.50% | 0.9693 | 0.9657 |
| CNN-2D (residual) | 96.40% | 0.9559 | 0.9506 |

### Segmentation Results (BraTS 2024)

| Region | Dice Score | Hausdorff 95% |
|--------|------------|---------------|
| Whole Tumor (WT) | 0.9005 | 4.69 mm |
| Tumor Core (TC) | 0.5803 | 34.20 mm |
| Enhancing Tumor (ET) | 0.6010 | 16.33 mm |

- Segmentation training: 30/100 epochs completed, best validation Dice 0.5243
- BraTS2024 case folders in `data/raw/BraTS2024/`, patient-level split
- Labels: background, NETC, SNFH, ET, RC (BraTS 2024 schema)

### Dataset Summary

| Dataset | Images/Volumes | Classes | Split |
|---------|---------------|---------|-------|
| Internal (train) | 9,291 images | 4 | 80/20 train/val |
| Internal (test) | 1,857 images | 4 | Held-out test |
| BRISC 2025 (external) | 1,000 images | 4 | Test only |
| Kaggle (external) | 1,311 images | 4 | Test only |
| BraTS 2024 (segmentation) | 200 volumes | 4 seg classes | 70/15/15 patient-level |

---

## Setup

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Data layout (classification dataset, 4 classes):

```
data/raw/classification/
├── Train/           (9,291 images total)
│   ├── glioma/      (3,018)
│   ├── meningioma/  (2,183)
│   ├── pituitary/   (2,145)
│   └── no tumor/    (1,945)
└── Test/            (1,857 images total)
    ├── glioma/      (603)
    ├── meningioma/  (436)
    ├── pituitary/   (429)
    └── no tumor/    (389)
```

**Note:** Dataset discovery handles case-insensitive class directory matching
(glioma, meningioma, pituitary, "no tumor" variants are all supported).

BraTS2024 case folders live in `data/raw/BraTS2024/` and are split at
**patient level** (4–10 exam folders per patient) into 129/26/45 cases.

---

## Training

Unified config-driven trainer. The official Test split is never used
for validation; validation is held out from the official training
split (20%).

```powershell
# Classification training (all backbones)
python scripts/train_classification.py --config configs/densenet121.yaml
python scripts/train_classification.py --config configs/resnet50.yaml
python scripts/train_classification.py --config configs/efficientnet_v2_s.yaml
python scripts/train_classification.py --config configs/convnext_tiny.yaml
python scripts/train_classification.py --config configs/swin_t.yaml
python scripts/train_classification.py --config configs/vit_b_16.yaml
python scripts/train_classification.py --config configs/classification.yaml

# Segmentation (3D U-Net on BraTS 2024)
python scripts/train_segmentation.py

# Resume from checkpoint
python scripts/train_classification.py --config configs/densenet121.yaml `
  --resume-from checkpoints/classification/densenet121/last.pt

# Verify the loop quickly
python scripts/train_classification.py --config configs/classification.yaml --smoke
```

Checkpoints land in `checkpoints/classification/<backbone>/best.pt` and
`last.pt` with training history in `outputs/<backbone>_training/`.

### Training manager (stop / resume / progress)

A persistent manager tracks project-level and per-backbone progress in
`outputs/training_state.json`. Training survives PC shutdowns: the
trainer saves `last.pt` + history every epoch, and the manager resumes
interrupted runs automatically.

```powershell
# initialize state (detects existing checkpoints)
python scripts/manage_training.py init

# start everything that is not complete (resumes where it left off)
python scripts/manage_training.py start --all

# or train a single backbone
python scripts/manage_training.py start --backbone resnet50

# graceful stop: the running trainer exits after the current epoch
python scripts/manage_training.py stop

# project % and per-backbone % (epochs done / target), once
python scripts/manage_training.py status

# ...and every N minutes while training runs (default 10)
python scripts/manage_training.py watch --interval 10

# after a PC shutdown: just start again, interrupted runs resume
python scripts/manage_training.py resume
```

Project completion is a weighted milestone score: segmentation 15%,
classification baselines 60%, self-supervised pretraining 10%,
test-set evaluations 10%, Grad-CAM figures 5%. Each backbone's
training percentage is `epochs completed / target epochs`.

### Self-supervised pretraining (SimCLR)

```powershell
# Train SimCLR (50 epochs, DenseNet121 backbone)
python scripts/train_self_supervised.py --config configs/self_supervised.yaml

# Resume SSL training
python scripts/train_self_supervised.py `
  --resume-from checkpoints/self_supervised/densenet121/last.pt
```

The SSL checkpoint stores the full encoder (`encoder_state_dict`);
loading it into a classification model uses `strict=False`, so the
final classification head is freshly initialized while the backbone
weights transfer.

---

## Evaluation

### Comprehensive Metrics (30+ metrics)

```powershell
# Run full evaluation on all backbones (internal test + BRISC 2025 external)
python -m evaluation.comprehensive_metrics
```

Computes 11 metrics per backbone:
- **Accuracy, Precision, Recall, F1-Score** (macro-averaged)
- **Specificity** (true negative rate)
- **ROC-AUC, PR-AUC** (one-vs-rest)
- **MCC** (Matthews Correlation Coefficient)
- **Cohen's Kappa**
- **LogLoss** (cross-entropy)
- **Per-class F1** (glioma, meningioma, pituitary, no tumor)

Includes:
- **Robustness analysis** (Gaussian noise σ=0.05, 0.1, 0.2)
- **Inference speed benchmarking** (latency, FPS)
- **External validation** on BRISC 2025 dataset

Results saved to: `outputs/comprehensive_metrics/`

### Single Backbone Evaluation

```powershell
python -m evaluation.classification_evaluator --backbone densenet121
python -m evaluation.classification_evaluator --backbone cnn2d
```

Metrics are written to `outputs/<backbone>_evaluation/evaluation_metrics.json`
(accuracy, weighted precision/recall/F1, ROC-AUC ovr, confusion matrix,
per-class report) plus `predictions.json`.

---

## Explainability (Grad-CAM)

```powershell
# Generate Grad-CAM for all 7 backbones x 4 classes
python -m evaluation.explainability

# Single backbone
python -m evaluation.explainability --backbone densenet121 `
  --image data/raw/classification/test/glioma/1049.jpg
```

Writes a 3-panel figure (original, overlay, heatmap) next to the input
image. ViT/Swin use the last-stage feature map; ViT uses CLS-gradient
channel weights (token gradients are zero by construction).

Output: 28 PNG files in `outputs/gradcam/` (7 backbones x 4 classes).

---

## Inference (single case)

```powershell
python -m inference.predict data/raw/BraTS2024/BraTS-GLI-02062-100
```

Runs segmentation + classification, writes a JSON report and the
segmentation NIfTI to `outputs/predictions/`. End-to-end pipeline with
visualizations and reports: `python -m engine.pipeline`.

---

## Tests

```powershell
python -m pytest tests -q
```

Covers the model factory and backbone zoo, checkpoint management,
metric utilities, dataset discovery and patient-level splitting, the
unified trainer (including a tiny end-to-end run), the evaluator,
Grad-CAM, and the SimCLR module.

---

## Project structure

```
TAMNet_Paper.docx        current paper draft (single source of truth for the audit)
docs/                    audit documents: AUDIT_REPORT.md, TODO.md
paper/                   paper generator, manuscript/, figures/, results/, tables/
checkpoints/             trained weights (classification/<backbone>/) [gitignored]
configs/                 per-backbone YAML training configs
data/                    datasets (classification, segmentation, external) [gitignored]
datasets/                dataset loaders, preprocessing, patient-level splitting
engine/                  inference, pipeline, reports, visualization
evaluation/              evaluator, comprehensive metrics, Grad-CAM, robustness, calibration, uncertainty
inference/               CLI entry points
losses/                  classification / segmentation losses
models/                  factory + backbone zoo + TAMNet (tumor_aware.py) + TURMA-Net
outputs/                 metrics, predictions, histories, logs, training_state.json [gitignored]
preprocessing/           volume preprocessing utilities
results/                 audit results (leakage reports)
scripts/                 training CLI + experiment scripts + runners (run.bat, run_project.bat, run_everything.py, resume_training.ps1)
segmentation/            segmentation utilities
training/                unified classifier + SimCLR + segmentation + multitask + manager
tests/                   pytest suite
utils/                   device, seeds, metrics, logging
```

## Audit report

The audit of `TAMNet_Paper.docx` (every number traced to a script output,
code-vs-paper consistency, reproducibility plan) is documented in
`docs/AUDIT_REPORT.md`, with the phased checklist in `docs/TODO.md`.
