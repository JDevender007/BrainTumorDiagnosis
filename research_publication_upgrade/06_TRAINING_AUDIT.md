# 06 TRAINING AUDIT

## Classification Training Protocol

### Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| Optimizer | AdamW | classification_trainer.py |
| Learning rate | 1e-4 | classification.yaml |
| Weight decay | 1e-4 | classification.yaml |
| Epochs | 30 | classification.yaml |
| Batch size | 16 | classification.yaml |
| Patience | 7 | classification.yaml |
| Min delta | 1e-3 | classification_trainer.py |
| Label smoothing | 0.0 | classification.yaml |
| Class weights | True | classification.yaml |
| AMP | True | classification.yaml |
| Gradient clipping | 1.0 | classification.yaml |
| Gradient accumulation | 1 | classification.yaml |
| LR scheduler | ReduceLROnPlateau | classification_trainer.py |

### Training Data Split

| Split | Source | Method |
|-------|--------|--------|
| Train | 80% of training directory | Stratified by class, seed=42 |
| Validation | 20% of training directory | Stratified by class, seed=42 |
| Test | Separate test directory | Never used during training |

### Data Augmentation

| Transform | Parameters |
|-----------|------------|
| RandomHorizontalFlip | p=0.5 |
| RandomRotation | degrees=10 |
| RandomAffine | translate=(0.05,0.05), scale=(0.95,1.05) |
| ColorJitter | brightness=0.10, contrast=0.10 |

### Training Issues

| Issue | Severity | Description |
|-------|----------|-------------|
| No multi-seed training | HIGH | Only single-seed results reported |
| No early stopping on metrics other than accuracy | MEDIUM | Could overfit on accuracy while other metrics degrade |
| Label smoothing = 0.0 | LOW | Could benefit from non-zero smoothing |
| Augmentation is minimal for medical images | MEDIUM | No elastic deformation, no affine beyond basic |
| No test-time augmentation | LOW | Could improve robustness |

---

## Segmentation Training Protocol

### Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| Model | 3D U-Net (UNet3D) | segmentation.yaml |
| Optimizer | AdamW | segmentation_trainer.py |
| Loss | Dice + CE combo | losses/segmentation.py |
| Scheduler | CosineAnnealing | segmentation_trainer.py |
| Patch size | 96×96×96 | segmentation.yaml |
| Batch size | 1 | segmentation.yaml |
| Foreground sampling | True | segmentation.yaml |
| Epochs target | 100 | segmentation.yaml |
| Epochs completed | 30 | metadata.json |

### Segmentation Training Issues

| Issue | Severity | Description |
|-------|----------|-------------|
| Only 30/100 epochs completed | HIGH | Model is undertrained |
| Best val Dice = 0.5243 | HIGH | Low performance (vs claimed 0.5669) |
| No multi-seed evaluation | MEDIUM | Single seed only |
| No post-training evaluation on BraTS regions | MEDIUM | WT/TC/ET evaluation exists but may not match training |

---

## Self-Supervised Training Protocol

### Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| Method | SimCLR | self_supervised.yaml |
| Backbone | DenseNet121 | self_supervised.yaml |
| Loss | NT-Xent | self_supervised_trainer.py |
| Epochs | 50 | self_supervised.yaml |
| Projection head | 512→128 | self_supervised_trainer.py |

### SSL Training Issues

| Issue | Severity | Description |
|-------|----------|-------------|
| Not integrated into main training pipeline | MEDIUM | SSL checkpoint exists but unclear if used for fine-tuning |
| Single backbone only | LOW | Only DenseNet121 SSL |
| No evaluation of SSL benefit | MEDIUM | No ablation comparing with/without SSL |

---

## Multi-Task Training Protocol

### Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| Model | TumorAwareMultiTaskNet | proposed_model.yaml |
| Epochs | 50 (config) / 30 (claimed) | INCONSISTENT |
| Batch size | 8 | proposed_model.yaml |
| Label smoothing | 0.1 | proposed_model.yaml |
| Seg loss weight | 0.5 | proposed_model.yaml |
| Uncertainty loss weight | 0.1 | proposed_model.yaml |
| Gradient accumulation | 2 | proposed_model.yaml |

### Multi-Task Training Issues

| Issue | Severity | Description |
|-------|----------|-------------|
| Config vs documentation epoch inconsistency | MEDIUM | 50 in config, 30 in docs |
| No learning rate scheduling documented | MEDIUM | Missing from config |
| Loss weight tuning not documented | MEDIUM | Arbitrary weights without justification |
| No ablation of loss weights | MEDIUM | Critical for reproducibility |

---

## Overall Training Assessment

| Dimension | Rating (1-10) | Notes |
|-----------|---------------|-------|
| Training protocol | 6 | Standard but complete |
| Hyperparameter tuning | 3 | No systematic tuning documented |
| Reproducibility | 5 | Seeds inconsistent, epochs inconsistent |
| Multi-seed validation | 2 | Not performed for most experiments |
| Statistical rigor | 2 | No confidence intervals, no significance tests |
| Ablation | 2 | Runner exists but results not verified |
| Documentation | 3 | Multiple inconsistencies |
