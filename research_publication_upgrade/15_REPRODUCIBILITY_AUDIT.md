# 15 REPRODUCIBILITY AUDIT

## Reproducibility Components

### What Exists

| Component | Implementation | Status |
|-----------|---------------|--------|
| Seed setting | utils/seed.py | IMPLEMENTED |
| Checkpoint saving | engine/checkpoint.py | IMPLEMENTED |
| Config files | configs/*.yaml | IMPLEMENTED |
| Training state | outputs/training_state.json | IMPLEMENTED |
| Split manifest | data/splits/brats/split_manifest.json | IMPLEMENTED |

### What Is Missing

| Component | Status |
|-----------|--------|
| Experiment manifest | NOT IMPLEMENTED |
| Environment fingerprint | PARTIAL (in checkpoint) |
| Git commit tracking | NOT VERIFIED |
| Dataset hash | NOT IMPLEMENTED |
| Full hyperparameter logging | PARTIAL |
| Random state management | PARTIAL |

---

## Reproducibility Issues

### 1. Inconsistent Seeds

| Source | Seeds |
|--------|-------|
| paper.txt | [42, 123, 456] |
| Braintumor.txt | [42, 123, 2024] |
| classification.yaml | 42 (single) |

**Impact:** Cannot reproduce multi-seed results if seeds are inconsistent.

### 2. No Experiment Manifest

No JSON manifest records:
- Random seed
- Python version
- PyTorch version
- CUDA version
- GPU model
- Dataset version/hash
- Config file hash
- Git commit
- Training duration
- Final metrics

### 3. No Dataset Versioning

| Aspect | Status |
|--------|--------|
| Dataset hash | NOT COMPUTED |
| Dataset version | NOT TRACKED |
| Dataset source | NOT DOCUMENTED |
| Preprocessing version | NOT TRACKED |

### 4. Checkpoint Environment Fingerprint

The checkpoint manager includes some environment info:
- Platform
- Python version
- PyTorch version
- CUDA version
- GPU name

**Status:** PARTIALLY IMPLEMENTED

---

## Required Reproducibility Components

### Priority 1: Experiment Manifest

```json
{
  "experiment_id": "EXP001_baseline_densenet121",
  "seed": 42,
  "python_version": "3.13.0",
  "pytorch_version": "2.4.0",
  "cuda_version": "12.1",
  "gpu": "NVIDIA RTX 3050",
  "dataset_hash": "abc123...",
  "config_hash": "def456...",
  "git_commit": "789abc...",
  "training_duration_seconds": 3600,
  "metrics": {
    "accuracy": 0.9795,
    "f1": 0.9809
  }
}
```

### Priority 2: Deterministic Training

```python
import torch
import numpy as np
import random

def set_deterministic(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### Priority 3: Dataset Hashing

```python
import hashlib

def hash_directory(directory):
    hasher = hashlib.md5()
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            hasher.update(path.read_bytes())
    return hasher.hexdigest()
```

---

## Reproducibility Audit Summary

| Criterion | Status |
|-----------|--------|
| Seed setting | IMPLEMENTED |
| Checkpoint saving | IMPLEMENTED |
| Config files | IMPLEMENTED |
| Experiment manifest | NOT IMPLEMENTED |
| Environment fingerprint | PARTIAL |
| Dataset versioning | NOT IMPLEMENTED |
| Deterministic training | NOT VERIFIED |
| Git tracking | NOT VERIFIED |
