# 13 EXPLAINABILITY AUDIT

## Current Explainability Implementation

### Methods Available

| Method | Implementation | Status |
|--------|---------------|--------|
| Grad-CAM | pytorch-grad-cam | IMPLEMENTED |
| Grad-CAM++ | pytorch-grad-cam | EXISTS |
| ViT adaptation | CLS-gradient channel weights | IMPLEMENTED |

### Output Format

- 3-panel figures: original, overlay, heatmap
- Per-class representative examples
- 24 PNG files (6 backbones × 4 classes)

---

## Explainability Gaps

### 1. No Quantitative Localization Evaluation

| Metric | Status | Description |
|--------|--------|-------------|
| IoU with tumor mask | NOT PERFORMED | Does Grad-CAM overlap with tumor? |
| Pointing game accuracy | NOT PERFORMED | Is max activation inside tumor? |
| Inside-tumor activation ratio | NOT PERFORMED | What fraction of activation is inside tumor? |
| Outside-tumor activation ratio | NOT PERFORMED | What fraction of activation is outside tumor? |
| Center-of-mass distance | NOT PERFORMED | Distance between activation and tumor centers |

### 2. No Comparative Analysis

| Analysis | Status |
|----------|--------|
| Grad-CAM across models | NOT COMPARED |
| Which model localizes better? | NOT TESTED |
| Does tumor-guided model focus on tumor? | NOT TESTED |

### 3. No Correct vs Incorrect Comparison

| Analysis | Status |
|----------|--------|
| Grad-CAM for correct predictions | NOT SYSTEMATIC |
| Grad-CAM for incorrect predictions | NOT SYSTEMATIC |
| Does explainability differ for errors? | NOT TESTED |

### 4. No Confidence-Stratified Analysis

| Analysis | Status |
|----------|--------|
| High-confidence Grad-CAM | NOT PERFORMED |
| Low-confidence Grad-CAM | NOT PERFORMED |
| Does confidence affect localization? | NOT TESTED |

### 5. No Multi-Method Comparison

| Analysis | Status |
|----------|--------|
| Grad-CAM vs Grad-CAM++ | NOT COMPARED |
| Grad-CAM vs Integrated Gradients | NOT COMPARED |
| Which method localizes best? | NOT TESTED |

---

## Required Explainability Analyses

### Priority 1: Quantitative Localization

If tumor masks are available (from 3D U-Net or annotations):

```python
# Grad-CAM IoU with tumor mask
cam = compute_gradcam(model, image, target_class)
cam_binary = (cam > threshold).float()
iou = (cam_binary * tumor_mask).sum() / (cam_binary + tumor_mask - cam_binary * tumor_mask).sum()

# Pointing game
max_location = np.unravel_index(cam.argmax(), cam.shape)
inside_tumor = tumor_mask[max_location] > 0
```

### Priority 2: Comparative Analysis

```python
# Compare Grad-CAM across models
for model_name, model in models.items():
    cam = compute_gradcam(model, image)
    iou = compute_iou(cam, tumor_mask)
    results[model_name] = iou
```

### Priority 3: Error Analysis

```python
# Grad-CAM for correct vs incorrect
correct_mask = predictions == labels
incorrect_mask = ~correct_mask

# Do incorrect predictions have worse localization?
correct_iou = [compute_iou(cam[i], mask[i]) for i in correct_mask]
incorrect_iou = [compute_iou(cam[i], mask[i]) for i in incorrect_mask]
```

---

## Explainability Audit Summary

| Criterion | Status |
|-----------|--------|
| Grad-CAM implementation | IMPLEMENTED |
| Visual output | IMPLEMENTED |
| Quantitative localization | NOT PERFORMED |
| Comparative analysis | NOT PERFORMED |
| Error analysis | NOT PERFORMED |
| Multi-method comparison | NOT PERFORMED |
| Confidence-stratified | NOT PERFORMED |
