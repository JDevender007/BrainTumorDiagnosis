"""
Quantitative Grad-CAM localization analysis.

Compares Grad-CAM heatmaps with tumor masks to answer:
Does the classifier actually focus on tumor-relevant regions?

Computes:
- Grad-CAM/Tumor IoU
- Grad-CAM/Tumor Dice
- Localization accuracy
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from tqdm import tqdm


@dataclass
class LocalizationResult:
    iou: float
    dice: float
    localization_accuracy: float
    gradcam_mean: float
    tumor_mean: float
    overlap_ratio: float


@dataclass
class GradCAMAnalyzer:
    """Quantitative Grad-CAM analysis against tumor masks."""

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.gradients = None
        self.activations = None

    def _register_hooks(self, target_layer: nn.Module):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate_gradcam(
        self,
        image: torch.Tensor,
        target_class: int,
        target_layer: nn.Module,
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap."""
        self._register_hooks(target_layer)
        self.model.zero_grad()

        image = image.unsqueeze(0).to(self.device) if image.dim() == 3 else image.to(self.device)
        output = self.model(image)
        if isinstance(output, dict):
            logits = output["logits"]
        else:
            logits = output

        target = logits[0, target_class]
        target.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)

        self.gradients = None
        self.activations = None

        return cam

    def compute_localization_metrics(
        self,
        gradcam: np.ndarray,
        tumor_mask: np.ndarray,
        threshold: float = 0.5,
    ) -> LocalizationResult:
        """Compute localization metrics between Grad-CAM and tumor mask."""
        gradcam_binary = (gradcam > threshold).astype(float)
        tumor_binary = (tumor_mask > 0.5).astype(float)

        intersection = np.sum(gradcam_binary * tumor_binary)
        union = np.sum(gradcam_binary) + np.sum(tumor_binary) - intersection

        iou = intersection / union if union > 0 else 0.0
        dice = 2 * intersection / (np.sum(gradcam_binary) + np.sum(tumor_binary)) if (np.sum(gradcam_binary) + np.sum(tumor_binary)) > 0 else 0.0

        tumor_pixels = np.where(tumor_binary > 0.5)
        if len(tumor_pixels[0]) > 0:
            gradcam_at_tumor = gradcam[tumor_pixels]
            localization_accuracy = float(np.mean(gradcam_at_tumor > threshold))
        else:
            localization_accuracy = 0.0

        overlap_ratio = float(np.sum(gradcam_binary * tumor_binary) / np.sum(tumor_binary)) if np.sum(tumor_binary) > 0 else 0.0

        return LocalizationResult(
            iou=float(iou),
            dice=float(dice),
            localization_accuracy=localization_accuracy,
            gradcam_mean=float(np.mean(gradcam)),
            tumor_mean=float(np.mean(tumor_binary)),
            overlap_ratio=overlap_ratio,
        )

    def batch_analyze(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        tumor_masks: Optional[torch.Tensor],
        target_layer: nn.Module,
        class_names: list[str],
        threshold: float = 0.5,
    ) -> dict:
        """Batch Grad-CAM localization analysis."""
        results_per_class = {name: [] for name in class_names}

        for i in range(len(images)):
            image = images[i]
            label = labels[i].item()

            gradcam = self.generate_gradcam(image, label, target_layer)

            if tumor_masks is not None and i < len(tumor_masks):
                mask = tumor_masks[i].cpu().numpy()
                if mask.ndim == 3:
                    mask = mask[0]
                metrics = self.compute_localization_metrics(gradcam, mask, threshold)
            else:
                metrics = self.compute_localization_metrics(gradcam, np.ones_like(gradcam), threshold)

            results_per_class[class_names[label]].append({
                "iou": metrics.iou,
                "dice": metrics.dice,
                "localization_accuracy": metrics.localization_accuracy,
                "overlap_ratio": metrics.overlap_ratio,
            })

        summary = {}
        for class_name, class_results in results_per_class.items():
            if class_results:
                summary[class_name] = {
                    "mean_iou": float(np.mean([r["iou"] for r in class_results])),
                    "mean_dice": float(np.mean([r["dice"] for r in class_results])),
                    "mean_localization_accuracy": float(np.mean([r["localization_accuracy"] for r in class_results])),
                    "mean_overlap_ratio": float(np.mean([r["overlap_ratio"] for r in class_results])),
                    "num_samples": len(class_results),
                }

        all_results = []
        for class_results in results_per_class.values():
            all_results.extend(class_results)

        if all_results:
            summary["overall"] = {
                "mean_iou": float(np.mean([r["iou"] for r in all_results])),
                "mean_dice": float(np.mean([r["dice"] for r in all_results])),
                "mean_localization_accuracy": float(np.mean([r["localization_accuracy"] for r in all_results])),
                "mean_overlap_ratio": float(np.mean([r["overlap_ratio"] for r in all_results])),
                "num_samples": len(all_results),
            }

        return summary

    def save_results(self, summary: dict, output_dir: Path, model_name: str = "model"):
        """Save localization results."""
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / f"localization_{model_name}.json", "w") as f:
            json.dump(summary, f, indent=2)
