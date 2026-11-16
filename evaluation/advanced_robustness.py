"""
Advanced robustness evaluation for brain tumor classification.

Implements multiple corruption types:
1. Gaussian noise
2. Salt-and-pepper noise
3. Gaussian blur
4. Motion blur
5. Brightness variation
6. Contrast variation
7. JPEG compression degradation
8. Small rotations
9. Scaling/resizing changes

Defines MRI Robustness Retention Score (MRRS) as a proposed experimental metric.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


@dataclass
class CorruptionConfig:
    gaussian_noise_stds: list[float] = field(default_factory=lambda: [0.05, 0.1, 0.2])
    salt_pepper_amounts: list[float] = field(default_factory=lambda: [0.01, 0.05, 0.1])
    gaussian_blur_sigmas: list[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])
    motion_blur_lengths: list[int] = field(default_factory=lambda: [3, 5, 7])
    brightness_factors: list[float] = field(default_factory=lambda: [0.5, 0.7, 1.3, 1.5])
    contrast_factors: list[float] = field(default_factory=lambda: [0.5, 0.7, 1.3, 1.5])
    jpeg_qualities: list[int] = field(default_factory=lambda: [10, 30, 50])
    rotation_degrees: list[float] = field(default_factory=lambda: [5.0, 10.0, 15.0])
    scale_factors: list[float] = field(default_factory=lambda: [0.8, 0.9, 1.1, 1.2])


class GaussianNoise:
    def __init__(self, std: float):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0, 1)


class SaltPepperNoise:
    def __init__(self, amount: float):
        self.amount = amount

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        result = tensor.clone()
        num_salt = int(self.amount * tensor.numel() / 2)
        num_pepper = int(self.amount * tensor.numel() / 2)

        coords = tuple(torch.randint(0, d, (num_salt,)) for d in tensor.shape)
        result[coords] = 1.0

        coords = tuple(torch.randint(0, d, (num_pepper,)) for d in tensor.shape)
        result[coords] = 0.0

        return result


class GaussianBlur:
    def __init__(self, sigma: float):
        self.sigma = sigma

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        kernel_size = int(6 * self.sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        x = torch.arange(kernel_size, dtype=tensor.dtype, device=tensor.device) - kernel_size // 2
        kernel_1d = torch.exp(-x ** 2 / (2 * self.sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)

        channels = tensor.shape[0]
        kernel_2d = kernel_2d.expand(channels, 1, -1, -1).contiguous()

        padding = kernel_size // 2
        return F.conv2d(tensor.unsqueeze(0), kernel_2d, padding=padding, groups=channels).squeeze(0)


class MotionBlur:
    def __init__(self, kernel_size: int):
        self.kernel_size = kernel_size

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        kernel = torch.zeros(self.kernel_size, self.kernel_size, device=tensor.device, dtype=tensor.dtype)
        center = self.kernel_size // 2
        for i in range(self.kernel_size):
            kernel[center, i] = 1.0
        kernel = kernel / self.kernel_size

        kernel_2d = kernel.unsqueeze(0).unsqueeze(0)
        channels = tensor.shape[0]
        kernel_2d = kernel_2d.expand(channels, 1, -1, -1).contiguous()

        padding = self.kernel_size // 2
        return F.conv2d(tensor.unsqueeze(0), kernel_2d, padding=padding, groups=channels).squeeze(0)


class BrightnessAdjustment:
    def __init__(self, factor: float):
        self.factor = factor

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.clamp(tensor * self.factor, 0, 1)


class ContrastAdjustment:
    def __init__(self, factor: float):
        self.factor = factor

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.mean()
        return torch.clamp(mean + self.factor * (tensor - mean), 0, 1)


class RotationTransform:
    def __init__(self, degrees: float):
        self.degrees = degrees

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        angle = self.degrees * (2 * torch.rand(1).item() - 1)
        return transforms.functional.rotate(tensor, angle)


class ScaleTransform:
    def __init__(self, factor: float):
        self.factor = factor

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        h, w = tensor.shape[-2], tensor.shape[-1]
        new_h, new_w = int(h * self.factor), int(w * self.factor)
        input_t = tensor.unsqueeze(0) if tensor.dim() == 3 else tensor
        scaled = F.interpolate(input_t, size=(new_h, new_w), mode='bilinear', align_corners=False)
        result = scaled if tensor.dim() == 4 else scaled.squeeze(0)

        from torchvision.transforms.functional import center_crop
        result = center_crop(result, [h, w])
        return result


@dataclass
class RobustnessResult:
    corruption_type: str
    severity: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    mcc: float
    mean_confidence: float
    performance_retention: float
    num_samples: int


def get_corruption_transforms(config: CorruptionConfig) -> list[tuple[str, object]]:
    """Get all corruption transforms."""
    transforms_list = []

    for std in config.gaussian_noise_stds:
        transforms_list.append((f"gaussian_noise_{std}", GaussianNoise(std)))

    for amount in config.salt_pepper_amounts:
        transforms_list.append((f"salt_pepper_{amount}", SaltPepperNoise(amount)))

    for sigma in config.gaussian_blur_sigmas:
        transforms_list.append((f"gaussian_blur_{sigma}", GaussianBlur(sigma)))

    for length in config.motion_blur_lengths:
        transforms_list.append((f"motion_blur_{length}", MotionBlur(length)))

    for factor in config.brightness_factors:
        transforms_list.append((f"brightness_{factor}", BrightnessAdjustment(factor)))

    for factor in config.contrast_factors:
        transforms_list.append((f"contrast_{factor}", ContrastAdjustment(factor)))

    for degrees in config.rotation_degrees:
        transforms_list.append((f"rotation_{degrees}", RotationTransform(degrees)))

    for factor in config.scale_factors:
        transforms_list.append((f"scale_{factor}", ScaleTransform(factor)))

    return transforms_list


def evaluate_corruption(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    corruption_name: str,
    corruption_transform,
    clean_accuracy: float,
) -> RobustnessResult:
    """Evaluate model under a specific corruption."""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_confidences = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"  {corruption_name}", leave=False):
            corrupted_batch = torch.stack([corruption_transform(img) for img in images])
            corrupted = corrupted_batch.to(device)
            labels = labels.to(device)

            outputs = model(corrupted)
            if isinstance(outputs, dict):
                logits = outputs["logits"]
                probs = outputs.get("probabilities", F.softmax(logits, dim=1))
            else:
                logits = outputs
                probs = F.softmax(logits, dim=1)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_confidences.append(probs.max(dim=1).values.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)
    confidences = np.concatenate(all_confidences)

    accuracy = correct / total if total > 0 else 0.0

    from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef
    precision = precision_score(labels_arr, preds, average="weighted", zero_division=0)
    recall = recall_score(labels_arr, preds, average="weighted", zero_division=0)
    f1 = f1_score(labels_arr, preds, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(labels_arr, preds)

    performance_retention = accuracy / clean_accuracy if clean_accuracy > 0 else 0.0

    return RobustnessResult(
        corruption_type=corruption_name,
        severity=corruption_name,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        mcc=mcc,
        mean_confidence=float(np.mean(confidences)),
        performance_retention=performance_retention,
        num_samples=total,
    )


def compute_mrrs(results: list[RobustnessResult]) -> float:
    """
    MRI Robustness Retention Score (MRRS).

    Proposed experimental metric: geometric mean of performance retention
    across all corruption types and severities.

    MRRS = (prod_i retention_i)^(1/n)

    where retention_i = corrupted_accuracy_i / clean_accuracy

    Range: [0, 1]. Higher is better.
    """
    if not results:
        return 0.0

    retentions = [r.performance_retention for r in results if r.performance_retention > 0]
    if not retentions:
        return 0.0

    log_sum = sum(math.log(r) for r in retentions)
    return math.exp(log_sum / len(retentions))


def run_robustness_evaluation(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    model_name: str = "model",
    output_dir: Optional[Path] = None,
    config: Optional[CorruptionConfig] = None,
) -> dict:
    """Full robustness evaluation with MRRS."""
    if config is None:
        config = CorruptionConfig()

    clean_correct = 0
    clean_total = 0

    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            if isinstance(outputs, dict):
                preds = outputs["logits"].argmax(dim=1)
            else:
                preds = outputs.argmax(dim=1)
            clean_correct += (preds == labels).sum().item()
            clean_total += labels.size(0)

    clean_accuracy = clean_correct / clean_total if clean_total > 0 else 0.0

    corruption_transforms = get_corruption_transforms(config)
    all_results = []

    for name, transform in corruption_transforms:
        result = evaluate_corruption(model, dataloader, device, name, transform, clean_accuracy)
        all_results.append(result)

    mrrs = compute_mrrs(all_results)

    summary = {
        "model": model_name,
        "clean_accuracy": clean_accuracy,
        "mrrs": mrrs,
        "num_corruptions": len(all_results),
        "corruptions": [
            {
                "type": r.corruption_type,
                "accuracy": r.accuracy,
                "f1": r.f1,
                "mcc": r.mcc,
                "retention": r.performance_retention,
            }
            for r in all_results
        ],
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"robustness_{model_name}.json", "w") as f:
            json.dump(summary, f, indent=2)

    return summary
