"""
MRI-specific robustness transforms for medical imaging evaluation.

Implements realistic MRI artifacts:
1. Bias field (B1 inhomogeneity)
2. Gibbs ringing
3. Motion artifacts
4. Ghosting
5. Intensity non-uniformity
6. Resolution degradation
7. Slice dropout
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional


class BiasFieldArtifact:
    """
    Simulate MRI bias field (B1 inhomogeneity).

    Creates a smooth, low-frequency intensity variation across the image,
    typical of B1 field inhomogeneity in MRI.
    """

    def __init__(self, severity: float = 0.5):
        self.severity = severity

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        h, w = tensor.shape[-2], tensor.shape[-1]

        # Create smooth bias field using polynomial
        x = torch.linspace(-1, 1, w)
        y = torch.linspace(-1, 1, h)
        xx, yy = torch.meshgrid(x, y, indexing='ij')

        # Low-frequency polynomial
        bias = 1 + self.severity * (0.3 * xx + 0.2 * yy + 0.1 * xx * yy)

        # Apply to all channels
        if tensor.dim() == 3:
            bias = bias.unsqueeze(0)

        return torch.clamp(tensor * bias, 0, 1)


class GibbsRingingArtifact:
    """
    Simulate Gibbs ringing artifact.

    Gibbs ringing occurs near sharp intensity transitions (e.g., tissue boundaries)
    and appears as oscillatory patterns.
    """

    def __init__(self, severity: float = 0.3):
        self.severity = severity

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Convert to frequency domain
        if tensor.dim() == 3:
            tensor_2d = tensor
        else:
            tensor_2d = tensor.squeeze(0)

        # Apply FFT
        kspace = torch.fft.fft2(tensor_2d.float())

        # Create truncation mask (simulates k-space truncation)
        h, w = kspace.shape[-2], kspace.shape[-1]
        mask = torch.ones_like(kspace)

        # Truncate high frequencies
        center_h, center_w = h // 2, w // 2
        truncate_h = int(h * self.severity * 0.3)
        truncate_w = int(w * self.severity * 0.3)

        mask[center_h-truncate_h:center_h+truncate_h, :] = 0.5
        mask[:, center_w-truncate_w:center_w+truncate_w] = 0.5

        # Apply mask and inverse FFT
        kspace_truncated = kspace * mask
        result = torch.fft.ifft2(kspace_truncated).real

        return torch.clamp(result, 0, 1)


class MotionArtifact:
    """
    Simulate MRI motion artifact.

    Motion during acquisition causes blurring and ghosting along the
    phase-encoding direction.
    """

    def __init__(self, severity: float = 0.5):
        self.severity = severity

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Create motion blur kernel
        kernel_size = int(7 * self.severity) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Horizontal motion blur
        kernel = torch.zeros(kernel_size, kernel_size)
        center = kernel_size // 2
        for i in range(kernel_size):
            kernel[center, i] = 1.0 / kernel_size

        # Apply convolution
        channels = tensor.shape[0] if tensor.dim() == 3 else 1
        kernel_2d = kernel.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1).contiguous()

        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)

        input_t = tensor.unsqueeze(0) if tensor.dim() == 3 else tensor
        result = F.conv2d(input_t, kernel_2d, padding=kernel_size // 2, groups=channels)

        return result.squeeze(0)


class GhostingArtifact:
    """
    Simulate MRI ghosting artifact.

    Ghosting appears as replicated copies of the image, typically along
    the phase-encoding direction.
    """

    def __init__(self, severity: float = 0.3):
        self.severity = severity

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Create ghost copies
        shift = int(tensor.shape[-1] * self.severity * 0.2)
        ghost = torch.roll(tensor, shifts=shift, dims=-1)

        # Blend original and ghost
        return torch.clamp((1 - self.severity) * tensor + self.severity * ghost, 0, 1)


class IntensityNonUniformity:
    """
    Simulate intensity non-uniformity (INU).

    Common in MRI due to coil sensitivity variations.
    """

    def __init__(self, severity: float = 0.5):
        self.severity = severity

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        h, w = tensor.shape[-2], tensor.shape[-1]

        # Create smooth intensity variation
        x = torch.linspace(0, 1, w)
        y = torch.linspace(0, 1, h)
        xx, yy = torch.meshgrid(x, y, indexing='ij')

        # Quadratic variation
        inu = 1 + self.severity * ((xx - 0.5) ** 2 + (yy - 0.5) ** 2)

        if tensor.dim() == 3:
            inu = inu.unsqueeze(0)

        return torch.clamp(tensor * inu, 0, 1)


class ResolutionDegradation:
    """
    Simulate resolution degradation.

    Downsample and upsample to simulate lower resolution acquisition.
    """

    def __init__(self, severity: float = 0.5):
        self.severity = severity

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Determine downsample factor
        factor = 1 + int(4 * self.severity)

        h, w = tensor.shape[-2], tensor.shape[-1]
        new_h = max(1, h // factor)
        new_w = max(1, w // factor)

        # Downsample
        input_t = tensor.unsqueeze(0) if tensor.dim() == 3 else tensor
        downsampled = F.interpolate(input_t, size=(new_h, new_w), mode='bilinear', align_corners=False)

        # Upsample back
        upsampled = F.interpolate(downsampled, size=(h, w), mode='bilinear', align_corners=False)

        return upsampled.squeeze(0)


class SliceDropout:
    """
    Simulate slice dropout in 3D MRI.

    Randomly drops slices from a 3D volume (simulated as channel dropout
    for 2D images).
    """

    def __init__(self, severity: float = 0.3):
        self.severity = severity

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # For 2D images, simulate by zeroing out random spatial regions
        mask = torch.ones_like(tensor)
        h, w = tensor.shape[-2], tensor.shape[-1]

        # Random rectangular region to drop
        drop_h = int(h * self.severity * 0.5)
        drop_w = int(w * self.severity * 0.5)

        start_h = torch.randint(0, h - drop_h + 1, (1,)).item()
        start_w = torch.randint(0, w - drop_w + 1, (1,)).item()

        mask[:, start_h:start_h+drop_h, start_w:start_w+drop_w] = 0

        return tensor * mask


def get_mri_corruption_transforms() -> list[tuple[str, object]]:
    """
    Get all MRI-specific corruption transforms.

    Returns:
        list of (name, transform) tuples
    """
    transforms = []

    # Bias field
    for severity in [0.2, 0.4, 0.6, 0.8]:
        transforms.append((f"bias_field_{severity}", BiasFieldArtifact(severity)))

    # Gibbs ringing
    for severity in [0.1, 0.2, 0.3, 0.4]:
        transforms.append((f"gibbs_ringing_{severity}", GibbsRingingArtifact(severity)))

    # Motion
    for severity in [0.2, 0.4, 0.6, 0.8]:
        transforms.append((f"motion_{severity}", MotionArtifact(severity)))

    # Ghosting
    for severity in [0.1, 0.2, 0.3]:
        transforms.append((f"ghosting_{severity}", GhostingArtifact(severity)))

    # Intensity non-uniformity
    for severity in [0.2, 0.4, 0.6, 0.8]:
        transforms.append((f"intensity_nu_{severity}", IntensityNonUniformity(severity)))

    # Resolution degradation
    for severity in [0.2, 0.4, 0.6, 0.8]:
        transforms.append((f"resolution_deg_{severity}", ResolutionDegradation(severity)))

    # Slice dropout
    for severity in [0.1, 0.2, 0.3]:
        transforms.append((f"slice_dropout_{severity}", SliceDropout(severity)))

    return transforms
