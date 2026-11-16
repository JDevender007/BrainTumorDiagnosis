"""
BrainTumorDiagnosis explainability module.

Grad-CAM heatmaps for the classification backbones:

    cnn2d, densenet121, resnet50, efficientnet_v2_s,
    convnext_tiny, swin_t

Usage (from the project root):

    python -m evaluation.explainability --backbone densenet121 \
        --image data/raw/classification/test/glioma/Tr-gl_0010.jpg \
        --output outputs/figures/gradcam_gl_0010.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from types import MethodType
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.factory import (
    ModelConfig,
    create_backbone_classifier,
    create_classification_model,
)


LOGGER = logging.getLogger(__name__)


CLASS_NAMES = (
    "Glioma",
    "Meningioma",
    "Pituitary",
    "No Tumor",
)


IMAGE_SIZE = (224, 224)

SUPPORTED_BACKBONES = (
    "cnn2d",
    "densenet121",
    "resnet50",
    "efficientnet_v2_s",
    "convnext_tiny",
    "swin_t",
    "vit_b_16",
)

TARGET_LAYERS: dict[str, str] = {
    "cnn2d": "layer4",
    "densenet121": "features",
    "resnet50": "layer4",
    "efficientnet_v2_s": "features",
    "convnext_tiny": "features",
    "swin_t": "features.4",
    "vit_b_16": "encoder.ln",
}


def create_test_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def _get_layer(
    model: torch.nn.Module,
    path: str,
) -> torch.nn.Module:
    module: Any = model

    for part in path.split("."):
        module = getattr(module, part)

    return module


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Records the forward activations and backward gradients of a
    target layer and produces a spatial relevance heatmap.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module,
    ) -> None:
        self.model = model

        self.activations: torch.Tensor | None = None

        self.gradients: torch.Tensor | None = None

        self._forward_handle = target_layer.register_forward_hook(
            self._save_activation
        )

        self._backward_handle = target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(
        self,
        module: torch.nn.Module,
        input: Any,
        output: torch.Tensor,
    ) -> None:
        self.activations = output.detach()

    def _save_gradient(
        self,
        module: torch.nn.Module,
        grad_input: Any,
        grad_output: Any,
    ) -> None:
        self.gradients = grad_output[0].detach()

    def generate_heatmap(
        self,
        image: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Generate a Grad-CAM heatmap for one input image.

        Args:
            image: (1, 3, H, W) tensor on the model device.
            target_class: Class index to explain. Defaults to the
                predicted class.

        Returns:
            (heatmap, probabilities, class_index)
        """

        self.activations = None
        self.gradients = None

        output = self.model(image)

        if isinstance(output, dict):
            logits = output.get("classification")
            if logits is None:
                logits = output.get("logits")
            if logits is None:
                logits = output.get("out")
            if logits is not None:
                output = logits

        if isinstance(output, (tuple, list)):
            output = output[0]

        probabilities = torch.softmax(
            output.detach().float(),
            dim=1,
        )[0].cpu().numpy()

        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0

        output.backward(gradient=one_hot)

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks did not fire. Check the target layer path."
            )

        activations = self.activations
        gradients = self.gradients

        if activations.ndim == 3:
            # ViT-style (B, N, C) activations: the final head selects
            # only the classification token, so the patch-token
            # gradients are zero. The gradient of the CLS token is
            # used as channel weights instead (CLS-GradCAM).
            token_weights = gradients[:, 0, :]

            patch_tokens = activations[:, 1:, :]

            n = patch_tokens.shape[1]

            side = int(round(n ** 0.5))

            if side * side != n:
                raise ValueError(
                    f"Cannot reshape {n} ViT tokens into a square grid."
                )

            token_heatmap = torch.einsum(
                "bc,bnc->bn",
                token_weights,
                patch_tokens,
            )

            heatmap = F.relu(
                token_heatmap.reshape(1, side, side)
            )
        else:
            weights = gradients.mean(
                dim=(2, 3),
                keepdim=True,
            )

            heatmap = F.relu(
                (weights * activations).sum(dim=1)
            )

        heatmap = heatmap.squeeze(0)

        if heatmap.ndim != 2:
            heatmap = F.interpolate(
                heatmap.unsqueeze(0).unsqueeze(0),
                size=IMAGE_SIZE,
                mode="bilinear",
                align_corners=False,
            ).squeeze()

        heatmap = F.interpolate(
            heatmap.unsqueeze(0).unsqueeze(0),
            size=IMAGE_SIZE,
            mode="bilinear",
            align_corners=False,
        )

        heatmap_np = heatmap.squeeze().cpu().numpy()

        positive = heatmap_np[heatmap_np > 0]

        if positive.size > 0:
            heatmap_np = heatmap_np / positive.max()
        else:
            heatmap_np = np.zeros_like(heatmap_np)

        self.remove_hooks()

        return heatmap_np, probabilities, target_class

    def remove_hooks(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()


def _patch_densenet_forward(model: torch.nn.Module) -> torch.nn.Module:
    """Replace torchvision DenseNet's in-place final ReLU.

    DenseNet applies ``F.relu(features, inplace=True)`` on the hooked
    module output, which conflicts with the Grad-CAM backward hook.
    A non-in-place forward avoids the autograd view error.
    """

    def _forward(
        module: torch.nn.Module,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        features = module.features(input_tensor)

        output = F.relu(features)

        output = F.adaptive_avg_pool2d(
            output,
            (1, 1),
        )

        output = torch.flatten(output, 1)

        return module.classifier(output)

    model.forward = MethodType(_forward, model)

    return model


def load_model(
    backbone: str,
    device: torch.device,
    checkpoint_path: Path,
) -> torch.nn.Module:
    if backbone == "cnn2d":
        model = create_classification_model(
            ModelConfig(
                model_type="classification",
                in_channels=3,
                num_classes=4,
                classification_input_size=IMAGE_SIZE,
                pretrained=False,
            )
        )
    else:
        model = create_backbone_classifier(
            backbone=backbone,
            num_classes=4,
            pretrained=False,
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint
        )
    else:
        state_dict = checkpoint

    cleaned = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }

    model.load_state_dict(
        cleaned,
        strict=False,
    )

    if backbone == "densenet121":
        _patch_densenet_forward(model)

    model.to(device)
    model.eval()

    return model


def explain_image(
    model: torch.nn.Module,
    backbone: str,
    image: torch.Tensor,
    device: torch.device,
    target_class: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Generate a Grad-CAM heatmap for one image tensor."""

    if backbone == "densenet121":
        model = _patch_densenet_forward(model)

    target_layer = _get_layer(
        model,
        TARGET_LAYERS[backbone],
    )

    cam = GradCAM(
        model,
        target_layer,
    )

    image = image.to(device)

    image.requires_grad_(True)

    return cam.generate_heatmap(
        image,
        target_class,
    )


def save_overlay(
    image_path: str | Path,
    heatmap: np.ndarray,
    output_path: str | Path,
    class_name: str,
    probability: float,
) -> Path:
    """Save the original MRI, heatmap, and overlay side by side."""

    image_path = Path(image_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    original = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)

    original_np = np.asarray(original).astype(np.float32) / 255.0

    colormap = plt.get_cmap("jet")

    colored = colormap(heatmap)[:, :, :3]

    overlay = 0.6 * original_np + 0.4 * colored

    figure, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    axes[0].imshow(original_np)
    axes[0].set_title("Original MRI")
    axes[0].axis("off")

    axes[1].imshow(original_np)
    axes[1].imshow(
        colored,
        alpha=0.45,
    )
    axes[1].set_title("Grad-CAM Overlay")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(
        f"{class_name} ({probability * 100:.1f}%)"
    )
    axes[2].axis("off")

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM heatmaps for classification models."
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="densenet121",
        choices=SUPPORTED_BACKBONES,
        help="Model backbone.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path (defaults to checkpoints/classification/{backbone}/best.pt).",
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to a test MRI image.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG path (defaults next to the input image).",
    )

    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args = build_parser().parse_args()

    image_path = Path(args.image)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    checkpoint_path = Path(
        args.checkpoint
        or f"checkpoints/classification/{args.backbone}/best.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LOGGER.info("Device: %s", device)

    model = load_model(
        args.backbone,
        device,
        checkpoint_path,
    )

    transform = create_test_transform()

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image).unsqueeze(0)

    heatmap, probabilities, class_index = explain_image(
        model,
        args.backbone,
        image_tensor,
        device,
    )

    class_name = CLASS_NAMES[class_index]

    probability = float(probabilities[class_index])

    output_path = Path(
        args.output or image_path.with_name(f"gradcam_{image_path.stem}.png")
    )

    save_overlay(
        image_path,
        heatmap,
        output_path,
        class_name,
        probability,
    )

    LOGGER.info("Prediction: %s (%.2f%%)", class_name, probability * 100)

    LOGGER.info("Heatmap saved: %s", output_path)


if __name__ == "__main__":
    main()