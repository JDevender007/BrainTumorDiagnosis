"""
Classify brain tumor images from a directory.

Usage:
    python -m scripts.predict_classification --input data/raw/classification/Test
    python -m scripts.predict_classification --input data/raw/classification/Test --backbone densenet121
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from datasets.classification import CLASS_NAMES
from engine.inference import InferenceConfig, load_classification_model, extract_state_dict
from models.factory import ModelConfig, create_classification_model, create_backbone_classifier, BACKBONE_CLASSIFIERS
from utils.device import get_device
from utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def predict_image(
    model: torch.nn.Module,
    image_path: Path,
    device: torch.device,
) -> dict:
    image = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, predicted = probs.max(dim=1)

    return {
        "file": image_path.name,
        "prediction": CLASS_NAMES[predicted.item()],
        "confidence": confidence.item(),
        "probabilities": {
            CLASS_NAMES[i]: probs[0, i].item()
            for i in range(len(CLASS_NAMES))
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify brain tumor images.")
    parser.add_argument("--input", type=str, required=True, help="Directory with images or a single image file")
    parser.add_argument("--backbone", type=str, default="densenet121", choices=["cnn2d", "densenet121", "resnet50", "efficientnet_v2_s", "convnext_tiny", "swin_t", "vit_b_16"])
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    setup_logging(level=logging.INFO)

    device = get_device()
    LOGGER.info("Device: %s", device)

    config = InferenceConfig(classification_backbone=args.backbone)

    if args.backbone == "cnn2d":
        model = create_classification_model(ModelConfig(model_type="classification", num_classes=4))
    else:
        model = create_backbone_classifier(backbone=args.backbone, num_classes=4, pretrained=False)

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = Path(f"checkpoints/classification/{args.backbone}/best.pt")

    if not checkpoint_path.exists():
        LOGGER.error("Checkpoint not found: %s", checkpoint_path)
        return 1

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = extract_state_dict(checkpoint)
    cleaned = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()

    LOGGER.info("Loaded model: %s from %s", args.backbone, checkpoint_path)

    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        image_paths = sorted(
            p for p in input_path.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        )
    else:
        LOGGER.error("Path not found: %s", input_path)
        return 1

    LOGGER.info("Found %d images to classify", len(image_paths))

    results = []
    for image_path in image_paths:
        result = predict_image(model, image_path, device)
        results.append(result)
        print(f"{result['file']:20s} -> {result['prediction']:12s} ({result['confidence']:.1%})")

    correct = sum(1 for r, p in zip(results, image_paths) if r["prediction"].lower() == p.parent.name.lower())
    total = len(results)
    if total > 0:
        print(f"\nAccuracy: {correct}/{total} ({correct/total:.1%})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
