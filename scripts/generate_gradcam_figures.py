"""
Generate Grad-CAM figures for the paper:
- Baseline DenseNet121 standalone
- TumorAwareMultiTaskNet (proposed)
One example per class, side-by-side comparison.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.explainability import GradCAM, create_test_transform, CLASS_NAMES, TARGET_LAYERS, _get_layer, _patch_densenet_forward
from models.factory import create_backbone_classifier
from models.tumor_aware import TumorAwareMultiTaskNet, TumorAwareConfig

CLASS_DIRS = {
    "Glioma": "data/raw/classification/test/glioma",
    "Meningioma": "data/raw/classification/test/meningioma",
    "No Tumor": "data/raw/classification/test/no tumor",
    "Pituitary": "data/raw/classification/test/pituitary",
}
OUTPUT_DIR = Path("outputs/research_experiments/figures")


def get_sample_images() -> dict[str, Path]:
    samples = {}
    for cls_name, cls_dir in CLASS_DIRS.items():
        p = Path(cls_dir)
        imgs = sorted(p.glob("*.jpg")) or sorted(p.glob("*.png"))
        if imgs:
            samples[cls_name] = imgs[0]
    return samples


def generate_gradcam_grid():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = create_test_transform()
    samples = get_sample_images()

    baseline_model = create_backbone_classifier("densenet121", num_classes=4, pretrained=False)
    baseline_ckpt = torch.load("checkpoints/classification/densenet121/best.pt", map_location=device, weights_only=False)
    if isinstance(baseline_ckpt, dict):
        sd = baseline_ckpt.get("model_state_dict", baseline_ckpt.get("state_dict", baseline_ckpt))
    else:
        sd = baseline_ckpt
    cleaned = {k.removeprefix("module."): v for k, v in sd.items()}
    baseline_model.load_state_dict(cleaned, strict=False)
    baseline_model.to(device).eval()
    _patch_densenet_forward(baseline_model)

    proposed_model = TumorAwareMultiTaskNet(TumorAwareConfig(backbone="densenet121"))
    proposed_ckpt = torch.load("outputs/research_experiments/proposed_model/best.pt", map_location=device, weights_only=False)
    proposed_model.load_state_dict(proposed_ckpt["model_state_dict"])
    proposed_model.to(device).eval()
    _patch_densenet_forward(proposed_model.encoder)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    baseline_target = _get_layer(baseline_model, TARGET_LAYERS["densenet121"])
    proposed_target = _get_layer(proposed_model.encoder, TARGET_LAYERS["densenet121"])

    for col, (cls_name, img_path) in enumerate(samples.items()):
        image = Image.open(img_path).convert("RGB")

        image_tensor = transform(image).unsqueeze(0).to(device)
        image_tensor.requires_grad_(True)
        cam_b = GradCAM(baseline_model, baseline_target)
        hm_b, probs_b, cls_b = cam_b.generate_heatmap(image_tensor)

        image_tensor2 = transform(image).unsqueeze(0).to(device)
        image_tensor2.requires_grad_(True)
        cam_p = GradCAM(proposed_model.encoder, proposed_target)
        hm_p, probs_p_encoder, cls_p_encoder = cam_p.generate_heatmap(image_tensor2)

        with torch.no_grad():
            out = proposed_model(transform(image).unsqueeze(0).to(device))
            probs_p = out["probabilities"].cpu().numpy()[0]
            cls_p = int(np.argmax(probs_p))

        original_np = np.asarray(image.resize((224, 224))).astype(np.float32) / 255.0
        colormap = plt.get_cmap("jet")

        for row, (hm, probs, cls_idx, label) in enumerate([
            (hm_b, probs_b, cls_b, "Baseline"),
            (hm_p, probs_p, cls_p, "Proposed"),
        ]):
            colored = colormap(hm)[:, :, :3]
            overlay = 0.6 * original_np + 0.4 * colored
            axes[row, col].imshow(overlay)
            axes[row, col].set_title(f"{label}: {CLASS_NAMES[cls_idx]} ({probs[cls_idx]*100:.1f}%)", fontsize=9)
            axes[row, col].axis("off")

    axes[0, 0].set_ylabel("Baseline", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Proposed", fontsize=12, fontweight="bold")

    plt.suptitle("Grad-CAM Localization: Baseline vs Proposed", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig11_gradcam.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Grad-CAM comparison saved to fig11_gradcam.png")


if __name__ == "__main__":
    generate_gradcam_grid()
