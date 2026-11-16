"""Generate Grad-CAM figures for all backbones on one example per class."""
import subprocess
import sys
from pathlib import Path

BACKBONES = [
    "densenet121", "resnet50", "efficientnet_v2_s",
    "convnext_tiny", "swin_t", "vit_b_16",
]

CLASS_DIRS = {
    "Glioma": "data/raw/classification/test/glioma",
    "Meningioma": "data/raw/classification/test/meningioma",
    "Pituitary": "data/raw/classification/test/pituitary",
    "No Tumor": "data/raw/classification/test/no tumor",
}

output_dir = Path("outputs/gradcam")
output_dir.mkdir(parents=True, exist_ok=True)

for cls_name, cls_dir in CLASS_DIRS.items():
    imgs = sorted(Path(cls_dir).glob("*.jpg"))
    if not imgs:
        imgs = sorted(Path(cls_dir).glob("*.png"))
    if not imgs:
        print(f"No images found for {cls_name}")
        continue
    img = imgs[0]
    for backbone in BACKBONES:
        out_path = output_dir / f"{backbone}_{cls_name}.png"
        cmd = [
            sys.executable, "-m", "evaluation.explainability",
            "--backbone", backbone,
            "--image", str(img),
            "--output", str(out_path),
        ]
        print(f"Generating: {backbone} {cls_name}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr[-500:]}")
        else:
            print(f"  Saved: {out_path}")

print("Done.")
