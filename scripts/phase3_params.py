"""Parameter counts for every Phase 3 configuration (CPU only, no weights).

The paper's 9.03 M claim was verified in Phase 2 for the full TAMNet config. The
baseline table needs the same treatment for every backbone, measured the same
way: instantiate with ``pretrained=False`` and sum ``numel()`` over parameters
with ``requires_grad`` (trainable) plus the total for reference.

CPU only - safe to run while trainings occupy the GPU.

Usage:
    python -X utf8 scripts/phase3_params.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets.classification import CLASS_NAMES  # noqa: E402
from models.factory import create_backbone_classifier  # noqa: E402
from models.tumor_aware import TumorAwareConfig, TumorAwareMultiTaskNet  # noqa: E402

OUT = ROOT / "results" / "phase3" / "params_table.json"

BACKBONES = [
    "densenet121",
    "resnet50",
    "efficientnet_v2_s",
    "convnext_tiny",
    "swin_t",
    "vit_b_16",
]

TAMNET_VARIANTS = {
    "tamnet_full": dict(),
    "tamnet_tonly": dict(fusion_mode="t"),
    "tamnet_concat": dict(use_attention_fusion=False),
    "tamnet_mc": dict(use_channel_attention=True),
}


def count(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def main() -> int:
    rows = {}

    for name in BACKBONES:
        m = create_backbone_classifier(name, num_classes=len(CLASS_NAMES),
                                       pretrained=False, dropout=0.3)
        rows[name] = {"kind": "backbone", **count(m)}
        del m

    for label, kwargs in TAMNET_VARIANTS.items():
        cfg = TumorAwareConfig(
            backbone="densenet121",
            pretrained=False,
            num_classes=len(CLASS_NAMES),
            dropout=0.3,
            use_channel_attention=kwargs.get("use_channel_attention", False),
            seg_decoder="lightweight",
            use_attention_fusion=kwargs.get("use_attention_fusion", True),
            use_segmentation_head=True,
            use_uncertainty_head=True,
            fusion_mode=kwargs.get("fusion_mode", "g+t"),
            segmentation_classes=4,
        )
        m = TumorAwareMultiTaskNet(cfg)
        rows[label] = {"kind": "tamnet", **count(m)}
        del m

    payload = {
        "kind": "phase3_params",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": "instantiate with pretrained=False, sum numel() over requires_grad",
        "note": "TAMNet variants differ in fusion/attention flags only; the shared "
                "backbone is densenet121 as in the paper",
        "params": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    width = max(len(k) for k in rows)
    for name, v in sorted(rows.items(), key=lambda kv: kv[1]["total"]):
        print(f"{name:<{width}}  {v['trainable']:>12,} trainable  "
              f"{v['total']:>12,} total  {v['total'] / 1e6:>6.2f} M")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
