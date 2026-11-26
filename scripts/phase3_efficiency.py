"""Phase 3I — efficiency under a stated, reproducible protocol.

The paper reports 9,034,416 params / 34.46 MB / 48.63 ms / 20.6 FPS with no
hardware, no batch size, no warmup budget, and no statement of whether the
latency includes the 10 MC-dropout passes.  This script fixes all of that:

    protocol:  batch 1, fp32, 20 warm-up iterations (untimed),
               200 timed iterations, torch.cuda.synchronize() around each
               iteration, median and mean ± sd reported, plus p95.

Measured for every backbone in the paper's baseline table and for the TAMNet
variants, each as **single pass** and **full 10-pass MC-dropout** latency, so
the "near-real-time" claim can be checked either way.

Writes ``results/phase3/efficiency.json``.

    python -X utf8 scripts/phase3_efficiency.py
    python -X utf8 scripts/phase3_efficiency.py --timed 50 --warmup 10   # quick check
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets.classification import CLASS_NAMES  # noqa: E402
from models.factory import create_backbone_classifier  # noqa: E402
from models.shared_unet import SharedUNetMultiTask  # noqa: E402
from models.tumor_aware import TumorAwareConfig, TumorAwareMultiTaskNet  # noqa: E402

OUT = ROOT / "results" / "phase3" / "efficiency.json"

BACKBONES = ("densenet121", "resnet50", "efficientnet_v2_s", "convnext_tiny", "swin_t", "vit_b_16")
TAMNET_VARIANTS = {
    "tamnet_full": dict(),
    "tamnet_tonly": dict(fusion_mode="t"),
    "tamnet_concat": dict(use_attention_fusion=False),
    "tamnet_mc": dict(use_channel_attention=True),
}


def build(name: str) -> torch.nn.Module:
    if name in BACKBONES:
        return create_backbone_classifier(name, num_classes=len(CLASS_NAMES), pretrained=False, dropout=0.3)
    if name == "shared_unet":
        return SharedUNetMultiTask(num_classes=len(CLASS_NAMES), num_seg_classes=2, pretrained=False)
    cfg = TumorAwareConfig(
        backbone="densenet121",
        pretrained=False,
        num_classes=len(CLASS_NAMES),
        dropout=0.3,
        use_channel_attention=TAMNET_VARIANTS[name].get("use_channel_attention", False),
        seg_decoder="lightweight",
        use_attention_fusion=TAMNET_VARIANTS[name].get("use_attention_fusion", True),
        use_segmentation_head=True,
        use_uncertainty_head=True,
        fusion_mode=TAMNET_VARIANTS[name].get("fusion_mode", "g+t"),
        segmentation_classes=4,
    )
    return TumorAwareMultiTaskNet(cfg)


def params_and_size(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bytes_total = sum(p.numel() * p.element_size() for p in model.parameters())
    return {
        "params_total": int(total),
        "params_trainable": int(trainable),
        "params_millions": round(total / 1e6, 4),
        "weights_mib": round(bytes_total / (1024 ** 2), 2),
    }


def timeit(fn, warmup: int, timed: int, device) -> dict:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    samples = []
    for _ in range(timed):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return {
        "n": len(samples),
        "mean_ms": statistics.fmean(samples),
        "std_ms": statistics.stdev(samples) if len(samples) > 1 else None,
        "median_ms": statistics.median(samples),
        "p95_ms": samples[int(0.95 * (len(samples) - 1))],
        "min_ms": samples[0],
        "fps_from_mean": 1000.0 / statistics.fmean(samples),
        "fps_from_median": 1000.0 / statistics.median(samples),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--timed", type=int, default=200)
    ap.add_argument("--mc-passes", type=int, default=10)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--models", nargs="*", default=list(BACKBONES) + ["shared_unet"] + list(TAMNET_VARIANTS))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 3, args.image_size, args.image_size, device=device)
    if device.type == "cuda":
        print(f"device: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | "
              f"cuda {torch.version.cuda} | fp32 | batch 1")
    else:
        print("device: cpu (no CUDA available) - note this does not match the paper's setup")

    rows = {}
    for name in args.models:
        model = build(name).to(device).eval()
        info = params_and_size(model)

        with torch.no_grad():
            single = timeit(lambda: model(x), args.warmup, args.timed, device)

            def mc_forward():
                model.eval()
                for m in model.modules():
                    if isinstance(m, torch.nn.modules.dropout._DropoutNd):
                        m.train()
                with torch.no_grad():
                    for _ in range(args.mc_passes):
                        model(x)
                model.eval()

            mc = timeit(mc_forward, max(1, args.warmup // 4), args.timed, device)

        rows[name] = {
            **info,
            "single_pass": single,
            f"mc_{args.mc_passes}_pass": mc,
            "mc_pass_overhead_x": mc["median_ms"] / single["median_ms"],
        }
        print(f" {name:<18} {info['params_millions']:>8.3f} M  {info['weights_mib']:>7.2f} MiB  "
              f"single {single['median_ms']:>7.2f} ms  MC{args.mc_passes} {mc['median_ms']:>8.2f} ms  "
              f"({single['fps_from_median']:>5.1f} / {mc['fps_from_median']:>5.1f} FPS)")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "kind": "phase3_efficiency",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": {
            "batch_size": 1,
            "dtype": "fp32",
            "warmup_iterations": args.warmup,
            "timed_iterations": args.timed,
            "synchronize": "torch.cuda.synchronize() before and after every timed iteration",
            "mc_passes": args.mc_passes,
            "mc_mode": "model.eval() with only Dropout in train mode (D5 protocol)",
            "input": f"1x3x{args.image_size}x{args.image_size} random tensor",
            "note": "backbones are classification-only; TAMNet/shared_unet include the segmentation "
                    "and uncertainty heads, so their latency is not head-to-head with the backbones",
        },
        "hardware": {
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "models": rows,
        "paper_claims_to_check": {
            "params": 9034416,
            "model_size_mb": 34.46,
            "latency_ms": 48.63,
            "fps": 20.6,
            "latency_includes_mc_passes": "UNSTATED in the paper - compare both columns",
        },
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
