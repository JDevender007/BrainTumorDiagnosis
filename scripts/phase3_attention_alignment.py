"""Phase 3H — is the spatial attention actually on the tumour?

The paper promises an "interpretability analysis" but only shows qualitative
Grad-CAM pictures.  Two quantitative questions are answerable from the
checkpoints already on disk:

1. **Attention alignment.**  TAMNet's spatial attention ``M_s`` produces a
   7×7 sigmoid map per image.  Does that map put more mass inside the
   radiologist's tumour mask than (a) a uniform map and (b) DenseNet-121
   Grad-CAM on the same slices?  Reported as *mass in mask* and *pointing
   game hit rate* (argmax of the map falls inside the mask).
2. **Fusion weights.**  ``AttentionFusion`` emits a 2-way softmax
       z = α_g · g + α_t · t
   and the paper's claim ("tumour-aware fusion shifts weight to the tumour
   feature") is directly measurable: report mean α_t overall and per class.

Everything is computed on the BRISC 2025 test split (the only split with
aligned masks) so the mask reference is real, not a proxy.

    python -X utf8 scripts/phase3_attention_alignment.py --dry-run
    python -X utf8 scripts/phase3_attention_alignment.py

Outputs ``results/phase3/attention_alignment.json`` and the per-slice raw
records inside it.  No numbers are hard-coded; if a map cannot be extracted the
run is reported as ``MISSING`` rather than skipped silently.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets.classification import (  # noqa: E402
    CLASS_NAMES,
    BrainTumorClassificationDataset,
)
from datasets.normalization import build_classification_transform  # noqa: E402
from scripts.phase3_evaluate import build_model, sha256_file  # noqa: E402
from scripts.phase3_train import load_brisc_split  # noqa: E402
from evaluation.explainability import _patch_densenet_forward  # noqa: E402

RUNS = ROOT / "results" / "phase3" / "runs"
OUT = ROOT / "results" / "phase3" / "attention_alignment.json"

TAMNET_RUN = "tamnet_clean_s42"
BASELINE_RUN = "densenet121_clean_s42"
MASK_RUN = "tamnet_brisc_s42+ls0.25+sc2"


def upsample_normalise(cam: torch.Tensor, size: int = 224) -> torch.Tensor:
    """(B,1,h,w) or (B,h,w) -> (B,size,size) non-negative, sums to 1 per image."""
    if cam.dim() == 3:
        cam = cam.unsqueeze(1)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(size, size), mode="bilinear", align_corners=False)
    cam = cam - cam.amin(dim=(1, 2, 3), keepdim=True)
    denom = cam.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
    return cam / denom


def mass_in_mask(cam: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (cam * mask.unsqueeze(1).float()).sum(dim=(1, 2, 3))


def pointing_hit(cam: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    flat = cam.flatten(1).argmax(dim=1)
    m = mask.flatten(1).float()
    return m.gather(1, flat.unsqueeze(1)).squeeze(1)


# --------------------------------------------------------------------------- #
# TAMNet: spatial attention + fusion weights
# --------------------------------------------------------------------------- #
@torch.no_grad()
def tamnet_maps(model, x: torch.Tensor):
    feats = model.encode(x)          # (global_feat, feature_map)
    feature_map = feats[1] if isinstance(feats, tuple) else feats
    pool = model.tumor_pool
    avg = feature_map.mean(dim=1, keepdim=True)
    mx = feature_map.amax(dim=1, keepdim=True)
    att = torch.sigmoid(pool.attention.conv(torch.cat([avg, mx], dim=1)))  # (B,1,7,7)
    return att


class FusionWeightCatcher:
    """Capture the 2-way softmax produced by AttentionFusion."""

    def __init__(self):
        self.weights: list[torch.Tensor] = []
        self.handle = None

    def __enter__(self):
        return self

    def attach(self, model):
        if getattr(model, "fusion", None) is not None:
            self.handle = model.fusion.attention.register_forward_hook(self._hook)
        return self

    def _hook(self, module, inputs, output):
        if isinstance(output, torch.Tensor) and output.dim() == 2 and output.shape[1] == 2:
            self.weights.append(output.detach().cpu())

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()


# --------------------------------------------------------------------------- #
# DenseNet-121 Grad-CAM baseline
# --------------------------------------------------------------------------- #
class GradCAM:
    def __init__(self, net, target_layer):
        self.acts = None
        self.grads = None
        self.h1 = target_layer.register_forward_hook(self._fwd)
        self.h2 = target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, module, inputs, output):
        self.acts = output
        self.acts.retain_grad()

    def _bwd(self, module, grad_in, grad_out):
        self.grads = grad_out[0].detach()

    def __call__(self, net, x, class_idx=None):
        net.zero_grad(set_to_none=True)
        out = net(x)
        probs = out["probabilities"] if isinstance(out, dict) else torch.softmax(out, 1)
        cls = probs.argmax(dim=1) if class_idx is None else class_idx
        score = probs.gather(1, cls.unsqueeze(1)).sum()
        score.backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)     # (B,C,1,1)
        cam = (w * self.acts).sum(dim=1, keepdim=True)    # (B,1,h,w)
        del self.acts, self.grads
        self.acts = None
        self.grads = None
        return cam

    def close(self):
        self.h1.remove()
        self.h2.remove()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if OUT.is_file() and not args.force:
        print(f"{OUT.relative_to(ROOT)} exists - use --force to recompute")
    if args.dry_run:
        print("dry run: would evaluate")
        for name in (TAMNET_RUN, MASK_RUN, BASELINE_RUN):
            print(f"  {name}")
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = [s for s in load_brisc_split()["test"] if s.mask_path is not None]
    tumor_samples = []
    ds_full = BrainTumorClassificationDataset(
        samples, transform=build_classification_transform((224, 224))
    )
    for i, s in enumerate(samples):
        item = ds_full[i]
        if int(item[2].sum()) > 0:
            tumor_samples.append(i)
    print(f"BRISC test: {len(samples)} slices, {len(tumor_samples)} with a non-empty tumour mask")

    x_all, y_all, m_all = [], [], []
    for i in tumor_samples:
        img, lab, msk = ds_full[i]
        x_all.append(img)
        y_all.append(int(lab))
        m_all.append((msk > 0).float())
    # host-resident: a full BRISC test tensor does not fit on a 4 GiB card
    # together with activations, and spilling to shared memory is ~18x slower
    x = torch.stack(x_all)
    y = np.asarray(y_all)
    masks = torch.stack(m_all)

    records: dict[str, list] = {"tamnet_attention": [], "densenet121_gradcam": [], "uniform": []}
    fusion_weights: list[np.ndarray] = []
    missing: list[str] = []

    # ---------------- TAMNet spatial attention ----------------
    d = RUNS / TAMNET_RUN
    if (d / "best.pt").is_file():
        cfg = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
        model = build_model(cfg)
        model.load_state_dict(torch.load(d / "best.pt", map_location="cpu", weights_only=False)["model_state_dict"])
        model = model.to(device).eval()
        t0 = time.time()
        catcher = FusionWeightCatcher().attach(model)
        for i in range(0, len(x), args.batch_size):
            xb = x[i:i + args.batch_size].to(device)
            mb = masks[i:i + args.batch_size].to(device)
            cam = upsample_normalise(tamnet_maps(model, xb))
            mass = mass_in_mask(cam, mb)
            hit = pointing_hit(cam, mb)
            for k in range(len(mb)):
                records["tamnet_attention"].append({
                    "slice": tumor_samples[i + k], "class": CLASS_NAMES[y[i + k]],
                    "mass_in_mask": float(mass[k]), "hit": int(hit[k]),
                    "mask_fraction": float(mb[k].mean()),
                })
            if getattr(model, "fusion", None) is not None:
                model(xb)
        for w in catcher.weights:
            fusion_weights.extend(w.numpy().tolist())
        catcher.__exit__(None, None, None)
        print(f"  TAMNet attention: {len(records['tamnet_attention'])} slices ({time.time() - t0:.0f}s)")
    else:
        missing.append(f"{TAMNET_RUN} (no best.pt)")

    # uniform reference
    for i in range(len(tumor_samples)):
        uni = torch.full((1, 224, 224), 1.0 / (224 * 224), device=device)
        mb = masks[i:i + 1].to(device)
        records["uniform"].append({
            "slice": tumor_samples[i], "class": CLASS_NAMES[y[i]],
            "mass_in_mask": float(mass_in_mask(uni, mb)[0]),
            "hit": int(pointing_hit(uni, mb)[0]),
            "mask_fraction": float(mb.mean()),
        })

    # ---------------- DenseNet-121 Grad-CAM ----------------
    d = RUNS / BASELINE_RUN
    if (d / "best.pt").is_file():
        cfg = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
        net = build_model(cfg)
        net.load_state_dict(torch.load(d / "best.pt", map_location="cpu", weights_only=False)["model_state_dict"])
        net = net.to(device).eval()
        # build_model returns DictWrapper(backbone=...) for every torchvision
        # baseline; the DenseNet itself — and its .features — live one level
        # down, which is why this probe used to record "no .features module".
        inner = (net.backbone if hasattr(net, "backbone")
                 else net.model if hasattr(net, "model") else net)
        target = getattr(inner, "features", None)
        if target is None:
            missing.append(f"{BASELINE_RUN} (no .features module for Grad-CAM)")
        else:
            # DenseNet's final F.relu(..., inplace=True) sits on the hooked
            # output and the backward hook rejects the view; the shared patch
            # makes that ReLU non-inplace with identical numerics.
            inner = _patch_densenet_forward(inner)
            cam_gen = GradCAM(inner, target)
            t0 = time.time()
            try:
                for i in range(0, len(x), args.batch_size):
                    xb = x[i:i + args.batch_size].to(device)
                    mb = masks[i:i + args.batch_size].to(device)
                    cam = upsample_normalise(cam_gen(net, xb))
                    mass = mass_in_mask(cam, mb)
                    hit = pointing_hit(cam, mb)
                    for k in range(len(mb)):
                        records["densenet121_gradcam"].append({
                            "slice": tumor_samples[i + k], "class": CLASS_NAMES[y[i + k]],
                            "mass_in_mask": float(mass[k]), "hit": int(hit[k]),
                            "mask_fraction": float(mb[k].mean()),
                        })
            finally:
                cam_gen.close()
            print(f"  DenseNet-121 Grad-CAM: {len(records['densenet121_gradcam'])} slices ({time.time() - t0:.0f}s)")
    else:
        missing.append(f"{BASELINE_RUN} (no best.pt)")

    def summarize(rows: list[dict]) -> dict | None:
        if not rows:
            return None
        mass = [r["mass_in_mask"] for r in rows]
        hits = [r["hit"] for r in rows]
        return {
            "n": len(rows),
            "mass_in_mask_mean": float(np.mean(mass)),
            "mass_in_mask_std": float(np.std(mass, ddof=1)) if len(mass) > 1 else None,
            "mass_in_mask_median": float(np.median(mass)),
            "pointing_hit_rate": float(np.mean(hits)),
            "mask_fraction_mean": float(np.mean([r["mask_fraction"] for r in rows])),
            "lift_over_mask_fraction": float(np.mean(mass) / max(np.mean([r["mask_fraction"] for r in rows]), 1e-9)),
        }

    def by_class(rows: list[dict]) -> dict:
        out = {}
        for name in CLASS_NAMES:
            sub = [r for r in rows if r["class"] == name]
            if sub:
                out[name] = summarize(sub)
        return out

    # paired TAMNet - GradCAM over identical slices
    paired = None
    a = {r["slice"]: r for r in records["tamnet_attention"]}
    b = {r["slice"]: r for r in records["densenet121_gradcam"]}
    common = sorted(set(a) & set(b))
    if common:
        deltas = np.array([a[i]["mass_in_mask"] - b[i]["mass_in_mask"] for i in common])
        rng = np.random.default_rng(12345)
        boot = np.array([deltas[rng.integers(0, len(deltas), len(deltas))].mean() for _ in range(1000)])
        paired = {
            "n": len(common),
            "delta_mass_mean": float(deltas.mean()),
            "ci95_low": float(np.percentile(boot, 2.5)),
            "ci95_high": float(np.percentile(boot, 97.5)),
            "significant": bool(np.percentile(boot, 2.5) > 0 or np.percentile(boot, 97.5) < 0),
        }

    fw = np.asarray(fusion_weights) if fusion_weights else None
    payload = {
        "kind": "phase3_attention_alignment",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "BRISC 2025 official test split, tumour slices only (GT mask non-empty)",
        "definitions": {
            "mass_in_mask": "sum of the normalised (sum=1) upsampled map inside the GT mask; "
                            "uniform baseline equals mask_fraction",
            "pointing_hit": "argmax of the map lies inside the GT mask",
            "lift_over_mask_fraction": "mass_in_mask / mask area, >1 = better than uniform",
        },
        "runs": {"tamnet": TAMNET_RUN, "baseline": BASELINE_RUN, "seg_trained": MASK_RUN},
        "tamnet_attention": summarize(records["tamnet_attention"]),
        "densenet121_gradcam": summarize(records["densenet121_gradcam"]),
        "uniform": summarize(records["uniform"]),
        "tamnet_attention_by_class": by_class(records["tamnet_attention"]),
        "densenet121_gradcam_by_class": by_class(records["densenet121_gradcam"]),
        "paired_tamnet_minus_gradcam": paired,
        "fusion_weights": {
            "n": int(fw.shape[0]) if fw is not None else 0,
            "alpha_global_mean": float(fw[:, 0].mean()) if fw is not None else None,
            "alpha_tumor_mean": float(fw[:, 1].mean()) if fw is not None else None,
            "alpha_tumor_std": float(fw[:, 1].std(ddof=1)) if fw is not None and len(fw) > 1 else None,
            "note": "2-way softmax over (global, tumour) features; see models/tumor_aware.py AttentionFusion",
        },
        "missing": missing,
        "provenance": {
            "script": "scripts/phase3_attention_alignment.py",
            "checkpoint_sha256": {
                n: sha256_file(RUNS / n / "best.pt")
                for n in (TAMNET_RUN, BASELINE_RUN) if (RUNS / n / "best.pt").is_file()
            },
        },
        "per_slice": records,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=" * 88)
    print(" Phase 3H attention alignment (BRISC test, tumour slices)")
    print("=" * 88)
    for key in ("tamnet_attention", "densenet121_gradcam", "uniform"):
        s = payload[key]
        if not s:
            print(f" {key:<24} MISSING")
            continue
        print(f" {key:<24} mass {s['mass_in_mask_mean']:.4f} (mask area {s['mask_fraction_mean']:.4f}, "
              f"lift {s['lift_over_mask_fraction']:.2f}x) pointing {s['pointing_hit_rate']:.3f} n={s['n']}")
    if paired:
        print(f" paired TAMNet-GradCAM delta {paired['delta_mass_mean']:+.4f} "
              f"[{paired['ci95_low']:+.4f}, {paired['ci95_high']:+.4f}] "
              f"{'significant' if paired['significant'] else 'not distinguishable'}")
    if fw is not None:
        print(f" fusion weights: alpha_global {fw[:, 0].mean():.4f} | alpha_tumor {fw[:, 1].mean():.4f}")
    print(f" wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
